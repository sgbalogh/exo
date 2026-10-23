use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow};
use futures::{StreamExt, future::BoxFuture, stream::FuturesUnordered};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, Command};
use tokio::sync::{Notify, mpsc};

use super::types::{AdapterAttachment, AdapterConfig};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerCommand {
    SendMessage {
        id: String,
        #[serde(default)]
        target: Option<String>,
        text: String,
        #[serde(default)]
        attachments: Vec<AdapterAttachment>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerEvent {
    Connected {
        #[serde(default)]
        subject: Option<String>,
        #[serde(default)]
        metadata: Value,
    },
    Message {
        target: String,
        #[serde(default)]
        sender: Option<String>,
        text: String,
        #[serde(default)]
        message_id: Option<String>,
        #[serde(default)]
        metadata: Value,
        #[serde(default)]
        attachments: Vec<AdapterAttachment>,
    },
    Lifecycle {
        name: String,
        #[serde(default)]
        metadata: Value,
    },
    Error {
        message: String,
    },
    CommandAck {
        command_id: String,
    },
    CommandNack {
        command_id: String,
        message: String,
    },
    Disconnected {
        #[serde(default)]
        reason: Option<String>,
    },
}

pub async fn run_worker_loop<F, Fut, G, OutFut, S, StopFut>(
    adapter_id: &str,
    config: &AdapterConfig,
    secret_env: Vec<(String, String)>,
    outbound_notify: Arc<Notify>,
    on_event: F,
    take_outbound_messages: G,
    mut should_stop: S,
) -> Result<()>
where
    F: FnMut(WorkerEvent) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    G: FnMut() -> OutFut + Send + 'static,
    OutFut: std::future::Future<Output = Result<Vec<WorkerCommand>>> + Send + 'static,
    S: FnMut() -> StopFut,
    StopFut: std::future::Future<Output = Result<bool>>,
{
    tracing::info!(
        adapter_type = %config.adapter_type,
        adapter_id = %adapter_id,
        worker_command = ?config.worker_command,
        "starting adapter worker"
    );
    let mut command = worker_command(adapter_id, config, secret_env);
    // Workers can spawn their own children (pnpm -> tsx -> node). Put the
    // whole tree in its own process group so shutdown can signal everything;
    // kill_on_drop alone would only reach the direct child and orphan the
    // grandchild that actually holds the external connection.
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .context("failed to spawn adapter worker")?;
    let worker_pid = child.id();
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("adapter worker stdin was not piped"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("adapter worker stdout was not piped"))?;
    let mut lines = BufReader::new(stdout).lines();
    let mut stop_interval = tokio::time::interval(std::time::Duration::from_secs(1));
    let pending_events = Arc::new(AtomicUsize::new(0));
    let (event_tx, event_rx) = mpsc::channel(256);
    let mut event_task = tokio::spawn(process_worker_events(
        event_rx,
        on_event,
        Arc::clone(&pending_events),
        super::runtime::uses_target_conversation_scope(config),
    ));
    let mut command_task = tokio::spawn(process_worker_commands(
        stdin,
        outbound_notify,
        take_outbound_messages,
    ));

    // Once a stop request is observed it is latched: the loop stops reading
    // new worker events and exits as soon as already-accepted events finish.
    let mut stopping = false;
    let result = loop {
        tokio::select! {
            status = child.wait() => {
                break match status {
                    Ok(status) => Err(anyhow!("adapter worker exited with status {status}")),
                    Err(error) => Err(error.into()),
                };
            }
            line = lines.next_line(), if !stopping => {
                let line = match line {
                    Ok(Some(line)) => line,
                    Ok(None) => break Err(anyhow!("adapter worker closed stdout")),
                    Err(error) => break Err(error.into()),
                };
                let event = match serde_json::from_str::<WorkerEvent>(&line) {
                    Ok(event) => event,
                    Err(error) => {
                        break Err(anyhow::Error::from(error)
                            .context(format!("failed to parse adapter worker event: {line}")));
                    }
                };
                tracing::debug!(
                    adapter_type = %config.adapter_type,
                    event = ?event,
                    "adapter worker event"
                );
                pending_events.fetch_add(1, Ordering::SeqCst);
                if event_tx.send(event).await.is_err() {
                    break Err(anyhow!("adapter event handler stopped"));
                }
            }
            _ = stop_interval.tick() => {
                // Latch the stop request even while events are pending so a
                // steady inbound stream cannot defer shutdown forever; only
                // exit between events so an in-flight wakeup turn is not
                // cancelled halfway through.
                if !stopping && should_stop().await? {
                    stopping = true;
                }
                if stopping && pending_events.load(Ordering::SeqCst) == 0 {
                    break Ok(());
                }
            }
            result = &mut event_task => {
                break match result {
                    Ok(Ok(())) => Err(anyhow!("adapter event handler stopped")),
                    Ok(Err(error)) => Err(error),
                    Err(error) => Err(anyhow!("adapter event handler task failed: {error}")),
                };
            }
            result = &mut command_task => {
                break match result {
                    Ok(Ok(())) => Err(anyhow!("adapter command sender stopped")),
                    Ok(Err(error)) => Err(error),
                    Err(error) => Err(anyhow!("adapter command sender task failed: {error}")),
                };
            }
        }
    };
    command_task.abort();
    event_task.abort();
    terminate_worker_process_group(worker_pid).await;
    result
}

#[cfg(unix)]
async fn terminate_worker_process_group(pid: Option<u32>) {
    let Some(pid) = pid else {
        return;
    };
    let group = -(pid as i32);
    // SIGTERM lets workers shut down cleanly; SIGKILL sweeps up anything
    // (including descendants of an already-dead leader) that ignored it.
    unsafe { libc::kill(group, libc::SIGTERM) };
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    unsafe { libc::kill(group, libc::SIGKILL) };
}

#[cfg(not(unix))]
async fn terminate_worker_process_group(_pid: Option<u32>) {}

async fn process_worker_events<F, Fut>(
    mut event_rx: mpsc::Receiver<WorkerEvent>,
    mut on_event: F,
    pending_events: Arc<AtomicUsize>,
    target_scoped: bool,
) -> Result<()>
where
    F: FnMut(WorkerEvent) -> Fut,
    Fut: std::future::Future<Output = Result<()>> + Send + 'static,
{
    // Adapt upstream #382's independent-thread dispatch to the older worker
    // protocol. Keep FIFO admission per target, including conversation creation
    // and attachment downloads, while acknowledgements keep flowing.
    const MAX_ACTIVE_TARGETS: usize = 8;
    const MAX_QUEUED_MESSAGES: usize = 256;
    let mut queued: HashMap<String, VecDeque<WorkerEvent>> = HashMap::new();
    let mut queued_count = 0;
    let mut active = HashSet::new();
    let mut running: FuturesUnordered<BoxFuture<'static, (String, Result<()>)>> =
        FuturesUnordered::new();
    let mut receiving = true;

    loop {
        while running.len() < MAX_ACTIVE_TARGETS {
            let Some(key) = queued.keys().find(|key| !active.contains(*key)).cloned() else {
                break;
            };
            let queue = queued.get_mut(&key).expect("queued target");
            let event = queue.pop_front().expect("queued message");
            if queue.is_empty() {
                queued.remove(&key);
            }
            queued_count -= 1;
            active.insert(key.clone());
            let future = on_event(event);
            running.push(Box::pin(async move { (key, future.await) }));
        }
        if !receiving && running.is_empty() && queued.is_empty() {
            return Ok(());
        }
        tokio::select! {
            Some((key, result)) = running.next(), if !running.is_empty() => {
                active.remove(&key);
                pending_events.fetch_sub(1, Ordering::SeqCst);
                result?;
            }
            event = event_rx.recv(), if receiving && queued_count < MAX_QUEUED_MESSAGES => {
                match event {
                    Some(event @ WorkerEvent::Message { .. }) => {
                        let WorkerEvent::Message { target, .. } = &event else { unreachable!() };
                        let key = if target_scoped { target.clone() } else { String::new() };
                        queued.entry(key).or_default().push_back(event);
                        queued_count += 1;
                    }
                    Some(event) => {
                        let result = on_event(event).await;
                        pending_events.fetch_sub(1, Ordering::SeqCst);
                        result?;
                    }
                    None => receiving = false,
                }
            }
        }
    }
}

async fn process_worker_commands<G, OutFut>(
    mut stdin: ChildStdin,
    outbound_notify: Arc<Notify>,
    mut take_outbound_messages: G,
) -> Result<()>
where
    G: FnMut() -> OutFut,
    OutFut: std::future::Future<Output = Result<Vec<WorkerCommand>>>,
{
    loop {
        send_pending_commands(&mut stdin, &outbound_notify, &mut take_outbound_messages).await?;
    }
}

fn worker_command(
    adapter_id: &str,
    config: &AdapterConfig,
    secret_env: Vec<(String, String)>,
) -> Command {
    let args = &config.worker_command;
    let mut command = Command::new(&args[0]);
    command.args(&args[1..]);
    command.env("EXO_ADAPTER_ID", adapter_id);
    command.env("EXO_ADAPTER_TYPE", &config.adapter_type);
    command.env("EXO_ADAPTER_STATE_DIR", state_dir(adapter_id, config));
    command.env(
        "EXO_ADAPTER_CONFIG",
        serde_json::to_string(&config.initialization).expect("adapter initialization is JSON"),
    );
    for (name, value) in secret_env {
        command.env(name, value);
    }
    command
}

fn state_dir(adapter_id: &str, config: &AdapterConfig) -> String {
    config.state_dir.clone().unwrap_or_else(|| {
        Path::new(".exo")
            .join("adapters")
            .join(&config.adapter_type)
            .join(adapter_id)
            .to_string_lossy()
            .to_string()
    })
}

async fn send_pending_commands<G, OutFut>(
    stdin: &mut ChildStdin,
    outbound_notify: &Notify,
    take_outbound_messages: &mut G,
) -> Result<()>
where
    G: FnMut() -> OutFut,
    OutFut: std::future::Future<Output = Result<Vec<WorkerCommand>>>,
{
    tokio::select! {
        () = outbound_notify.notified() => {}
        () = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
    }
    for command in take_outbound_messages().await? {
        tracing::debug!(command = ?command, "sending adapter worker command");
        stdin
            .write_all(serde_json::to_string(&command)?.as_bytes())
            .await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use serde_json::json;
    use tempfile::TempDir;
    use tokio::sync::{Notify, oneshot};

    use super::*;

    #[test]
    fn message_event_parses_with_and_without_attachments() {
        let plain: WorkerEvent = serde_json::from_str(
            r#"{"type":"message","target":"channel-1","text":"hi","metadata":{}}"#,
        )
        .unwrap();
        let WorkerEvent::Message { attachments, .. } = plain else {
            panic!("expected message event");
        };
        assert!(attachments.is_empty());

        let rich: WorkerEvent = serde_json::from_str(
            r#"{"type":"message","target":"channel-1","text":"hi","attachments":[{"kind":"image","url":"https://cdn.example/a.png","mimeType":"image/png","fileName":"a.png"}]}"#,
        )
        .unwrap();
        let WorkerEvent::Message { attachments, .. } = rich else {
            panic!("expected message event");
        };
        assert_eq!(attachments.len(), 1);
        assert_eq!(
            attachments[0].url.as_deref(),
            Some("https://cdn.example/a.png")
        );
        assert_eq!(attachments[0].mime_type.as_deref(), Some("image/png"));
    }

    #[tokio::test]
    async fn target_conversations_overlap_preserving_fifo_control_events_and_drain() {
        check_dispatch(true).await;
        check_dispatch(false).await;
    }

    async fn check_dispatch(target_scoped: bool) {
        let (events_tx, events_rx) = mpsc::channel(8);
        for (target, text) in [("a", "a1"), ("a", "a2"), ("b", "b1")] {
            events_tx
                .send(WorkerEvent::Message {
                    target: target.into(),
                    sender: None,
                    text: text.into(),
                    message_id: None,
                    metadata: Value::Null,
                    attachments: Vec::new(),
                })
                .await
                .unwrap();
        }
        events_tx
            .send(WorkerEvent::CommandAck {
                command_id: "ack".into(),
            })
            .await
            .unwrap();
        drop(events_tx);
        let pending = Arc::new(AtomicUsize::new(4));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let (started_tx, mut started_rx) = mpsc::unbounded_channel();
        let handler = tokio::spawn(process_worker_events(
            events_rx,
            {
                let release = release.clone();
                move |event| {
                    let release = release.clone();
                    let started = started_tx.clone();
                    async move {
                        match event {
                            WorkerEvent::Message { text, .. } => {
                                started.send(text)?;
                                release.acquire().await?.forget();
                            }
                            WorkerEvent::CommandAck { command_id } => {
                                started.send(command_id)?;
                            }
                            _ => unreachable!(),
                        }
                        Ok(())
                    }
                }
            },
            pending.clone(),
            target_scoped,
        ));
        let mut seen = HashSet::new();
        let expected = if target_scoped { 3 } else { 2 };
        for _ in 0..expected {
            seen.insert(
                tokio::time::timeout(Duration::from_secs(5), started_rx.recv())
                    .await
                    .expect("dispatch stalled")
                    .unwrap(),
            );
        }
        assert!(seen.contains("a1"));
        assert!(seen.contains("ack"), "acks must not wait for a model turn");
        assert!(
            !seen.contains("a2"),
            "same-target messages ran concurrently"
        );
        assert_eq!(seen.contains("b1"), target_scoped);
        assert!(
            !handler.is_finished(),
            "closed input must drain accepted turns"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(100), started_rx.recv())
                .await
                .is_err()
        );
        release.add_permits(1);
        if target_scoped {
            // Either active target may take the permit; release the other too.
            release.add_permits(1);
        }
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), started_rx.recv())
                .await
                .unwrap()
                .as_deref(),
            Some("a2")
        );
        release.add_permits(1);
        if !target_scoped {
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(5), started_rx.recv())
                    .await
                    .unwrap()
                    .as_deref(),
                Some("b1")
            );
            release.add_permits(1);
        }
        tokio::time::timeout(Duration::from_secs(5), handler)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(pending.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn stops_worker_when_cancellation_trips() {
        let config = AdapterConfig {
            adapter_type: "test".to_string(),
            worker_command: vec![
                "sh".to_string(),
                "-c".to_string(),
                "while true; do sleep 1; done".to_string(),
            ],
            initialization: json!({}),
            state_dir: None,
            secret_env: Vec::new(),
        };
        let outbound_notify = Arc::new(Notify::new());

        run_worker_loop(
            "adapter",
            &config,
            Vec::new(),
            outbound_notify,
            |_event| async { Ok(()) },
            || async { Ok(Vec::new()) },
            || async { Ok(true) },
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn stops_busy_worker_once_accepted_events_drain() {
        let config = AdapterConfig {
            adapter_type: "test".to_string(),
            worker_command: vec![
                "sh".to_string(),
                "-c".to_string(),
                // Emit events faster than the handler drains them so the
                // pending-event count stays nonzero at stop-check ticks.
                "while true; do printf '%s\\n' '{\"type\":\"message\",\"target\":\"t\",\"text\":\"hi\"}'; sleep 0.05; done".to_string(),
            ],
            initialization: json!({}),
            state_dir: None,
            secret_env: Vec::new(),
        };
        let outbound_notify = Arc::new(Notify::new());
        let stop_checks = Arc::new(AtomicUsize::new(0));

        // The stop request arrives only after a backlog has built up. Without
        // latching, the loop would keep accepting new events and never observe
        // an idle tick, deferring shutdown forever.
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            run_worker_loop(
                "adapter",
                &config,
                Vec::new(),
                outbound_notify,
                |_event| async {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    Ok(())
                },
                || async { Ok(Vec::new()) },
                move || {
                    let stop_checks = Arc::clone(&stop_checks);
                    async move { Ok(stop_checks.fetch_add(1, Ordering::SeqCst) >= 2) }
                },
            ),
        )
        .await
        .expect("worker loop did not honor the stop request");
        result.unwrap();
    }

    #[tokio::test]
    async fn dispatches_outbound_commands_while_event_handler_is_busy() {
        let tempdir = TempDir::new().unwrap();
        let output_path = tempdir.path().join("command.json");
        let config = AdapterConfig {
            adapter_type: "test".to_string(),
            worker_command: vec![
                "sh".to_string(),
                "-c".to_string(),
                "printf '%s\n' '{\"type\":\"message\",\"target\":\"target\",\"text\":\"hello\"}'; IFS= read -r line; printf '%s\n' \"$line\" > \"$OUTPUT_PATH\"; sleep 60".to_string(),
            ],
            initialization: json!({}),
            state_dir: None,
            secret_env: Vec::new(),
        };
        let outbound = Arc::new(Mutex::new(Vec::<WorkerCommand>::new()));
        let outbound_for_worker = Arc::clone(&outbound);
        let outbound_notify = Arc::new(Notify::new());
        let outbound_notify_for_worker = Arc::clone(&outbound_notify);
        let (event_started_tx, event_started_rx) = oneshot::channel();
        let (release_event_tx, release_event_rx) = oneshot::channel();
        let mut event_started_tx = Some(event_started_tx);
        let mut release_event_rx = Some(release_event_rx);

        let worker = tokio::spawn(async move {
            run_worker_loop(
                "adapter",
                &config,
                vec![(
                    "OUTPUT_PATH".to_string(),
                    output_path.to_string_lossy().into_owned(),
                )],
                outbound_notify_for_worker,
                move |event| {
                    let event_started_tx = event_started_tx.take();
                    let release_event_rx = release_event_rx.take();
                    async move {
                        if matches!(event, WorkerEvent::Message { .. }) {
                            if let Some(event_started_tx) = event_started_tx
                                && event_started_tx.send(()).is_err()
                            {
                                panic!("test receiver dropped before event started");
                            }
                            if let Some(release_event_rx) = release_event_rx {
                                release_event_rx.await.unwrap();
                            }
                        }
                        Ok(())
                    }
                },
                move || {
                    let outbound = Arc::clone(&outbound_for_worker);
                    async move {
                        let mut outbound = outbound.lock().unwrap();
                        Ok(outbound.drain(..).collect())
                    }
                },
                || async { Ok(false) },
            )
            .await
        });

        event_started_rx.await.unwrap();
        outbound.lock().unwrap().push(WorkerCommand::SendMessage {
            id: "command".to_string(),
            target: Some("target".to_string()),
            text: "pong".to_string(),
            attachments: Vec::new(),
        });
        outbound_notify.notify_one();

        let written = wait_for_file(tempdir.path().join("command.json")).await;
        assert!(written.contains("\"type\":\"send_message\""));
        assert!(written.contains("\"text\":\"pong\""));

        release_event_tx.send(()).unwrap();
        worker.abort();
        match worker.await {
            Err(error) if error.is_cancelled() => {}
            other => panic!("worker task should have been cancelled, got {other:?}"),
        }
    }

    async fn wait_for_file(path: impl AsRef<Path>) -> String {
        let path = path.as_ref().to_path_buf();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                // The worker creates the file before the content lands (shell
                // `>` truncates first), so wait until the contents are a
                // complete JSON document, not merely until the file exists.
                if let Ok(contents) = tokio::fs::read_to_string(&path).await
                    && serde_json::from_str::<serde_json::Value>(&contents).is_ok()
                {
                    return contents;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("worker did not receive outbound command")
    }
}

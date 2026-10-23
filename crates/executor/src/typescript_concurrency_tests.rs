use super::*;
use crate::{
    AgentHarnessKind, CreateAgentRequest, CreateConversationRequest, Harness, HarnessAgent,
    HarnessConversation, SandboxProvider, SandboxScope, TypeScriptHarnessConfig,
    test_support::local_test_config,
};
use tempfile::TempDir;

#[derive(Debug)]
enum Probe {
    Entered(ConversationId),
    Output(ConversationId, String),
}

struct GateTools {
    events: mpsc::UnboundedSender<Probe>,
    release: Semaphore,
}

#[async_trait]
impl ToolRuntime for GateTools {
    async fn execute(
        &self,
        _agent: &dyn AgentHandle,
        conversation: &dyn ConversationHandle,
        _turn: Option<&dyn TurnHandle>,
        _agent_config: &AgentConfig,
        _config: &ConversationConfig,
        request: &ToolRequest,
    ) -> Result<ToolResult> {
        let id = conversation.record().id;
        match request.function_name.as_str() {
            "gate" => {
                self.events.send(Probe::Entered(id))?;
                self.release.acquire().await?.forget();
            }
            "output" => {
                #[derive(Deserialize)]
                struct Output {
                    stdout: String,
                    exit: i32,
                }
                let output: Output =
                    serde_json::from_value(serde_json::Value::Object(request.arguments.clone()))?;
                assert_eq!(output.exit, 0);
                self.events.send(Probe::Output(id, output.stdout))?;
            }
            name => bail!("unexpected test tool {name}"),
        }
        Ok(serde_json::Value::Null)
    }
}

struct Fixture {
    executor: TypeScriptExecutor<GateTools>,
    agent: Arc<dyn HarnessAgent>,
    events: mpsc::UnboundedReceiver<Probe>,
    tools: Arc<GateTools>,
    _root: TempDir,
}

impl Fixture {
    async fn new() -> Result<Self> {
        Self::with_harness(
            r#"
export default {
  async runTurn(context) {
    const process = await context.startSandboxProcess({
      command: ["/bin/sh", "-c", "sleep 0.1; printf '%s\\n' \"$PROBE_ID\""],
      env: { PROBE_ID: context.exoharness.current.conversation.record.id },
    });
    await context.executeTool({ functionName: "gate", arguments: {} });
    let stdout = "";
    const reader = process.stdout.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      stdout += value;
    }
    const exit = await process.wait();
    await process.close();
    await context.executeTool({ functionName: "output", arguments: { stdout, exit } });
  }
};
"#,
        )
        .await
    }

    async fn with_harness(source: &str) -> Result<Self> {
        let root = TempDir::new()?;
        let module = root.path().join("harness.mjs");
        std::fs::write(&module, source)?;
        let state: Arc<dyn ExoHarness> =
            Arc::new(BasicExoHarness::new(local_test_config(root.path().join("state"))).await?);
        let (events_tx, events) = mpsc::unbounded_channel();
        let tools = Arc::new(GateTools {
            events: events_tx,
            release: Semaphore::new(0),
        });
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let executor =
            TypeScriptExecutor::new(state.clone(), workspace, HashMap::new(), tools.clone());
        let harness = TypeScriptHarness {
            inner: SharedHarness::new(state, ExecutorHarnessRuntime::new(executor.clone(), None)),
        };
        let agent = harness
            .create_agent(CreateAgentRequest {
                slug: "concurrency-test".into(),
                name: None,
                harness: AgentHarnessKind::TypeScript,
                typescript: Some(TypeScriptHarnessConfig {
                    module_path: module.to_string_lossy().into_owned(),
                    tool_module_paths: Vec::new(),
                }),
                enable_agent_tool_creation: false,
                sandbox_image: None,
                sandbox_provider: SandboxProvider::LocalProcess,
                sandbox_scope: Some(SandboxScope::Conversation),
                enable_networking: false,
                model: "unused".into(),
                max_output_tokens: None,
                max_tool_round_trips: None,
                braintrust: None,
            })
            .await?;
        Ok(Self {
            executor,
            agent,
            events,
            tools,
            _root: root,
        })
    }

    async fn conversation(&self) -> Result<Arc<dyn HarnessConversation>> {
        self.agent
            .create_conversation(CreateConversationRequest::default())
            .await
    }

    async fn next(&mut self) -> Probe {
        tokio::time::timeout(Duration::from_secs(10), self.events.recv())
            .await
            .expect("turn did not progress")
            .expect("probe closed")
    }
}

fn send(conversation: Arc<dyn HarnessConversation>) -> JoinHandle<Result<crate::SendResult>> {
    tokio::spawn(async move {
        conversation
            .send(SendRequest {
                input: vec![lingua::Message::User {
                    content: lingua::universal::UserContent::String("run sandbox probe".into()),
                }],
                session_id: None,
            })
            .await
    })
}

async fn finish(turn: JoinHandle<Result<crate::SendResult>>) {
    tokio::time::timeout(Duration::from_secs(10), turn)
        .await
        .expect("turn did not finish")
        .expect("turn panicked")
        .expect("turn failed");
}

#[tokio::test]
async fn separate_conversations_overlap_and_keep_sandbox_output_isolated() -> Result<()> {
    let mut f = Fixture::new().await?;
    let a = f.conversation().await?;
    let b = f.conversation().await?;
    let a_id = a.record().id;
    let b_id = b.record().id;
    let first = send(a);
    assert!(matches!(f.next().await, Probe::Entered(id) if id == a_id));
    let second = send(b);
    // Both host tool calls must be active before either is allowed to finish.
    // A shared module runner deadlocks here until the test's deadline.
    assert!(matches!(f.next().await, Probe::Entered(id) if id == b_id));
    f.tools.release.add_permits(2);
    let mut outputs = HashMap::new();
    for _ in 0..2 {
        let Probe::Output(id, output) = f.next().await else {
            panic!("expected output")
        };
        outputs.insert(id, output);
    }
    assert_eq!(outputs.get(&a_id), Some(&format!("{a_id}\n")));
    assert_eq!(outputs.get(&b_id), Some(&format!("{b_id}\n")));
    finish(first).await;
    finish(second).await;
    Ok(())
}

#[tokio::test]
async fn same_conversation_waits_until_previous_turn_finishes() -> Result<()> {
    let mut f = Fixture::new().await?;
    let conversation = f.conversation().await?;
    let first = send(conversation.clone());
    assert!(matches!(f.next().await, Probe::Entered(_)));
    let second = send(conversation);
    assert!(
        tokio::time::timeout(Duration::from_millis(150), f.events.recv())
            .await
            .is_err()
    );
    f.tools.release.add_permits(1);
    assert!(matches!(f.next().await, Probe::Output(_, _)));
    finish(first).await;
    assert!(matches!(f.next().await, Probe::Entered(_)));
    f.tools.release.add_permits(1);
    assert!(matches!(f.next().await, Probe::Output(_, _)));
    finish(second).await;
    Ok(())
}

#[tokio::test]
async fn idle_runner_is_reaped_but_active_turn_is_protected_and_history_survives() -> Result<()> {
    let mut f = Fixture::new().await?;
    let conversation = f.conversation().await?;
    let first = send(conversation.clone());
    assert!(matches!(f.next().await, Probe::Entered(_)));
    let process = {
        let entries = f.executor.runners.entries.lock().unwrap();
        Arc::downgrade(&entries.values().next().unwrap().process)
    };
    f.executor
        .runners
        .retire_idle(Instant::now() + RUNNER_IDLE_TTL);
    assert!(process.upgrade().is_some(), "active runner was evicted");
    f.tools.release.add_permits(1);
    assert!(matches!(f.next().await, Probe::Output(_, _)));
    finish(first).await;
    let pid = process.upgrade().unwrap().lock().await.child.id().unwrap();
    let previous_events = conversation
        .exoharness_handle()
        .get_events(None)
        .await?
        .events
        .len();
    f.executor
        .runners
        .retire_idle(Instant::now() + RUNNER_IDLE_TTL);
    assert!(process.upgrade().is_none(), "idle runner stayed alive");
    assert!(f.executor.runners.entries.lock().unwrap().is_empty());
    #[cfg(unix)]
    tokio::time::timeout(Duration::from_secs(5), async {
        while unsafe { libc::kill(pid as i32, 0) } == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("evicted Node process is still running");
    let second = send(conversation.clone());
    assert!(matches!(f.next().await, Probe::Entered(_)));
    f.tools.release.add_permits(1);
    assert!(matches!(f.next().await, Probe::Output(_, _)));
    finish(second).await;
    assert!(
        conversation
            .exoharness_handle()
            .get_events(None)
            .await?
            .events
            .len()
            > previous_events
    );
    Ok(())
}

#[tokio::test]
async fn cancelling_turn_discards_its_runner() -> Result<()> {
    let mut f = Fixture::new().await?;
    let conversation = f.conversation().await?;
    let first = send(conversation.clone());
    assert!(matches!(f.next().await, Probe::Entered(_)));
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    assert!(f.executor.runners.entries.lock().unwrap().is_empty());
    let second = send(conversation);
    assert!(matches!(f.next().await, Probe::Entered(_)));
    f.tools.release.add_permits(1);
    assert!(matches!(f.next().await, Probe::Output(_, _)));
    finish(second).await;
    Ok(())
}

#[tokio::test]
async fn eviction_reattaches_to_the_same_warm_sandbox_process() -> Result<()> {
    let mut f = Fixture::with_harness(r#"
export default {
  async runTurn(context) {
    const process = await context.startSandboxProcess({
      command: ["/bin/sh", "-c", "while IFS= read -r line; do printf '%s:%s\\n' \"$$\" \"$line\"; done"],
      env: {}, reuseKey: "echo",
    });
    await context.executeTool({ functionName: "gate", arguments: {} });
    await process.writeStdin(context.exoharness.current.conversation.record.id + "\n");
    const reader = process.stdout.getReader();
    const { value: stdout } = await reader.read();
    reader.releaseLock();
    if (process.reused) await process.close();
    await context.executeTool({ functionName: "output", arguments: { stdout, exit: 0 } });
  }
};
"#).await?;
    let conversation = f.conversation().await?;
    let first = send(conversation.clone());
    assert!(matches!(f.next().await, Probe::Entered(_)));
    f.tools.release.add_permits(1);
    let Probe::Output(_, first_output) = f.next().await else {
        panic!("expected output")
    };
    finish(first).await;
    f.executor
        .runners
        .retire_idle(Instant::now() + RUNNER_IDLE_TTL);
    assert!(f.executor.runners.entries.lock().unwrap().is_empty());
    let second = send(conversation);
    assert!(matches!(f.next().await, Probe::Entered(_)));
    f.tools.release.add_permits(1);
    let Probe::Output(_, second_output) = f.next().await else {
        panic!("expected output")
    };
    finish(second).await;
    assert_eq!(
        first_output, second_output,
        "sandbox process PID changed after runner eviction"
    );
    Ok(())
}

#[tokio::test]
async fn runner_capacity_bounds_processes_and_admits_waiting_conversations() -> Result<()> {
    let mut f = Fixture::new().await?;
    let mut turns = Vec::new();
    for _ in 0..MAX_TYPESCRIPT_RUNNERS {
        turns.push(send(f.conversation().await?));
        assert!(matches!(f.next().await, Probe::Entered(_)));
    }
    let waiting = f.conversation().await?;
    let waiting_id = waiting.record().id;
    turns.push(send(waiting));
    assert!(
        tokio::time::timeout(Duration::from_millis(150), f.events.recv())
            .await
            .is_err()
    );
    assert_eq!(
        f.executor.runners.entries.lock().unwrap().len(),
        MAX_TYPESCRIPT_RUNNERS
    );
    f.tools.release.add_permits(1);
    assert!(matches!(f.next().await, Probe::Output(_, _)));
    assert!(matches!(f.next().await, Probe::Entered(id) if id == waiting_id));
    assert_eq!(
        f.executor.runners.entries.lock().unwrap().len(),
        MAX_TYPESCRIPT_RUNNERS
    );
    f.tools.release.add_permits(MAX_TYPESCRIPT_RUNNERS);
    for _ in 0..MAX_TYPESCRIPT_RUNNERS {
        assert!(matches!(f.next().await, Probe::Output(_, _)));
    }
    for turn in turns {
        finish(turn).await;
    }
    Ok(())
}

# openai-assistants

The OpenAI Assistants API creates and runs assistants server-side. The SDK
owns the create-agent and invoke-agent client operations; application code
executes function-tool calls. The model call and `file_search` run server-side.

| Operation | Should be instrumented here | Status |
| --- | --- | --- |
| create_agent | Yes — creates the assistant | ✅ Implemented |
| invoke_agent (client) | Yes — runs the assistant (threads / runs) | ✅ Implemented |
| execute_tool | No — application code runs tools | Application-only span; not SDK evidence |
| inference (`chat`) | No — runs server-side inside the assistant run | ✅ Correctly not emitted |
| retrieval | No — `file_search` runs server-side | ✅ Correctly not emitted |

The invoke-agent span's error handler records a correlated
`gen_ai.client.invoke_agent.exception` WARN log for SDK calls, tool execution,
and response processing. Regression tests exercise failures; the conformance
scenario exercises the success path.

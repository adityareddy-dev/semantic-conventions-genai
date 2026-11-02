"""Check tool failures and native handoff telemetry through the real OpenAI Agents SDK."""

import os
import unittest
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agents import set_tracing_disabled
from agents.items import ModelResponse
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

with patch.dict(os.environ, {"MOCK_LLM_URL": "http://127.0.0.1:9999"}):
    import scenario


class FailingOutput:
    def __init__(self, error):
        self.error = error

    def __str__(self):
        raise self.error


class InvocationExceptionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        set_tracing_disabled(True)
        self.addCleanup(set_tracing_disabled, False)
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        self.addCleanup(provider.shutdown)
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.events = []

        def emit(**kwargs):
            self.events.append((trace.get_current_span().get_span_context(), kwargs))

        tracer_patch = patch.object(scenario, "_reference_tracer", provider.get_tracer("test"))
        logger_patch = patch.object(scenario, "reference_event_logger", return_value=Mock(emit=emit))
        tracer_patch.start()
        logger_patch.start()
        self.addCleanup(tracer_patch.stop)
        self.addCleanup(logger_patch.stop)

    def assert_failure(self, error):
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.status.status_code, trace.StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], f"{type(error).__module__}.{type(error).__qualname__}")
        self.assertEqual(span.events, ())
        self.assertEqual(len(self.events), 1)
        context, event = self.events[0]
        self.assertEqual(context.span_id, span.context.span_id)
        self.assertEqual(event["event_name"], "gen_ai.invoke_agent.exception")
        self.assertEqual(event["severity_number"], SeverityNumber.WARN)
        self.assertIs(event["exception"], error)
        self.assertFalse(trace.get_current_span().is_recording())

    async def test_preparation_sdk_and_output_failures_record_one_exception(self):
        client = scenario.openai.AsyncOpenAI(base_url=scenario.MOCK_BASE_URL, api_key="mock-key")
        self.addAsyncCleanup(client.close)
        original_create = client.chat.completions.create
        original_set_attribute = Span.set_attribute

        for phase in ("preparation", "sdk", "output"):
            with self.subTest(phase=phase):
                self.exporter.clear()
                self.events.clear()
                error = RuntimeError(f"{phase} failed")

                def set_attribute(span, key, value, phase=phase, error=error):
                    if phase == "preparation" and key == "gen_ai.system_instructions":
                        raise error
                    return original_set_attribute(span, key, value)

                result = SimpleNamespace(
                    context_wrapper=SimpleNamespace(usage=SimpleNamespace(total_tokens=0)),
                    final_output=FailingOutput(error),
                )
                run = AsyncMock(side_effect=error) if phase == "sdk" else AsyncMock(return_value=result)
                with (
                    patch.object(scenario.openai, "AsyncOpenAI", return_value=client),
                    patch.object(scenario.Runner, "run", run),
                    patch.object(Span, "set_attribute", set_attribute),
                    self.assertRaises(RuntimeError) as raised,
                ):
                    await scenario.run_agent()
                self.assertIs(raised.exception, error)
                self.assertEqual(client.chat.completions.create, original_create)
                if phase == "preparation":
                    run.assert_not_awaited()
                else:
                    run.assert_awaited_once()
                self.assert_failure(error)

    async def test_delegation_output_failure_restores_response_context(self):
        error = RuntimeError("delegation output failed")
        responses = ContextVar("test_responses", default=None)
        result = SimpleNamespace(raw_responses=[], final_output=FailingOutput(error))
        with (
            patch.object(scenario, "ContextVar", return_value=responses),
            patch.object(scenario.AgentRunner, "run", AsyncMock(return_value=result)),
            self.assertRaises(RuntimeError) as raised,
        ):
            await scenario.run_agent_as_tool_delegation()
        self.assertIs(raised.exception, error)
        self.assertIsNone(responses.get())
        self.assert_failure(error)

    async def test_expected_turn_limit_is_recorded_before_scenario_handles_it(self):
        error = scenario.MaxTurnsExceeded("turn limit")
        with patch.object(scenario.Runner, "run", AsyncMock(side_effect=error)):
            await scenario.run_agent_error_reference()
        self.assert_failure(error)

    async def test_unexpected_error_in_turn_limit_demo_propagates(self):
        error = RuntimeError("unexpected failure")
        with (
            patch.object(scenario.Runner, "run", AsyncMock(side_effect=error)),
            self.assertRaises(RuntimeError) as raised,
        ):
            await scenario.run_agent_error_reference()
        self.assertIs(raised.exception, error)
        self.assert_failure(error)


class NativeHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_expected_tool_validation_error_records_one_correlated_log(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        self.addCleanup(provider.shutdown)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        logs = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        self.addCleanup(logger_provider.shutdown)
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))

        with (
            patch.object(scenario, "_reference_tracer", provider.get_tracer("test")),
            patch.object(scenario, "reference_event_logger", return_value=logger_provider.get_logger("test")),
        ):
            await scenario.run_tool_error_reference()

        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "agents.exceptions.ModelBehaviorError")
        self.assertEqual(span.events, ())
        records = logs.get_finished_logs()
        self.assertEqual(len(records), 1)
        record = records[0].log_record
        self.assertEqual(record.event_name, "gen_ai.execute_tool.exception")
        self.assertEqual(record.severity_number, SeverityNumber.WARN)
        self.assertEqual(record.span_id, span.context.span_id)
        self.assertEqual(record.trace_id, span.context.trace_id)

    async def test_handoff_records_sdk_caller_tool_call_and_target(self):
        set_tracing_disabled(True)
        self.addCleanup(set_tracing_disabled, False)
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        self.addCleanup(provider.shutdown)
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        async def respond(model, *args, **kwargs):
            handoffs = kwargs["handoffs"]
            if handoffs:
                output = [
                    ResponseFunctionToolCall(
                        id="handoff-item",
                        call_id="handoff-call",
                        name=handoffs[0].tool_name,
                        arguments="{}",
                        type="function_call",
                    )
                ]
            else:
                output = [
                    ResponseOutputMessage(
                        id="answer",
                        role="assistant",
                        status="completed",
                        type="message",
                        content=[ResponseOutputText(text="Sunny in Seattle.", annotations=[], type="output_text")],
                    )
                ]
            return ModelResponse(
                output=output, usage=Usage(requests=1, input_tokens=12, output_tokens=4), response_id=None
            )

        with (
            patch.object(scenario, "_reference_tracer", provider.get_tracer("test")),
            patch.object(OpenAIChatCompletionsModel, "get_response", respond),
        ):
            await scenario.run_workflow()

        spans = exporter.get_finished_spans()
        transfers = [span for span in spans if span.attributes.get("gen_ai.transfer.mode") == "pass_control"]
        self.assertEqual(len(transfers), 1)
        transfer = transfers[0]
        self.assertEqual(transfer.attributes["gen_ai.agent.name"], "agent-a")
        self.assertEqual(transfer.attributes["gen_ai.transfer.target.name"], "agent-b")
        self.assertEqual(transfer.attributes["gen_ai.tool.call.id"], "handoff-call")
        self.assertEqual(transfer.attributes["gen_ai.tool.call.arguments"], "{}")
        self.assertEqual(transfer.name, f"execute_tool {transfer.attributes['gen_ai.tool.name']}")
        agents = {
            span.attributes["gen_ai.agent.name"]: span
            for span in spans
            if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
        }
        self.assertEqual(set(agents), {"agent-a", "agent-b"})
        self.assertEqual(transfer.parent.span_id, agents["agent-a"].context.span_id)
        self.assertLessEqual(transfer.end_time, agents["agent-b"].start_time)
        self.assertEqual(transfer.context.trace_id, agents["agent-b"].context.trace_id)


if __name__ == "__main__":
    unittest.main()

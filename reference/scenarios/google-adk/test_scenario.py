"""Check native ADK transfers through the real agent and tool runners."""

import os
import unittest
from unittest.mock import Mock, patch

from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

with patch.dict(os.environ, {"MOCK_LLM_URL": "http://127.0.0.1:9999"}):
    import scenario


class NativeTransferTests(unittest.TestCase):
    def test_handoff_processing_and_usage_finalization_failures_record_one_exception(self):
        original_set_attribute = Span.set_attribute
        for attribute in ("gen_ai.output.messages", "gen_ai.usage.input_tokens"):
            with self.subTest(attribute=attribute):
                exporter = InMemorySpanExporter()
                provider = TracerProvider()
                self.addCleanup(provider.shutdown)
                provider.add_span_processor(SimpleSpanProcessor(exporter))
                error = RuntimeError(f"{attribute} failed")
                events = []

                async def respond(model, llm_request, stream=False):
                    yield LlmResponse(
                        content=types.Content(role="model", parts=[types.Part(text="Sunny.")]),
                        finish_reason=types.FinishReason.STOP,
                        usage_metadata=types.GenerateContentResponseUsageMetadata(
                            prompt_token_count=12, candidates_token_count=4
                        ),
                    )

                def set_attribute(span, key, value, attribute=attribute, error=error):
                    if key == attribute:
                        raise error
                    return original_set_attribute(span, key, value)

                def emit(events=events, **kwargs):
                    events.append((trace.get_current_span().get_span_context(), kwargs))

                with (
                    patch.object(scenario, "_reference_tracer", provider.get_tracer("test")),
                    patch.object(scenario, "reference_event_logger", return_value=Mock(emit=emit)),
                    patch.object(Gemini, "generate_content_async", respond),
                    patch.object(Span, "set_attribute", set_attribute),
                    self.assertRaises(RuntimeError) as raised,
                ):
                    scenario.run_multi_agent_handoff_reference()
                self.assertIs(raised.exception, error)
                agents = [
                    span
                    for span in exporter.get_finished_spans()
                    if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
                ]
                self.assertEqual(len(agents), 1)
                span = agents[0]
                self.assertEqual(span.status.status_code, trace.StatusCode.ERROR)
                self.assertEqual(span.attributes["error.type"], "builtins.RuntimeError")
                self.assertEqual(span.events, ())
                self.assertEqual(len(events), 1)
                context, event = events[0]
                self.assertEqual(context.span_id, span.context.span_id)
                self.assertEqual(event["event_name"], "gen_ai.invoke_agent.exception")
                self.assertEqual(event["severity_number"], SeverityNumber.WARN)
                self.assertIs(event["exception"], error)
                self.assertFalse(trace.get_current_span().is_recording())

    def test_transfer_records_caller_target_and_separate_agent_execution(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        async def respond(model, llm_request, stream=False):
            if "transfer_to_agent" in llm_request.tools_dict:
                part = types.Part(
                    function_call=types.FunctionCall(
                        id="transfer-call",
                        name="transfer_to_agent",
                        args={"agent_name": "weather_specialist"},
                    )
                )
            else:
                part = types.Part(text="Sunny in Seattle.")
            yield LlmResponse(
                content=types.Content(role="model", parts=[part]),
                finish_reason=types.FinishReason.STOP,
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=12, candidates_token_count=4
                ),
            )

        with (
            patch.object(scenario, "_reference_tracer", provider.get_tracer("test")),
            patch.object(Gemini, "generate_content_async", respond),
        ):
            self.assertTrue(
                callable(getattr(scenario, "run_multi_agent_handoff_reference", None)),
                "The scenario must exercise ADK's native transfer_to_agent tool",
            )
            scenario.run_multi_agent_handoff_reference()

        spans = exporter.get_finished_spans()
        transfers = [span for span in spans if span.attributes.get("gen_ai.transfer.mode") == "pass_control"]
        self.assertEqual(len(transfers), 1)
        transfer = transfers[0]
        self.assertEqual(transfer.name, "execute_tool transfer_to_agent")
        self.assertEqual(transfer.attributes["gen_ai.agent.name"], "root_agent")
        self.assertEqual(transfer.attributes["gen_ai.transfer.target.name"], "weather_specialist")
        self.assertEqual(transfer.attributes["gen_ai.tool.call.id"], "transfer-call")
        agents = {
            span.attributes["gen_ai.agent.name"]: span
            for span in spans
            if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
        }
        self.assertEqual(set(agents), {"root_agent", "weather_specialist"})
        self.assertEqual(transfer.parent.span_id, agents["root_agent"].context.span_id)
        self.assertLessEqual(transfer.end_time, agents["weather_specialist"].start_time)
        self.assertEqual(transfer.context.trace_id, agents["weather_specialist"].context.trace_id)
        for agent in agents.values():
            self.assertEqual(agent.attributes["gen_ai.usage.input_tokens"], 12)
            self.assertEqual(agent.attributes["gen_ai.usage.output_tokens"], 4)
        provider.shutdown()


if __name__ == "__main__":
    unittest.main()

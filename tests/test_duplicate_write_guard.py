"""Duplicate-write guard: identical write/edit/patch calls execute only once per run.

Engines (GLM double-submit habit) sometimes issue the IDENTICAL edit call
twice; with auto-approved permissions the repeat used to re-apply — anchored
insertions duplicated their block each time. The guard rejects the repeat
with an error result instead of executing it.
"""

from typing import Any

import pytest

from clawcode.llm.agent import Agent, AgentEvent, AgentEventType
from clawcode.llm.base import ToolCall
from clawcode.llm.tools import BaseTool, ToolContext, ToolInfo, ToolResponse

from test_agent import MockProvider, cleanup_test_environment, setup_test_environment


class _CountingEditTool(BaseTool):
    """Fake 'edit' tool that counts real executions."""

    def __init__(self) -> None:
        self.executions = 0

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="edit",
            description="Edit a file",
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, call: Any, context: ToolContext) -> ToolResponse:
        self.executions += 1
        return ToolResponse(content="edit_ok", is_error=False)


@pytest.mark.asyncio
async def test_identical_edit_call_rejected_within_run() -> None:
    """Second identical edit call in one run is rejected, not re-executed."""
    session_service, message_service, db = await setup_test_environment()
    try:
        same_input = '{"file_path": "/tmp/x.tsx", "old_string": "a", "new_string": "ab"}'
        provider = MockProvider(
            responses=["", "", "done"],
            tool_calls=[
                [ToolCall(id="c1", name="edit", input=same_input)],
                [ToolCall(id="c2", name="edit", input=same_input)],
                [],
            ],
        )
        tool = _CountingEditTool()
        agent = Agent(
            provider=provider,
            tools=[tool],
            message_service=message_service,
            session_service=session_service,
            max_iterations=10,
        )
        session = await session_service.create("Dup guard test")
        events: list[AgentEvent] = []
        async for ev in agent.run(session.id, "edit it"):
            events.append(ev)

        assert tool.executions == 1, f"edit ran {tool.executions} times, expected 1"
        results = [e for e in events if e.type == AgentEventType.TOOL_RESULT]
        dup_errors = [e for e in results if "duplicate" in (e.tool_result or "").lower()]
        assert dup_errors, (
            f"expected a duplicate-call error result, got: {[e.tool_result for e in results]}"
        )
    finally:
        await cleanup_test_environment(db)


@pytest.mark.asyncio
async def test_different_edit_calls_both_execute() -> None:
    """Non-identical edit calls are not blocked."""
    session_service, message_service, db = await setup_test_environment()
    try:
        provider = MockProvider(
            responses=["", "", "done"],
            tool_calls=[
                [ToolCall(id="c1", name="edit", input='{"file_path": "/tmp/x.tsx", "old_string": "a", "new_string": "ab"}')],
                [ToolCall(id="c2", name="edit", input='{"file_path": "/tmp/x.tsx", "old_string": "b", "new_string": "bc"}')],
                [],
            ],
        )
        tool = _CountingEditTool()
        agent = Agent(
            provider=provider,
            tools=[tool],
            message_service=message_service,
            session_service=session_service,
            max_iterations=10,
        )
        session = await session_service.create("Dup guard test 2")
        events: list[AgentEvent] = []
        async for ev in agent.run(session.id, "edit it twice differently"):
            events.append(ev)

        assert tool.executions == 2, f"edit ran {tool.executions} times, expected 2"
    finally:
        await cleanup_test_environment(db)


@pytest.mark.asyncio
async def test_guard_resets_between_runs() -> None:
    """The same edit call in a NEW run (new turn) executes normally."""
    session_service, message_service, db = await setup_test_environment()
    try:
        same_input = '{"file_path": "/tmp/x.tsx", "old_string": "a", "new_string": "ab"}'

        def make_provider() -> MockProvider:
            return MockProvider(
                responses=["", "done"],
                tool_calls=[[ToolCall(id="c1", name="edit", input=same_input)], []],
            )

        tool = _CountingEditTool()
        agent = Agent(
            provider=make_provider(),
            tools=[tool],
            message_service=message_service,
            session_service=session_service,
            max_iterations=10,
        )
        session = await session_service.create("Dup guard test 3")
        async for _ in agent.run(session.id, "turn one"):
            pass
        # Fresh provider state for the second turn
        agent._provider = make_provider()
        async for _ in agent.run(session.id, "turn two"):
            pass

        assert tool.executions == 2, f"edit ran {tool.executions} times across 2 runs, expected 2"
    finally:
        await cleanup_test_environment(db)

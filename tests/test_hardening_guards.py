"""Hardening guards: edit uniqueness, stale-read protection, loop breaker.

Companions to test_duplicate_write_guard.py — same MockProvider + fake-tool
pattern. Covers the 2026-07 hardening plan workstreams 1, 2, and 4.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from clawcode.llm.agent import Agent, AgentEvent, AgentEventType
from clawcode.llm.base import ToolCall
from clawcode.llm.tools import BaseTool, ToolContext, ToolInfo, ToolResponse
from clawcode.llm.tools.advanced import EditTool

from test_agent import MockProvider, cleanup_test_environment, setup_test_environment


# ── WS1: edit-tool uniqueness ────────────────────────────────────────


def _edit_call(path: str, replacements: list[dict]) -> ToolCall:
    import json

    return ToolCall(
        id="e1",
        name="edit",
        input=json.dumps({"file_path": path, "replacements": replacements}),
    )


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(
        session_id="s1",
        message_id="",
        working_directory=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_edit_ambiguous_match_errors(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x = 1\ny = 2\nx = 1\n")
    tool = EditTool()
    resp = await tool.run(
        _edit_call(str(f), [{"old_text": "x = 1", "new_text": "x = 9"}]),
        _ctx(tmp_path),
    )
    assert resp.is_error, resp.content
    assert "2 locations" in resp.content
    assert f.read_text() == "x = 1\ny = 2\nx = 1\n", "file must be untouched"


@pytest.mark.asyncio
async def test_edit_unique_match_applies(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x = 1\ny = 2\n")
    tool = EditTool()
    resp = await tool.run(
        _edit_call(str(f), [{"old_text": "y = 2", "new_text": "y = 9"}]),
        _ctx(tmp_path),
    )
    assert not resp.is_error, resp.content
    assert f.read_text() == "x = 1\ny = 9\n"


@pytest.mark.asyncio
async def test_edit_replace_all_still_works(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x = 1\ny = 2\nx = 1\n")
    tool = EditTool()
    resp = await tool.run(
        _edit_call(str(f), [{"old_text": "x = 1", "new_text": "x = 9", "replace_all": True}]),
        _ctx(tmp_path),
    )
    assert not resp.is_error, resp.content
    assert f.read_text() == "x = 9\ny = 2\nx = 9\n"


@pytest.mark.asyncio
async def test_edit_not_found_errors(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x = 1\n")
    tool = EditTool()
    resp = await tool.run(
        _edit_call(str(f), [{"old_text": "NOT PRESENT", "new_text": "z"}]),
        _ctx(tmp_path),
    )
    assert resp.is_error, resp.content
    assert "not found" in resp.content


# ── WS2: stale-read guard (agent-level) ──────────────────────────────


class _FakeViewTool(BaseTool):
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="view",
            description="View a file",
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, call: Any, context: ToolContext) -> ToolResponse:
        return ToolResponse(content="contents", is_error=False)


class _CountingEditTool(BaseTool):
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


def _tool_results(events: list[AgentEvent]) -> list[AgentEvent]:
    return [e for e in events if e.type == AgentEventType.TOOL_RESULT]


async def _run_agent(agent: Agent, session_id: str, prompt: str) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for ev in agent.run(session_id, prompt):
        events.append(ev)
    return events


@pytest.mark.asyncio
async def test_edit_without_read_rejected(tmp_path: Path) -> None:
    target = tmp_path / "t.txt"
    target.write_text("hello\n")
    import json as _json

    edit_input = _json.dumps({"file_path": str(target), "replacements": [{"old_text": "a", "new_text": "b"}]})
    session_service, message_service, db = await setup_test_environment()
    try:
        provider = MockProvider(
            responses=["", "done"],
            tool_calls=[[ToolCall(id="c1", name="edit", input=edit_input)], []],
        )
        tool = _CountingEditTool()
        agent = Agent(
            provider=provider,
            tools=[tool],
            message_service=message_service,
            session_service=session_service,
            max_iterations=10,
            working_directory=str(tmp_path),
        )
        session = await session_service.create("stale test")
        events = await _run_agent(agent, session.id, "edit")
        assert tool.executions == 0, "edit must not execute without a prior read"
        assert any("has not been read" in (e.tool_result or "") for e in _tool_results(events))
    finally:
        await cleanup_test_environment(db)


@pytest.mark.asyncio
async def test_edit_after_read_passes_and_stale_rejected(tmp_path: Path) -> None:
    target = tmp_path / "t.txt"
    target.write_text("hello\n")
    import json as _json

    view_input = _json.dumps({"file_path": str(target)})
    edit_input = _json.dumps({"file_path": str(target), "replacements": [{"old_text": "a", "new_text": "b"}]})
    session_service, message_service, db = await setup_test_environment()
    try:
        provider = MockProvider(
            responses=["", "", "done"],
            tool_calls=[
                [ToolCall(id="c1", name="view", input=view_input)],
                [ToolCall(id="c2", name="edit", input=edit_input)],
                [],
            ],
        )
        tool = _CountingEditTool()
        agent = Agent(
            provider=provider,
            tools=[_FakeViewTool(), tool],
            message_service=message_service,
            session_service=session_service,
            max_iterations=10,
            working_directory=str(tmp_path),
        )
        session = await session_service.create("stale test 2")
        await _run_agent(agent, session.id, "view then edit")
        assert tool.executions == 1, "edit after read must execute"

        # Now modify the file externally (bump mtime) and try an edit in a
        # NEW turn without re-reading: must be rejected as stale.
        import os

        st = target.stat()
        os.utime(target, (st.st_atime, st.st_mtime + 5))
        provider2 = MockProvider(
            responses=["", "done"],
            tool_calls=[[ToolCall(id="c3", name="edit", input=_json.dumps(
                {"file_path": str(target), "replacements": [{"old_text": "c", "new_text": "d"}]}
            ))], []],
        )
        agent._provider = provider2
        events2 = await _run_agent(agent, session.id, "edit again")
        assert tool.executions == 1, "stale edit must not execute"
        assert any("changed on disk" in (e.tool_result or "") for e in _tool_results(events2))
    finally:
        await cleanup_test_environment(db)


# ── WS4: consecutive-failure loop breaker ────────────────────────────


class _AlwaysFailingTool(BaseTool):
    def __init__(self) -> None:
        self.executions = 0

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="ls",
            description="List directory",
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, call: Any, context: ToolContext) -> ToolResponse:
        self.executions += 1
        return ToolResponse(content="boom", is_error=True)


@pytest.mark.asyncio
async def test_loop_breaker_stops_after_consecutive_failed_rounds() -> None:
    session_service, message_service, db = await setup_test_environment()
    try:
        # 10 rounds of failing tool calls available; breaker should stop at 3.
        provider = MockProvider(
            responses=[""] * 10 + ["done"],
            tool_calls=[[ToolCall(id=f"c{i}", name="ls", input="{}")] for i in range(10)] + [[]],
        )
        tool = _AlwaysFailingTool()
        agent = Agent(
            provider=provider,
            tools=[tool],
            message_service=message_service,
            session_service=session_service,
            max_iterations=50,
        )
        session = await session_service.create("loop breaker test")
        events: list[AgentEvent] = []
        async for ev in agent.run(session.id, "go"):
            events.append(ev)

        errs = [e for e in events if e.type == AgentEventType.ERROR]
        assert any("consecutive tool rounds" in (e.error or "") for e in errs), (
            f"expected loop-breaker error, got: {[(e.error or '') for e in errs]}"
        )
        assert tool.executions == 3, f"tool ran {tool.executions} times, expected 3 (breaker threshold)"
    finally:
        await cleanup_test_environment(db)

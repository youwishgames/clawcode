"""Hardening guards: edit uniqueness, stale-read protection, loop breaker.

Companions to test_duplicate_write_guard.py — same MockProvider + fake-tool
pattern. Covers the 2026-07 hardening plan workstreams 1, 2, and 4.

The stale-read guard lives in the TOOL layer (clawcode/llm/tools/file_guard.py)
so every caller inherits it — the agent loop AND direct callers like
execute_code. These tests therefore exercise the real tools, not fakes.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from clawcode.llm.agent import Agent, AgentEvent, AgentEventType
from clawcode.llm.base import ToolCall
from clawcode.llm.tools import BaseTool, ToolContext, ToolInfo, ToolResponse
from clawcode.llm.tools import file_guard
from clawcode.llm.tools.advanced import EditTool, PatchTool, WriteTool
from clawcode.llm.tools.file_ops import ViewTool

from test_agent import MockProvider, cleanup_test_environment, setup_test_environment


@pytest.fixture(autouse=True)
def _clean_read_registry():
    file_guard._reset_for_tests()
    yield
    file_guard._reset_for_tests()


# ── WS1: edit-tool uniqueness ────────────────────────────────────────


def _edit_call(path: str, replacements: list[dict]) -> ToolCall:
    import json

    return ToolCall(
        id="e1",
        name="edit",
        input=json.dumps({"file_path": path, "replacements": replacements}),
    )


def _ctx(tmp_path: Path, session_id: str = "s1") -> ToolContext:
    return ToolContext(
        session_id=session_id,
        message_id="",
        working_directory=str(tmp_path),
    )


def _seed_read(tmp_path: Path, f: Path, session_id: str = "s1") -> None:
    """Satisfy the stale-read guard for tests focused on other behavior."""
    file_guard.record_read(session_id, f)


@pytest.mark.asyncio
async def test_edit_ambiguous_match_errors(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x = 1\ny = 2\nx = 1\n")
    _seed_read(tmp_path, f)
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
    _seed_read(tmp_path, f)
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
    _seed_read(tmp_path, f)
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
    _seed_read(tmp_path, f)
    tool = EditTool()
    resp = await tool.run(
        _edit_call(str(f), [{"old_text": "NOT PRESENT", "new_text": "z"}]),
        _ctx(tmp_path),
    )
    assert resp.is_error, resp.content
    assert "not found" in resp.content


# ── WS2: stale-read guard, enforced in the TOOL layer ────────────────


def _view_call(path: str) -> ToolCall:
    import json

    return ToolCall(id="v1", name="view", input=json.dumps({"file_path": path}))


def _write_call(path: str, content: str) -> ToolCall:
    import json

    return ToolCall(id="w1", name="write", input=json.dumps({"file_path": path, "content": content}))


@pytest.mark.asyncio
async def test_tool_edit_without_read_rejected(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    resp = await EditTool().run(
        _edit_call(str(f), [{"old_text": "hello", "new_text": "bye"}]),
        _ctx(tmp_path),
    )
    assert resp.is_error and "has not been read" in resp.content
    assert f.read_text() == "hello\n"


@pytest.mark.asyncio
async def test_tool_edit_after_view_passes(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    ctx = _ctx(tmp_path)
    vresp = await ViewTool().run(_view_call(str(f)), ctx)
    assert not vresp.is_error
    resp = await EditTool().run(
        _edit_call(str(f), [{"old_text": "hello", "new_text": "bye"}]), ctx
    )
    assert not resp.is_error, resp.content
    assert f.read_text() == "bye\n"


@pytest.mark.asyncio
async def test_tool_edit_rejected_when_changed_on_disk(tmp_path: Path) -> None:
    import os

    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    ctx = _ctx(tmp_path)
    await ViewTool().run(_view_call(str(f)), ctx)
    # Someone else (another tab / the user) modifies it.
    f.write_text("hello there\n")
    st = f.stat()
    os.utime(f, (st.st_atime, st.st_mtime + 5))
    resp = await EditTool().run(
        _edit_call(str(f), [{"old_text": "hello", "new_text": "bye"}]), ctx
    )
    assert resp.is_error and "changed on disk" in resp.content


@pytest.mark.asyncio
async def test_own_write_refreshes_baseline(tmp_path: Path) -> None:
    """Consecutive edits by the same session pass without re-reading."""
    f = tmp_path / "a.txt"
    f.write_text("a\nb\n")
    ctx = _ctx(tmp_path)
    await ViewTool().run(_view_call(str(f)), ctx)
    r1 = await EditTool().run(_edit_call(str(f), [{"old_text": "a", "new_text": "x"}]), ctx)
    assert not r1.is_error, r1.content
    r2 = await EditTool().run(_edit_call(str(f), [{"old_text": "b", "new_text": "y"}]), ctx)
    assert not r2.is_error, r2.content
    assert f.read_text() == "x\ny\n"


@pytest.mark.asyncio
async def test_write_new_file_exempt_but_overwrite_guarded(tmp_path: Path) -> None:
    new = tmp_path / "new.txt"
    r1 = await WriteTool().run(_write_call(str(new), "created\n"), _ctx(tmp_path))
    assert not r1.is_error, r1.content
    assert new.read_text() == "created\n"

    # A DIFFERENT session has never read it → overwrite is blocked.
    r2 = await WriteTool().run(_write_call(str(new), "clobbered\n"), _ctx(tmp_path, "s2"))
    assert r2.is_error and "has not been read" in r2.content
    assert new.read_text() == "created\n"


@pytest.mark.asyncio
async def test_execute_code_write_path_is_guarded(tmp_path: Path) -> None:
    """The hole this refactor closed: execute_code calls WriteTool directly.

    Because the guard now lives in the tool, that caller inherits it — no
    agent loop involved.
    """
    f = tmp_path / "a.txt"
    f.write_text("original\n")
    # Simulate execute_code's RPC: WriteTool.run with the session's context.
    resp = await WriteTool().run(_write_call(str(f), "overwritten\n"), _ctx(tmp_path))
    assert resp.is_error and "has not been read" in resp.content
    assert f.read_text() == "original\n", "unguarded overwrite must not happen"


@pytest.mark.asyncio
async def test_cross_session_stale_edit_blocked(tmp_path: Path) -> None:
    """Two parallel tabs: B cannot clobber a file A just changed."""
    f = tmp_path / "shared.txt"
    f.write_text("line one\nline two\n")
    ctx_a, ctx_b = _ctx(tmp_path, "tabA"), _ctx(tmp_path, "tabB")
    await ViewTool().run(_view_call(str(f)), ctx_a)
    await ViewTool().run(_view_call(str(f)), ctx_b)

    ra = await EditTool().run(_edit_call(str(f), [{"old_text": "line one", "new_text": "A was here"}]), ctx_a)
    assert not ra.is_error, ra.content

    rb = await EditTool().run(_edit_call(str(f), [{"old_text": "line two", "new_text": "B was here"}]), ctx_b)
    assert rb.is_error and "changed on disk" in rb.content
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

"""Per-command bash permission grants in the ACP shim.

"Allow for this session" on bash used to mean allow EVERY bash command,
because PermissionService keys session grants by tool name. The shim now
scopes the sticky grant to the exact command, and withholds it entirely
from commands that obviously write to disk (those bypass the tool-layer
guards and /revert cannot undo them).
"""

import pytest

from clawcode.acp_shim.agent import (
    _command_mutates_files,
    _normalize_command,
    _permission_command,
)
from clawcode.core.permission import PermissionRequest


# ── command classification ───────────────────────────────────────────


@pytest.mark.parametrize(
    "cmd",
    [
        "npm test",
        "npm run build",
        "tsc --noEmit",
        "git status",
        "git diff HEAD",
        "pytest tests/ -q",
        "ls -la",
        "grep -rn foo src/",
        "cat package.json",
        "node --version",
        # Redirect noise must not read as a mutation.
        "pytest -q 2>&1",
        "npm test > /dev/null",
        "tsc --noEmit 2>/dev/null",
    ],
)
def test_read_only_commands_are_not_mutating(cmd: str) -> None:
    assert not _command_mutates_files(cmd), f"{cmd!r} misclassified as mutating"


@pytest.mark.parametrize(
    "cmd",
    [
        "echo hi > file.txt",
        "echo hi >> file.txt",
        "sed -i 's/a/b/' file.txt",
        "sed -Ei 's/a/b/' file.txt",
        "rm -rf build",
        "mv a b",
        "cp a b",
        "chmod +x script.sh",
        "mkdir -p out",
        "touch newfile",
        "git checkout main",
        "git reset --hard",
        "git commit -m 'x'",
        "npm install lodash",
        "pip install requests",
        "grep foo x | tee out.txt",
        "perl -i -pe 's/a/b/' f",
    ],
)
def test_mutating_commands_detected(cmd: str) -> None:
    assert _command_mutates_files(cmd), f"{cmd!r} not caught as mutating"


def test_normalize_collapses_whitespace() -> None:
    assert _normalize_command("npm   test\n") == "npm test"
    assert _normalize_command("  git  status ") == "git status"


# ── request → command extraction ─────────────────────────────────────


def test_permission_command_extracts_bash_command() -> None:
    req = PermissionRequest(
        tool_name="bash",
        description="Execute shell command: npm  test",
        input={"command": "npm  test"},
        session_id="s1",
    )
    assert _permission_command(req) == "npm test"


def test_permission_command_none_for_non_bash_tools() -> None:
    for tool in ("write", "edit", "patch", "execute_code"):
        req = PermissionRequest(
            tool_name=tool,
            description="x",
            input={"command": "npm test"},
            session_id="s1",
        )
        assert _permission_command(req) is None, tool


def test_permission_command_none_without_command() -> None:
    req = PermissionRequest(tool_name="bash", description="x", input={}, session_id="s1")
    assert _permission_command(req) is None


# ── the grant memory itself ──────────────────────────────────────────


def test_approved_commands_are_per_command_not_per_tool() -> None:
    """Approving `npm test` must not silently approve `rm -rf /`."""
    from clawcode.acp_shim.agent import _SessionState

    state = _SessionState(clawcode_session_id="c1", agent=None, permissions=None)
    state.approved_commands.add(_normalize_command("npm test"))

    assert _normalize_command("npm  test") in state.approved_commands
    assert _normalize_command("rm -rf /") not in state.approved_commands
    assert _normalize_command("npm test -- --watch") not in state.approved_commands


# ── no second prompt for an already-approved command ─────────────────


class _StubPermissions:
    def __init__(self) -> None:
        self.granted: list[tuple[str, bool]] = []

    async def grant(self, request_id: str, session_scoped: bool = False) -> None:
        self.granted.append((request_id, session_scoped))

    async def deny(self, request_id: str) -> None:
        self.granted.append((request_id, False))


class _StubConn:
    """Records permission round-trips; answers with the given option_id."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[list[str]] = []

    async def request_permission(self, session_id, tool_call, options):
        self.prompts.append([o.option_id for o in options])

        class _Outcome:
            option_id = self.answer

        class _Resp:
            outcome = _Outcome()

        return _Resp()


@pytest.mark.asyncio
async def test_approved_command_does_not_prompt_again() -> None:
    from clawcode.acp_shim.agent import ClawCodeAcpAgent, _SessionState

    agent = ClawCodeAcpAgent()
    conn = _StubConn(answer="allow_command")
    agent._conn = conn
    perms = _StubPermissions()
    state = _SessionState(clawcode_session_id="c1", agent=None, permissions=perms)

    def req() -> PermissionRequest:
        return PermissionRequest(
            tool_name="bash",
            description="Execute shell command: npm test",
            input={"command": "npm test"},
            session_id="s1",
        )

    # First call prompts, and the sticky option is offered.
    await agent._forward_permission("acp1", state, req())
    assert len(conn.prompts) == 1
    assert "allow_command" in conn.prompts[0]
    assert "npm test" in state.approved_commands

    # Second identical call is auto-granted with NO round-trip.
    await agent._forward_permission("acp1", state, req())
    assert len(conn.prompts) == 1, "approved command must not prompt again"
    assert len(perms.granted) == 2

    # A different command still prompts.
    other = PermissionRequest(
        tool_name="bash",
        description="Execute shell command: rm -rf build",
        input={"command": "rm -rf build"},
        session_id="s1",
    )
    await agent._forward_permission("acp1", state, other)
    assert len(conn.prompts) == 2
    # ...and a mutating command is never offered the sticky option.
    assert "allow_command" not in conn.prompts[1]

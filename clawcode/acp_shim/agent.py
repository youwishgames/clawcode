"""ClawCode ACP agent: maps ClawCode's agent loop onto the Agent Client Protocol.

Translation table:
    AgentEventType.CONTENT_DELTA -> session/update agent_message_chunk
    AgentEventType.THINKING      -> session/update agent_thought_chunk
    AgentEventType.TOOL_USE      -> session/update tool_call (start, with file locations)
    AgentEventType.TOOL_RESULT   -> session/update tool_call_update (diff card for file writes)
    AgentEventType.USAGE         -> session/update usage_update
    todo_write tool (shim-added) -> session/update plan (live task checklist)
    AgentEventType.ERROR         -> agent message + end of turn
    PermissionService callback   -> session/request_permission
    config option "model"        -> per-session engine switch (rebuilds the agent)
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from acp import (
    PROTOCOL_VERSION,
    Agent as AcpAgent,
    AuthenticateResponse,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    text_block,
    update_agent_message_text,
    update_agent_thought_text,
)
from acp.helpers import (
    plan_entry,
    start_tool_call,
    tool_content,
    tool_diff_content,
    update_available_commands,
    update_plan,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AvailableCommand,
    Implementation,
    PermissionOption,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
    ToolCallLocation,
    ToolCallUpdate,
    ToolKind,
    UsageUpdate,
)

from ..core.permission import PermissionRequest, PermissionService
from ..llm.tools.base import BaseTool, ToolCall, ToolContext, ToolInfo, ToolResponse

logger = structlog.get_logger(__name__)

# ClawCode tool name -> ACP tool kind (drives the icon in the client panel).
_TOOL_KINDS: dict[str, ToolKind] = {
    "view": "read",
    "batch_view": "read",
    "ls": "read",
    "glob": "search",
    "grep": "search",
    "write": "edit",
    "edit": "edit",
    "patch": "edit",
    "bash": "execute",
    "execute_code": "execute",
    "fetch": "fetch",
    "web_search": "fetch",
    "think": "think",
}

_TOOL_OUTPUT_LIMIT = 20_000  # chars of tool output forwarded to the panel
_SNAPSHOT_LIMIT = 262_144  # max bytes of a file we snapshot for diff cards

# Tools whose results should render as a diff card (file before/after).
_FILE_WRITE_TOOLS = {"write", "edit", "patch"}

# Known context-window sizes for the usage meter (tokens). Conservative default.
_CONTEXT_WINDOWS = {
    "z-ai/glm-5.2": 1_000_000,
    "z-ai/glm-5-turbo": 200_000,
    "deepseek/deepseek-chat": 128_000,
}
_DEFAULT_CONTEXT_WINDOW = 200_000

_TODO_TOOL_NAME = "todo_write"

_TODO_PROMPT_NUDGE = (
    "\n\n## Task checklist\n"
    f"For any multi-step task, maintain a live checklist with the `{_TODO_TOOL_NAME}` tool: "
    "call it when you start (all steps `pending`, the first `in_progress`), and call it again "
    "with the full updated list whenever a step's status changes. Keep exactly one step "
    "`in_progress` at a time. Skip it for trivial single-step requests."
)


def _tool_kind(tool_name: str | None) -> ToolKind:
    return _TOOL_KINDS.get((tool_name or "").lower(), "other")


def _tool_title(tool_name: str | None, tool_input: Any) -> str:
    """Short human-readable row title, e.g. ``bash: git status``."""
    name = tool_name or "tool"
    detail = ""
    if isinstance(tool_input, dict):
        if name == "mcp_call":
            server = tool_input.get("server") or ""
            tool = tool_input.get("tool") or tool_input.get("tool_name") or ""
            if server or tool:
                return f"{server}.{tool}" if server and tool else f"mcp: {server or tool}"
        for key in ("command", "file_path", "path", "pattern", "query", "url"):
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                detail = val.strip().splitlines()[0]
                break
    if len(detail) > 80:
        detail = detail[:77] + "..."
    return f"{name}: {detail}" if detail else name


def _extract_file_path(tool_input: Any, working_directory: str) -> str | None:
    """Absolute file path a tool call touches, if any (ACP wants absolute paths)."""
    if not isinstance(tool_input, dict):
        return None
    raw = tool_input.get("file_path") or tool_input.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        p = Path(working_directory or ".") / p
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


def _read_file_for_diff(path: str) -> str | None:
    """File text for a diff card, or None if unreadable/too big/binary."""
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size > _SNAPSHOT_LIMIT:
            return None
        return p.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return None


class ShimTodoTool(BaseTool):
    """Task-checklist tool the shim adds so the engine can drive the panel's plan view.

    Execution is a no-op; the shim intercepts the TOOL_USE event and turns it
    into an ACP ``plan`` session update instead of a tool row.
    """

    def info(self) -> ToolInfo:
        return ToolInfo(
            name=_TODO_TOOL_NAME,
            description=(
                "Maintain a live task checklist shown to the user. Send the FULL list every "
                "call (it replaces the previous one). Use for multi-step tasks; keep exactly "
                "one item in_progress at a time and mark items completed as you finish them."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The complete, current checklist.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Short imperative step description.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
            required=["todos"],
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResponse:
        todos = call.get_input_dict().get("todos")
        count = len(todos) if isinstance(todos, list) else 0
        return ToolResponse(content=f"Checklist updated ({count} items).")


@dataclass
class _SessionState:
    """Per-ACP-session state (one ClawCode session + agent per ACP session)."""

    clawcode_session_id: str
    agent: Any
    permissions: PermissionService
    working_directory: str = "."
    model: str = ""
    current_tool: tuple[str, str] | None = None  # (tool_call_id, tool_name)
    turn_task: asyncio.Task | None = None
    # tool_call_id -> (abs_path, pre-edit text) captured at TOOL_USE for diff cards
    file_snapshots: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    # abs_path -> (existed_before, first-seen content) — session baseline for
    # /revert. Content None while existed=True means unsnapshotable (too
    # large/binary): report it but never touch it on revert.
    file_baseline: dict[str, tuple[bool, str | None]] = field(default_factory=dict)
    # tool_call_ids of todo_write calls (suppressed as tool rows, sent as plan updates)
    todo_call_ids: set[str] = field(default_factory=set)
    extra: dict[str, Any] = field(default_factory=dict)


class ClawCodeAcpAgent(AcpAgent):
    """ACP agent implementation backed by a ClawCode runtime."""

    def __init__(self) -> None:
        self._conn: Client | None = None
        self._app_ctx: Any = None
        self._app_lock = asyncio.Lock()
        self._sessions: dict[str, _SessionState] = {}

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    # ── lifecycle ────────────────────────────────────────────────────

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=min(PROTOCOL_VERSION, protocol_version),
            agent_capabilities=AgentCapabilities(),
            agent_info=Implementation(
                name="clawcode-acp",
                title="ClawCode",
                version="0.1.0",
            ),
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        return AuthenticateResponse()

    async def _ensure_app(self, cwd: str) -> Any:
        """Create the ClawCode app context once; later sessions reuse it."""
        async with self._app_lock:
            if self._app_ctx is not None:
                current = str(getattr(self._app_ctx.settings, "working_directory", "") or "")
                if cwd and current and cwd != current:
                    logger.warning(
                        "acp_shim: app context already bound to another cwd",
                        bound=current,
                        requested=cwd,
                    )
                return self._app_ctx
            from ..app import create_app

            # Settings discovery reads Path.cwd() (project .clawcode.json, else
            # ~/.clawcode.json). Chdir to the session cwd so the shim resolves
            # config exactly like `clawcode` launched from that project.
            if cwd:
                import os

                with contextlib.suppress(OSError):
                    os.chdir(cwd)

            self._app_ctx = await create_app(
                working_dir=cwd or None,
                debug=False,
                launch_working_directory=cwd or None,
            )
            return self._app_ctx

    # ── sessions ─────────────────────────────────────────────────────

    def _build_agent(self, app_ctx: Any, permissions: PermissionService) -> Any:
        """Build a ClawCode agent (full tui_coder runtime + the shim's todo tool)."""
        from ..llm.runtime_bundle import build_coder_runtime

        bundle = build_coder_runtime(
            settings=app_ctx.settings,
            session_service=app_ctx.session_service,
            message_service=app_ctx.message_service,
            permissions=permissions,
            plugin_manager=getattr(app_ctx, "plugin_manager", None),
            lsp_manager=getattr(app_ctx, "lsp_manager", None),
            for_claw_mode=None,
            style="tui_coder",
        )
        bundle.tools.append(ShimTodoTool())
        if bundle.system_prompt:
            bundle.system_prompt = bundle.system_prompt + _TODO_PROMPT_NUDGE
        return bundle.make_plain_agent(permission_client=permissions)

    def _model_options(self, app_ctx: Any) -> tuple[str, list[SessionConfigSelectOption]]:
        """(current model, selectable models) from enabled provider configs."""
        settings = app_ctx.settings
        current = settings.get_agent_config("coder").model
        options: list[SessionConfigSelectOption] = []
        seen: set[str] = set()
        for provider_key, cfg in (settings.providers or {}).items():
            if getattr(cfg, "disabled", False) or not getattr(cfg, "api_key", None):
                continue
            for model in getattr(cfg, "models", None) or []:
                if model in seen:
                    continue
                seen.add(model)
                options.append(
                    SessionConfigSelectOption(value=model, name=f"{model} ({provider_key})")
                )
        if current not in seen:
            options.insert(0, SessionConfigSelectOption(value=current, name=current))
        return current, options

    def _model_config_option(self, app_ctx: Any) -> SessionConfigOptionSelect:
        current, options = self._model_options(app_ctx)
        return SessionConfigOptionSelect(
            type="select",
            id="model",
            name="Model",
            description="Engine used by ClawCode (from .clawcode.json provider configs)",
            current_value=current,
            options=options,
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        app_ctx = await self._ensure_app(cwd)
        session = await app_ctx.session_service.create("ACP chat")

        permissions = PermissionService()
        agent = self._build_agent(app_ctx, permissions)

        state = _SessionState(
            clawcode_session_id=session.id,
            agent=agent,
            permissions=permissions,
            working_directory=str(getattr(app_ctx.settings, "working_directory", "") or cwd or "."),
            model=app_ctx.settings.get_agent_config("coder").model,
        )
        acp_session_id = session.id
        self._sessions[acp_session_id] = state

        async def on_permission(request: PermissionRequest) -> None:
            await self._forward_permission(acp_session_id, state, request)

        permissions.register_callback(on_permission)

        # Best-effort: advertise loaded skills as slash commands after the
        # response goes out (can't notify before the session id is known).
        asyncio.get_running_loop().create_task(
            self._announce_commands(acp_session_id, app_ctx)
        )
        return NewSessionResponse(
            session_id=acp_session_id,
            config_options=[self._model_config_option(app_ctx)],
        )

    async def set_config_option(
        self, config_id: str, session_id: str, value: Any, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Handle the panel's model picker: rebuild this session's agent on the new engine."""
        if config_id != "model" or not isinstance(value, str):
            return None
        state = self._sessions.get(session_id)
        if state is None or self._app_ctx is None:
            return None

        settings = self._app_ctx.settings
        agent_config = settings.get_agent_config("coder")
        agent_config.model = value
        # Point provider_key at whichever enabled provider lists this model,
        # so resolve_provider_from_model picks the right credentials.
        for provider_key, cfg in (settings.providers or {}).items():
            if getattr(cfg, "disabled", False):
                continue
            if value in (getattr(cfg, "models", None) or []):
                agent_config.provider_key = provider_key
                break

        state.agent = self._build_agent(self._app_ctx, state.permissions)
        state.model = value
        logger.info("acp_shim: model switched", model=value)
        return SetSessionConfigOptionResponse(
            config_options=[self._model_config_option(self._app_ctx)]
        )

    async def _announce_commands(self, session_id: str, app_ctx: Any) -> None:
        try:
            pm = getattr(app_ctx, "plugin_manager", None)
            if pm is None or self._conn is None:
                return
            commands = [
                AvailableCommand(
                    name=skill.name,
                    description=(skill.description or "").strip()[:200] or skill.name,
                )
                for skill in pm.get_all_skills()
            ]
            # Shim built-ins (handled without the LLM).
            commands.append(AvailableCommand(
                name="revert",
                description="List files this session changed (/revert) or restore them all (/revert all)",
            ))
            if commands:
                await self._conn.session_update(
                    session_id=session_id,
                    update=update_available_commands(commands),
                )
        except Exception:
            logger.warning("acp_shim: failed to announce commands", exc_info=True)

    # ── permissions ──────────────────────────────────────────────────

    async def _forward_permission(
        self,
        acp_session_id: str,
        state: _SessionState,
        request: PermissionRequest,
    ) -> None:
        """Forward a ClawCode permission request to the ACP client and resolve it."""
        if self._conn is None:
            await state.permissions.deny(request.request_id)
            return

        # Attach to the live tool-call row when it is the same tool.
        tool_call_id = f"perm-{request.request_id}"
        if state.current_tool and state.current_tool[1] == request.tool_name:
            tool_call_id = state.current_tool[0]

        tool_call = ToolCallUpdate(
            tool_call_id=tool_call_id,
            title=request.description or request.tool_name,
            kind=_tool_kind(request.tool_name),
            status="pending",
            raw_input=request.input if isinstance(request.input, dict) else None,
        )
        options = [
            PermissionOption(option_id="allow_once", name="Allow once", kind="allow_once"),
        ]
        # Bash/execute never gets a session-scoped grant: `sed -i`/`echo >`
        # ride the execute kind past acceptEdits and dodge /revert tracking,
        # so a session-wide bash allow is effectively allow-everything.
        if _tool_kind(request.tool_name) != "execute":
            options.append(
                PermissionOption(option_id="allow_always", name="Allow for this session", kind="allow_always"),
            )
        options.append(
            PermissionOption(option_id="reject_once", name="Reject", kind="reject_once"),
        )
        try:
            response = await self._conn.request_permission(
                session_id=acp_session_id,
                tool_call=tool_call,
                options=options,
            )
            option_id = getattr(response.outcome, "option_id", None)
        except Exception:
            logger.warning("acp_shim: permission round-trip failed", exc_info=True)
            option_id = None

        if option_id == "allow_once":
            await state.permissions.grant(request.request_id)
        elif option_id == "allow_always":
            await state.permissions.grant(request.request_id, session_scoped=True)
        else:
            await state.permissions.deny(request.request_id)

    # ── prompting ────────────────────────────────────────────────────

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        state = self._sessions.get(session_id)
        if state is None or self._conn is None:
            return PromptResponse(stop_reason="refusal")

        text = self._extract_text(prompt)
        if not text.strip():
            return PromptResponse(stop_reason="end_turn")

        # Session file-change revert — handled by the shim, never the LLM.
        stripped = text.strip().lower()
        if stripped in ("/revert", "/revert all"):
            reply = self._handle_revert(state, apply=stripped == "/revert all")
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message_text(reply),
            )
            return PromptResponse(stop_reason="end_turn")

        # Same slash-command expansion the TUI applies (/brain, /plugin, ...).
        text, plugin_reply = self._dispatch_slash(text)
        if plugin_reply is not None:
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message_text(plugin_reply),
            )
            return PromptResponse(stop_reason="end_turn")

        task = asyncio.get_running_loop().create_task(
            self._run_turn(session_id, state, text)
        )
        state.turn_task = task
        try:
            await task
        except asyncio.CancelledError:
            return PromptResponse(stop_reason="cancelled")
        finally:
            state.turn_task = None
        return PromptResponse(stop_reason="end_turn")

    @staticmethod
    def _handle_revert(state: "_SessionState", apply: bool) -> str:
        """List (or restore) every file this session changed via write/edit/patch.

        Baseline = the file's content the FIRST time a file-write tool touched
        it this session. Bash-made changes are not tracked. Unsnapshotable
        files (too large / binary) are reported but never modified.
        """
        if not state.file_baseline:
            return (
                "No file changes tracked in this session. "
                "(Only write/edit/patch tool changes are tracked — bash changes are not.)"
            )

        lines: list[str] = []
        errors: list[str] = []
        for path, (existed, content) in sorted(state.file_baseline.items()):
            if not apply:
                label = (
                    "created" if not existed
                    else "modified — too large/binary, cannot revert" if content is None
                    else "modified"
                )
                lines.append(f"- {path} ({label})")
                continue
            try:
                p = Path(path)
                if not existed:
                    if p.is_file():
                        p.unlink()
                    lines.append(f"- deleted {path} (did not exist before this session)")
                elif content is None:
                    lines.append(f"- skipped {path} (no snapshot — too large/binary)")
                else:
                    p.write_text(content, encoding="utf-8")
                    lines.append(f"- restored {path}")
            except OSError as exc:
                errors.append(f"- FAILED {path}: {exc}")

        if not apply:
            return (
                f"{len(state.file_baseline)} file(s) changed by this session:\n"
                + "\n".join(lines)
                + "\n\nRun `/revert all` to restore every file to its pre-session state."
                + " Only write/edit/patch tool changes are tracked — bash changes are not."
            )

        state.file_baseline.clear()
        out = "Reverted this session's file changes:\n" + "\n".join(lines)
        if errors:
            out += "\n\nErrors:\n" + "\n".join(errors)
        return out

    def _dispatch_slash(self, text: str) -> tuple[str, str | None]:
        """Expand /skill messages; returns (llm_text, direct_reply_or_None)."""
        try:
            from ..plugin.slash import dispatch_slash

            settings = self._app_ctx.settings
            pm = getattr(self._app_ctx, "plugin_manager", None)
            slash = dispatch_slash(text, settings, pm)
            if slash is None:
                return text, None
            if slash.consume_without_llm:
                return text, slash.plugin_reply or ""
            return slash.llm_user_text, None
        except Exception:
            logger.warning("acp_shim: slash dispatch failed", exc_info=True)
            return text, None

    @staticmethod
    def _plan_entries(tool_input: Any) -> list[Any]:
        """ACP plan entries from a todo_write call's input; [] if malformed."""
        if isinstance(tool_input, str):
            import json

            try:
                tool_input = json.loads(tool_input)
            except (ValueError, TypeError):
                return []
        if not isinstance(tool_input, dict):
            return []
        todos = tool_input.get("todos")
        if not isinstance(todos, list):
            return []
        entries = []
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            content = str(todo.get("content") or "").strip()
            status = todo.get("status")
            if not content:
                continue
            if status not in ("pending", "in_progress", "completed"):
                status = "pending"
            entries.append(plan_entry(content, status=status))
        return entries

    @staticmethod
    def _extract_text(blocks: list[Any]) -> str:
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict):
                value = block.get("text")
            else:
                value = getattr(block, "text", None)
            if isinstance(value, str) and value:
                parts.append(value)
        return "\n".join(parts)

    async def _run_turn(self, session_id: str, state: _SessionState, text: str) -> None:
        """Consume ClawCode agent events and stream them as session/update."""
        from ..llm.agent import AgentEventType

        conn = self._conn
        assert conn is not None
        streamed_any_content = False
        # ClawCode's loop can yield the same tool event more than once
        # (streaming + final pass) — dedupe by tool_call_id so the panel
        # shows one row per call.
        started_tool_calls: set[str] = set()
        finished_tool_calls: set[str] = set()

        async for event in state.agent.run(state.clawcode_session_id, text):
            if event.type == AgentEventType.CONTENT_DELTA:
                if event.content:
                    streamed_any_content = True
                    await conn.session_update(
                        session_id=session_id,
                        update=update_agent_message_text(event.content),
                    )
            elif event.type == AgentEventType.THINKING:
                thought = (
                    (event.message.content if event.message else None) or event.content
                )
                if thought:
                    await conn.session_update(
                        session_id=session_id,
                        update=update_agent_thought_text(thought),
                    )
            elif event.type == AgentEventType.TOOL_USE:
                tool_call_id = event.tool_call_id or f"tool-{uuid4().hex}"
                tool_name = (event.tool_name or "tool").lower()
                state.current_tool = (tool_call_id, event.tool_name or "tool")
                if tool_call_id in started_tool_calls:
                    continue
                started_tool_calls.add(tool_call_id)

                # todo_write becomes the panel's live checklist, not a tool row.
                if tool_name == _TODO_TOOL_NAME:
                    state.todo_call_ids.add(tool_call_id)
                    entries = self._plan_entries(event.tool_input)
                    if entries:
                        await conn.session_update(
                            session_id=session_id, update=update_plan(entries)
                        )
                    continue

                file_path = _extract_file_path(event.tool_input, state.working_directory)
                if file_path and tool_name in _FILE_WRITE_TOOLS:
                    # Snapshot pre-edit content now; diffed against the file
                    # after the tool completes.
                    pre_text = _read_file_for_diff(file_path)
                    state.file_snapshots[tool_call_id] = (file_path, pre_text)
                    # First touch this session? Record the baseline for /revert.
                    if file_path not in state.file_baseline:
                        state.file_baseline[file_path] = (
                            Path(file_path).is_file(),
                            pre_text,
                        )
                await conn.session_update(
                    session_id=session_id,
                    update=start_tool_call(
                        tool_call_id,
                        title=_tool_title(event.tool_name, event.tool_input),
                        kind=_tool_kind(event.tool_name),
                        status="in_progress",
                        locations=[ToolCallLocation(path=file_path)] if file_path else None,
                        raw_input=event.tool_input if isinstance(event.tool_input, dict) else None,
                    ),
                )
            elif event.type == AgentEventType.TOOL_RESULT:
                tool_call_id = event.tool_call_id or (
                    state.current_tool[0] if state.current_tool else f"tool-{uuid4().hex}"
                )
                if tool_call_id in state.todo_call_ids or tool_call_id in finished_tool_calls:
                    continue
                finished_tool_calls.add(tool_call_id)

                content = None
                snapshot = state.file_snapshots.pop(tool_call_id, None)
                if snapshot and not event.is_error:
                    path, old_text = snapshot
                    new_text = _read_file_for_diff(path)
                    if new_text is not None and new_text != old_text:
                        content = [tool_diff_content(path, new_text, old_text=old_text)]
                if content is None:
                    result_text = event.tool_result or ""
                    if len(result_text) > _TOOL_OUTPUT_LIMIT:
                        result_text = result_text[:_TOOL_OUTPUT_LIMIT] + "\n… (truncated)"
                    if result_text:
                        content = [tool_content(text_block(result_text))]
                await conn.session_update(
                    session_id=session_id,
                    update=update_tool_call(
                        tool_call_id,
                        status="failed" if event.is_error else "completed",
                        content=content,
                    ),
                )
            elif event.type == AgentEventType.USAGE:
                usage = event.usage
                if usage is not None:
                    used = (
                        getattr(usage, "input_tokens", 0)
                        + getattr(usage, "output_tokens", 0)
                        + getattr(usage, "cache_read_tokens", 0)
                    )
                    if used > 0:
                        await conn.session_update(
                            session_id=session_id,
                            update=UsageUpdate(
                                session_update="usage_update",
                                used=used,
                                size=_CONTEXT_WINDOWS.get(state.model, _DEFAULT_CONTEXT_WINDOW),
                            ),
                        )
            elif event.type == AgentEventType.RESPONSE:
                if not streamed_any_content and event.message and event.message.content:
                    await conn.session_update(
                        session_id=session_id,
                        update=update_agent_message_text(event.message.content),
                    )
            elif event.type == AgentEventType.ERROR:
                await conn.session_update(
                    session_id=session_id,
                    update=update_agent_message_text(f"⚠️ {event.error or 'Unknown error'}"),
                )
                return

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self._sessions.get(session_id)
        if state and state.turn_task and not state.turn_task.done():
            state.turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.turn_task

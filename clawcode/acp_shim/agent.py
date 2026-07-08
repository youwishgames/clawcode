"""ClawCode ACP agent: maps ClawCode's agent loop onto the Agent Client Protocol.

Translation table:
    AgentEventType.CONTENT_DELTA -> session/update agent_message_chunk
    AgentEventType.THINKING      -> session/update agent_thought_chunk
    AgentEventType.TOOL_USE      -> session/update tool_call (start)
    AgentEventType.TOOL_RESULT   -> session/update tool_call_update
    AgentEventType.ERROR         -> agent message + end of turn
    PermissionService callback   -> session/request_permission
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
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
    start_tool_call,
    tool_content,
    update_available_commands,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AvailableCommand,
    Implementation,
    PermissionOption,
    ToolCallUpdate,
    ToolKind,
)

from ..core.permission import PermissionRequest, PermissionService

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


def _tool_kind(tool_name: str | None) -> ToolKind:
    return _TOOL_KINDS.get((tool_name or "").lower(), "other")


def _tool_title(tool_name: str | None, tool_input: Any) -> str:
    """Short human-readable row title, e.g. ``bash: git status``."""
    name = tool_name or "tool"
    detail = ""
    if isinstance(tool_input, dict):
        for key in ("command", "file_path", "path", "pattern", "query", "url"):
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                detail = val.strip().splitlines()[0]
                break
    if len(detail) > 80:
        detail = detail[:77] + "..."
    return f"{name}: {detail}" if detail else name


@dataclass
class _SessionState:
    """Per-ACP-session state (one ClawCode session + agent per ACP session)."""

    clawcode_session_id: str
    agent: Any
    permissions: PermissionService
    current_tool: tuple[str, str] | None = None  # (tool_call_id, tool_name)
    turn_task: asyncio.Task | None = None
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

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        from ..llm.runtime_bundle import build_coder_runtime

        app_ctx = await self._ensure_app(cwd)
        session = await app_ctx.session_service.create("ACP chat")

        permissions = PermissionService()
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
        agent = bundle.make_plain_agent(permission_client=permissions)

        state = _SessionState(
            clawcode_session_id=session.id,
            agent=agent,
            permissions=permissions,
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
        return NewSessionResponse(session_id=acp_session_id)

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
            PermissionOption(option_id="allow_always", name="Allow for this session", kind="allow_always"),
            PermissionOption(option_id="reject_once", name="Reject", kind="reject_once"),
        ]
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
                state.current_tool = (tool_call_id, event.tool_name or "tool")
                if tool_call_id in started_tool_calls:
                    continue
                started_tool_calls.add(tool_call_id)
                await conn.session_update(
                    session_id=session_id,
                    update=start_tool_call(
                        tool_call_id,
                        title=_tool_title(event.tool_name, event.tool_input),
                        kind=_tool_kind(event.tool_name),
                        status="in_progress",
                        raw_input=event.tool_input if isinstance(event.tool_input, dict) else None,
                    ),
                )
            elif event.type == AgentEventType.TOOL_RESULT:
                tool_call_id = event.tool_call_id or (
                    state.current_tool[0] if state.current_tool else f"tool-{uuid4().hex}"
                )
                if tool_call_id in finished_tool_calls:
                    continue
                finished_tool_calls.add(tool_call_id)
                result_text = event.tool_result or ""
                if len(result_text) > _TOOL_OUTPUT_LIMIT:
                    result_text = result_text[:_TOOL_OUTPUT_LIMIT] + "\n… (truncated)"
                await conn.session_update(
                    session_id=session_id,
                    update=update_tool_call(
                        tool_call_id,
                        status="failed" if event.is_error else "completed",
                        content=[tool_content(text_block(result_text))] if result_text else None,
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

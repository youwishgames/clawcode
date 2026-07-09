"""Test client: drive the clawcode ACP shim over stdio like vscode-acp would.

Usage: python acp_test_client.py <cwd-for-agent> "<prompt>" [--auto-allow]
Prints every session/update; answers permission requests with allow_once.
"""

import asyncio
import sys
from typing import Any

from acp import PROTOCOL_VERSION, Client, spawn_agent_process, text_block
from acp.schema import (
    AllowedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    ToolCallUpdate,
)

SHIM_PY = "/Users/tyler/Code/clawcode/.venv/bin/python"


class TestClient(Client):
    def __init__(self) -> None:
        self.permission_requests: list[str] = []

    def on_connect(self, conn: Any) -> None:
        pass

    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[PermissionOption], **kwargs: Any
    ) -> RequestPermissionResponse:
        title = getattr(tool_call, "title", "?")
        self.permission_requests.append(title)
        print(f"\n[PERMISSION REQUEST] {title} | options: {[o.option_id for o in options]} -> allow_once")
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id="allow_once"))

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        kind = type(update).__name__
        if kind == "AgentMessageChunk":
            text = getattr(getattr(update, "content", None), "text", "")
            print(text, end="", flush=True)
        elif kind == "AgentThoughtChunk":
            text = getattr(getattr(update, "content", None), "text", "")
            print(f"\n[THOUGHT] {text[:200]}", flush=True)
        elif kind == "ToolCallStart":
            locs = [l.path for l in (update.locations or [])]
            print(f"\n[TOOL START] {update.title} (kind={update.kind}, status={update.status}, locations={locs})", flush=True)
        elif kind == "ToolCallProgress":
            parts = []
            for c in update.content or []:
                cn = type(c).__name__
                if cn == "FileEditToolCallContent":
                    old = c.old_text or ""
                    print(f"\n[DIFF CARD] {c.path} old={len(old)}ch new={len(c.new_text)}ch", flush=True)
                parts.append(cn)
            print(f"\n[TOOL UPDATE] id={update.tool_call_id} status={update.status} content={parts}", flush=True)
        elif kind == "AgentPlanUpdate":
            for e in update.entries:
                print(f"\n[PLAN] [{e.status}] {e.content}", flush=True)
        elif kind == "UsageUpdate":
            print(f"\n[USAGE] {update.used}/{update.size} tokens", flush=True)
        elif kind == "AvailableCommandsUpdate":
            names = [c.name for c in update.available_commands]
            print(f"\n[COMMANDS] {len(names)} slash commands: {names[:8]}...", flush=True)
        else:
            print(f"\n[UPDATE {kind}]", flush=True)

    # Unused client capabilities: decline politely.
    async def write_text_file(self, *a: Any, **k: Any) -> None:  # pragma: no cover
        return None

    async def read_text_file(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise RuntimeError("fs not supported in test client")


async def main() -> None:
    cwd, prompt_text = sys.argv[1], sys.argv[2]
    client = TestClient()
    async with spawn_agent_process(
        client, SHIM_PY, "-m", "clawcode.acp_shim", "--cwd", cwd, cwd="/Users/tyler/Code/clawcode"
    ) as (conn, proc):
        init = await conn.initialize(protocol_version=PROTOCOL_VERSION)
        info = getattr(init, "agent_info", None)
        print(f"[INIT] protocol v{init.protocol_version}, agent: {getattr(info, 'name', '?')}")

        session = await conn.new_session(cwd=cwd, mcp_servers=[])
        print(f"[SESSION] {session.session_id}")
        for opt in session.config_options or []:
            vals = [o.value for o in (opt.options or [])][:6]
            print(f"[CONFIG] {opt.id}={opt.current_value} options={vals}")

        result = await asyncio.wait_for(
            conn.prompt(session_id=session.session_id, prompt=[text_block(prompt_text)]),
            timeout=120,
        )
        print(f"\n[DONE] stop_reason={result.stop_reason} permissions_asked={len(client.permission_requests)}")


if __name__ == "__main__":
    asyncio.run(main())

# ACP Shim — chat-panel UI for ClawCode

`clawcode/acp_shim/` exposes ClawCode as an [Agent Client Protocol](https://agentclientprotocol.com) agent, so any ACP client (the vscode-acp "ACP Client" extension in VS Code/Cursor, Zed, etc.) can drive it through a chat panel: streaming responses, thinking display, tool-call rows, and permission prompts.

## Install

The shim needs the optional `agent-client-protocol` dependency:

```bash
.venv/bin/pip install -e ".[acp]"
```

## Run

```bash
.venv/bin/python -m clawcode.acp_shim [--cwd /path/to/project]
```

stdout carries JSON-RPC; all logging goes to stderr. `--cwd` pre-warms the runtime; otherwise the working directory comes from the client's `session/new` request. The shim chdirs to that directory before loading settings, so config discovery matches launching `clawcode` from the project (project `.clawcode.json`, else `~/.clawcode.json`).

## Client setup (Cursor / VS Code)

1. Install the extension: `cursor --install-extension formulahendry.acp-client`
2. Register the agent in user settings:

```json
"acp.agents": {
    "ClawCode": {
        "command": "/Users/tyler/Code/clawcode/.venv/bin/python",
        "args": ["-m", "clawcode.acp_shim"],
        "env": {}
    }
}
```

3. Open the **ACP Client** panel from the activity bar and connect to **ClawCode**.

## What maps to what

| ClawCode | ACP |
|---|---|
| `AgentEventType.CONTENT_DELTA` | `agent_message_chunk` (streaming text) |
| `AgentEventType.THINKING` | `agent_thought_chunk` |
| `AgentEventType.TOOL_USE` / `TOOL_RESULT` | `tool_call` / `tool_call_update` rows |
| `PermissionService` callback | `session/request_permission` (allow once / allow for session / reject) |
| Plugin skills (`pm.get_all_skills()`) | `available_commands_update` (slash autocomplete) |
| `/skill` messages | expanded via `plugin.slash.dispatch_slash`, same as the TUI |

Each ACP session creates one ClawCode session (visible later in the TUI sidebar) with a full `tui_coder`-style runtime: system prompt, skills, MCP servers, and per-session permission grants.

## Regression drill

`tools/acp_test_client.py` drives the shim over real stdio JSON-RPC like a panel would (streams updates, auto-allows permissions):

```bash
.venv/bin/python tools/acp_test_client.py /path/to/project "Reply with exactly the word PONG and nothing else."
```

## Limitations

- One working directory per shim process (first session wins; a warning is logged if a later session asks for a different cwd).
- Model switching from the panel is not wired; the engine comes from `.clawcode.json` (`agents.coder.model`).
- `session/load` (history replay) is not implemented.

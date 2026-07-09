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

1. Install the owned fork of the ACP client extension (adds diff cards, tool output bodies, live checklist, usage meter, clickable locations, inline permission cards, parallel chat tabs, Claude Code-style layout):
   ```bash
   git clone https://github.com/youwishgames/vscode-acp && cd vscode-acp
   npm install && npm run compile && node tools/check_webview.js
   npx vsce package --allow-missing-repository --no-dependencies
   cursor --install-extension acp-client-*.vsix
   ```
   (Upstream `formulahendry.acp-client` from the marketplace also works, with plainer rendering. The fork carries a bumped version so the marketplace won't auto-update over it.)

   **Set `"acp.autoApprovePermissions": "ask"`.** If it is `"allowAll"`, no permission prompt ever reaches you — every tool call, including file-mutating bash commands, auto-approves silently.
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
| `AgentEventType.TOOL_USE` / `TOOL_RESULT` | `tool_call` / `tool_call_update` rows, with `locations` (clickable file paths) |
| file writes (`write`/`edit`/`patch`) | diff card (`FileEditToolCallContent`): file snapshotted at TOOL_USE, diffed after |
| `todo_write` tool (added by the shim) | `plan` update — live task checklist in the panel |
| `AgentEventType.USAGE` | `usage_update` (token meter vs. the model's context window) |
| `PermissionService` callback | `session/request_permission` (see **Permissions** below) |
| Plugin skills (`pm.get_all_skills()`) | `available_commands_update` (slash autocomplete) |
| `/skill` messages | expanded via `plugin.slash.dispatch_slash`, same as the TUI |
| `/revert`, `/revert all` | handled by the shim itself, before slash dispatch |
| config option `model` | model picker — lists models from enabled providers in `.clawcode.json`; switching rebuilds the session's agent on the new engine |

Each ACP session creates one ClawCode session (visible later in the TUI sidebar) with a full `tui_coder`-style runtime: system prompt, skills, MCP servers, and per-session permission grants. The fork's panel supports several **concurrent live sessions** on one shim process — the "+" button in a chat tab opens a new one rather than freezing the old.

## Permissions

`PermissionService` keys session-scoped grants by **tool name**. For most tools that is what you want. For `bash` it is not: "allow for this session" would mean *allow every shell command for the rest of the session*.

So the shim scopes bash grants to the **exact command string** (whitespace-normalized), tracked in `_SessionState.approved_commands`:

- `allow_once` — run this one call.
- `allow_command` — *"Always allow `npm test` this session."* Offered for `bash` only. An identical later command is granted with no round-trip to the client. Withheld entirely from commands that `_command_mutates_files()` flags as writing to disk (`>`, `sed -i`, `rm`/`mv`/`cp`/`tee`, `git checkout|reset|commit|…`, `npm install`, `pip install`, `perl -i`, …).
- `allow_always` — the old tool-scoped session grant. Offered for non-`bash`, non-`execute_code` tools only.
- `reject`.

`_command_mutates_files()` is a **mistake-catcher, not a security boundary**. `python -c "open('x','w')"` defeats it trivially. Bash bypasses the tool-layer file guards entirely (as it does in Claude Code); per-call visibility is the mitigation.

## `/revert`

`_SessionState.file_baseline` snapshots each file the session is about to touch (existence + contents) at `TOOL_USE` time, for `write`/`edit`/`patch`. `/revert` lists what would be undone; `/revert all` restores every snapshotted file and deletes the ones the session created. It does **not** undo anything `bash` did.

## Regression drill

`tools/acp_test_client.py` drives the shim over real stdio JSON-RPC like a panel would (streams updates, auto-allows permissions):

```bash
.venv/bin/python tools/acp_test_client.py /path/to/project "Reply with exactly the word PONG and nothing else."
```

## Limitations

- One working directory per shim process (first session wins; a warning is logged if a later session asks for a different cwd). Parallel chat tabs therefore share one checkout — the per-session stale-read guard (`clawcode/llm/tools/file_guard.py`) stops them clobbering each other's edits, but git state is shared, so **commit from one tab only**.
- `/revert` and the file guards cover `write`/`edit`/`patch`. `bash` mutations are invisible to both.
- Model switching from the panel changes the in-memory agent config only — it does not persist to `.clawcode.json`, and it applies process-wide (new sessions inherit the last switch).
- `session/load` (history replay) is not implemented.

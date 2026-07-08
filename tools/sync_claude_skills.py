#!/usr/bin/env python3
"""Sync Claude Code skills/commands into a ClawCode plugin.

Collects command markdown files from a project's `.claude/commands/` and the
user-wide `~/.claude/commands/` (project wins on name collisions), prepends a
thin ClawCode adaptation note, and writes them as a Claude Code-compatible
plugin at `<workspace>/.claw/plugins/youwish-skills/` where ClawCode's plugin
discovery picks them up as slash commands.

Re-syncing after skills change upstream is one command:

    python tools/sync_claude_skills.py "/Users/tyler/Code/YOU WISH/you-wish-management"

Optional second arg = target workspace (defaults to the source project).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PLUGIN_NAME = "youwish-skills"

ADAPTER_NOTE = """\
> **ClawCode adaptation:** This skill was written for Claude Code. Tool names
> like `mcp__supabase__execute_sql` do not exist here — instead call the
> `mcp_call` tool with `server="supabase"`, `tool="execute_sql"`, and
> `arguments={"query": "..."}`. Trigger.dev and Next.js MCP tools are not
> wired; skip those steps or use bash equivalents. Everything else applies.

"""


def collect_sources(project: Path) -> dict[str, Path]:
    """Map command name -> source file. Project commands override user-wide."""
    sources: dict[str, Path] = {}
    for root in (Path.home() / ".claude" / "commands", project / ".claude" / "commands"):
        if root.is_dir():
            for md in sorted(root.glob("*.md")):
                sources[md.stem] = md
    return sources


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: sync_claude_skills.py <source-project> [target-workspace]")
    project = Path(sys.argv[1]).expanduser().resolve()
    target_ws = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else project

    sources = collect_sources(project)
    if not sources:
        sys.exit(f"no commands found under {project}/.claude/commands or ~/.claude/commands")

    plugin_root = target_ws / ".claw" / "plugins" / PLUGIN_NAME
    commands_dir = plugin_root / "commands"
    if commands_dir.exists():
        shutil.rmtree(commands_dir)
    commands_dir.mkdir(parents=True)

    manifest_dir = plugin_root / ".claude-plugin"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "plugin.json").write_text(json.dumps({
        "name": PLUGIN_NAME,
        "description": "You Wish Games operational skills, synced from Claude Code",
        "version": "1.0.0",
    }, indent=2))

    for name, src in sorted(sources.items()):
        text = src.read_text(encoding="utf-8")
        # Insert the adapter note after YAML frontmatter if present, else at top.
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                head, body = text[: end + 4], text[end + 4 :]
                text = head + "\n" + ADAPTER_NOTE + body.lstrip("\n")
            else:
                text = ADAPTER_NOTE + text
        else:
            text = ADAPTER_NOTE + text
        (commands_dir / f"{name}.md").write_text(text, encoding="utf-8")

    print(f"synced {len(sources)} skills -> {commands_dir}")


if __name__ == "__main__":
    main()

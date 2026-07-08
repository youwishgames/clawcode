"""ACP shim: expose ClawCode as an Agent Client Protocol (ACP) agent.

Lets ACP clients (Zed, the vscode-acp extension in VS Code/Cursor, etc.)
drive ClawCode through a chat panel: streaming messages, tool-call display,
and permission prompts all map onto the ACP protocol.

Run with:
    python -m clawcode.acp_shim [--cwd <working-directory>]

Requires the optional dependency ``agent-client-protocol`` (PyPI).
"""

__version__ = "0.1.0"

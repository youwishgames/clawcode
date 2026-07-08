"""Entry point: run ClawCode as an ACP agent over stdio.

Usage (from an ACP client's agent config):
    /path/to/.venv/bin/python -m clawcode.acp_shim --cwd /path/to/project

stdout carries the JSON-RPC protocol; all logging goes to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def _configure_stderr_logging() -> None:
    """Route ALL logging to stderr — stdout belongs to the ACP protocol."""
    import structlog

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


async def _amain(cwd: str | None) -> None:
    from acp import run_agent
    from acp.stdio import stdio_streams

    from .agent import ClawCodeAcpAgent

    # Bind the protocol to the real stdout FIRST, then point sys.stdout at
    # stderr so any stray print() inside clawcode cannot corrupt JSON-RPC.
    output_stream, input_stream = await stdio_streams()
    sys.stdout = sys.stderr

    agent = ClawCodeAcpAgent()
    if cwd:
        # Pre-warm the app context so the first session starts fast and any
        # config problem surfaces at connect time instead of mid-chat.
        await agent._ensure_app(cwd)

    await run_agent(agent, input_stream, output_stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClawCode as an ACP agent over stdio")
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for the ClawCode runtime (defaults to the client session cwd)",
    )
    args = parser.parse_args()

    _configure_stderr_logging()
    try:
        asyncio.run(_amain(args.cwd))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Mirror the CLI entry point's DB cleanup.
        try:
            from ..db import close_database

            asyncio.run(close_database())
        except Exception:
            pass


if __name__ == "__main__":
    main()

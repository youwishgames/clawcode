"""Minimal stdio MCP server exposing SQL access to the You Wish brain database.

Replaces the hosted Supabase MCP (which requires OAuth) with an owned,
zero-dependency-on-Anthropic path: ClawCode -> this server -> Postgres.

Usage (configured in .clawcode.json under mcp_servers):
    command: <venv>/bin/python
    args:   ["/Users/tyler/Code/clawcode/tools/brain_mcp/server.py"]
    env:    ["DATABASE_URI=postgresql://..."]

Exposes one tool:
    execute_sql(query) -> JSON rows (SELECT) or affected rowcount (DML)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import psycopg
from psycopg.rows import dict_row
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("supabase")

MAX_ROWS = 500


def _connect() -> psycopg.Connection:
    uri = os.environ.get("DATABASE_URI")
    if not uri:
        print("DATABASE_URI env var is required", file=sys.stderr)
        raise RuntimeError("DATABASE_URI env var is required")
    return psycopg.connect(uri, row_factory=dict_row, autocommit=True)


@mcp.tool()
def execute_sql(query: str) -> str:
    """Execute a SQL statement against the You Wish brain Postgres database.

    SELECT statements return up to 500 rows as JSON. INSERT/UPDATE/DELETE
    return the affected row count. The brain schema is `brain` (plans, goals,
    strategies, decisions, lessons, research, context, ideas, metrics, ...).
    """
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(query)
            if cur.description is not None:
                rows = cur.fetchmany(MAX_ROWS)
                payload: dict[str, Any] = {"rows": rows, "row_count": len(rows)}
                if len(rows) == MAX_ROWS:
                    payload["truncated_at"] = MAX_ROWS
                return json.dumps(payload, default=str, ensure_ascii=False)
            return json.dumps({"affected_rows": cur.rowcount})
    except Exception as e:  # surface DB errors to the model, don't crash the server
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()

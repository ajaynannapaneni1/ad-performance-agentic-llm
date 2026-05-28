"""
SQL tool — executes safe, read-only queries against the advertising SQLite database.
Supports campaign-level, ad-group-level, and time-series aggregations.
"""

import sqlite3
import os
import re
import json
from pathlib import Path

DB_PATH = os.environ.get("AD_DB_PATH", str(Path(__file__).parents[2] / "data" / "ads.db"))

# Read-only SQL guard — rejects any mutation statements
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)

SCHEMA_HINT = """
Tables:
  campaigns(campaign_id, name, channel, objective, budget_usd, start_date, end_date)
  ad_groups(ad_group_id, campaign_id, name, targeting, bid_strategy)
  daily_metrics(
    date, campaign_id, ad_group_id,
    impressions, clicks, spend_usd, revenue_usd, conversions,
    ctr REAL,        -- clicks / impressions
    cpc REAL,        -- spend / clicks
    roas REAL        -- revenue / spend
  )
"""


def execute_ad_sql_query(sql: str) -> str:
    """Execute a read-only SQL query and return results as JSON."""
    if _FORBIDDEN.search(sql):
        return "Error: Only SELECT queries are allowed."

    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()

        if not rows:
            return "No results found."

        results = [dict(row) for row in rows]
        # Truncate to 50 rows max to keep context window manageable
        truncated = len(results) > 50
        output = {
            "row_count": len(results),
            "truncated": truncated,
            "rows": results[:50],
        }
        return json.dumps(output, indent=2, default=str)

    except sqlite3.Error as e:
        return f"SQL error: {e}\n\nSchema reference:\n{SCHEMA_HINT}"


def get_schema() -> str:
    return SCHEMA_HINT

"""
Generates a realistic synthetic advertising dataset (SQLite) for demo/testing.
Produces ~6 months of daily metrics across 8 campaigns / 24 ad groups.
"""

import sqlite3
import random
import math
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "ads.db"

CHANNELS = ["Search", "Display", "YouTube", "Shopping"]
OBJECTIVES = ["Conversions", "Awareness", "Traffic", "ROAS"]
BID_STRATEGIES = ["Target CPA", "Target ROAS", "Manual CPC", "Maximize Conversions"]
TARGETING_OPTIONS = [
    "Broad Match", "Phrase Match", "Exact Match",
    "Audience: Retargeting", "Audience: In-market", "Placement: News",
]

CAMPAIGNS = [
    (1, "Brand_Search", "Search", "Conversions", 5000),
    (2, "NonBrand_Search", "Search", "ROAS", 15000),
    (3, "Display_Prospecting", "Display", "Awareness", 8000),
    (4, "Display_Retargeting", "Display", "Conversions", 6000),
    (5, "YouTube_Awareness", "YouTube", "Awareness", 10000),
    (6, "Shopping_Core", "Shopping", "ROAS", 20000),
    (7, "Shopping_Promo", "Shopping", "ROAS", 12000),
    (8, "Search_Competitor", "Search", "Traffic", 7000),
]

AD_GROUPS = [
    # (ad_group_id, campaign_id, name, targeting, bid_strategy)
    (1, 1, "Brand_Exact", "Exact Match", "Manual CPC"),
    (2, 1, "Brand_Phrase", "Phrase Match", "Target CPA"),
    (3, 1, "Brand_Broad", "Broad Match", "Maximize Conversions"),
    (4, 2, "Generic_Keywords", "Broad Match", "Target ROAS"),
    (5, 2, "Category_Keywords", "Phrase Match", "Target ROAS"),
    (6, 2, "Long_Tail", "Exact Match", "Target CPA"),
    (7, 3, "Contextual_News", "Placement: News", "Manual CPC"),
    (8, 3, "Interest_InMarket", "Audience: In-market", "Maximize Conversions"),
    (9, 4, "Site_Visitors", "Audience: Retargeting", "Target CPA"),
    (10, 4, "Cart_Abandoners", "Audience: Retargeting", "Target ROAS"),
    (11, 5, "Skippable_15s", "Audience: In-market", "Manual CPC"),
    (12, 5, "NonSkippable_6s", "Audience: In-market", "Manual CPC"),
    (13, 6, "All_Products", "Broad Match", "Target ROAS"),
    (14, 6, "High_Margin_SKUs", "Exact Match", "Target ROAS"),
    (15, 7, "Seasonal_Promo", "Audience: In-market", "Target ROAS"),
    (16, 7, "Clearance_Items", "Broad Match", "Maximize Conversions"),
    (17, 8, "Competitor_Brand_A", "Exact Match", "Manual CPC"),
    (18, 8, "Competitor_Brand_B", "Phrase Match", "Manual CPC"),
    (19, 2, "Branded_Generic", "Broad Match", "Target ROAS"),
    (20, 3, "Lookalike_30d", "Audience: In-market", "Manual CPC"),
    (21, 4, "High_LTV_Retargeting", "Audience: Retargeting", "Target CPA"),
    (22, 6, "New_Arrivals", "Broad Match", "Target ROAS"),
    (23, 7, "Flash_Sale", "Audience: In-market", "Maximize Conversions"),
    (24, 8, "Competitor_Brand_C", "Exact Match", "Manual CPC"),
]

# Per-campaign baseline performance parameters
CAMPAIGN_PARAMS = {
    1: dict(base_impressions=12000, ctr=0.08, cpc=0.85, cvr=0.12, avg_order=95),
    2: dict(base_impressions=45000, ctr=0.028, cpc=1.40, cvr=0.04, avg_order=110),
    3: dict(base_impressions=120000, ctr=0.003, cpc=0.25, cvr=0.008, avg_order=100),
    4: dict(base_impressions=30000, ctr=0.018, cpc=0.60, cvr=0.06, avg_order=105),
    5: dict(base_impressions=80000, ctr=0.010, cpc=0.12, cvr=0.003, avg_order=90),
    6: dict(base_impressions=60000, ctr=0.015, cpc=0.70, cvr=0.05, avg_order=130),
    7: dict(base_impressions=40000, ctr=0.014, cpc=0.65, cvr=0.045, avg_order=115),
    8: dict(base_impressions=20000, ctr=0.025, cpc=1.80, cvr=0.02, avg_order=100),
}


def seasonal_multiplier(d: date) -> float:
    """Q4 uplift + weekly pattern."""
    month_factor = {10: 1.15, 11: 1.30, 12: 1.45}.get(d.month, 1.0)
    dow_factor = [1.0, 1.05, 1.08, 1.06, 1.10, 0.85, 0.80][d.weekday()]
    return month_factor * dow_factor


def generate_database(db_path: Path = DB_PATH, seed: int = 42):
    random.seed(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
    CREATE TABLE campaigns (
        campaign_id INTEGER PRIMARY KEY,
        name TEXT, channel TEXT, objective TEXT, budget_usd REAL,
        start_date TEXT, end_date TEXT
    );
    CREATE TABLE ad_groups (
        ad_group_id INTEGER PRIMARY KEY,
        campaign_id INTEGER, name TEXT, targeting TEXT, bid_strategy TEXT,
        FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id)
    );
    CREATE TABLE daily_metrics (
        date TEXT, campaign_id INTEGER, ad_group_id INTEGER,
        impressions INTEGER, clicks INTEGER, spend_usd REAL, revenue_usd REAL, conversions INTEGER,
        ctr REAL, cpc REAL, roas REAL,
        PRIMARY KEY (date, campaign_id, ad_group_id)
    );
    CREATE INDEX idx_daily_date ON daily_metrics(date);
    CREATE INDEX idx_daily_campaign ON daily_metrics(campaign_id);
    """)

    start = date(2024, 1, 1)
    end = date(2024, 6, 30)

    cur.executemany(
        "INSERT INTO campaigns VALUES (?,?,?,?,?,?,?)",
        [(cid, name, ch, obj, bud, str(start), str(end)) for cid, name, ch, obj, bud in CAMPAIGNS],
    )
    cur.executemany(
        "INSERT INTO ad_groups VALUES (?,?,?,?,?)",
        AD_GROUPS,
    )

    rows = []
    d = start
    while d <= end:
        season = seasonal_multiplier(d)
        for ag_id, camp_id, *_ in AD_GROUPS:
            p = CAMPAIGN_PARAMS[camp_id]
            noise = lambda: random.gauss(1.0, 0.08)
            impr = max(100, int(p["base_impressions"] / 24 * season * noise()))
            clicks = max(0, int(impr * p["ctr"] * noise()))
            spend = round(clicks * p["cpc"] * noise(), 2)
            convs = max(0, int(clicks * p["cvr"] * noise()))
            revenue = round(convs * p["avg_order"] * noise(), 2)
            ctr = round(clicks / impr, 5) if impr else 0
            cpc = round(spend / clicks, 3) if clicks else 0
            roas = round(revenue / spend, 3) if spend else 0
            rows.append((str(d), camp_id, ag_id, impr, clicks, spend, revenue, convs, ctr, cpc, roas))
        d += timedelta(days=1)

    cur.executemany(
        "INSERT INTO daily_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    con.close()

    total_days = (end - start).days + 1
    print(f"✅ Database created: {db_path}")
    print(f"   Campaigns: {len(CAMPAIGNS)} | Ad groups: {len(AD_GROUPS)} | Days: {total_days}")
    print(f"   Daily metric rows: {len(rows):,}")


if __name__ == "__main__":
    generate_database()

"""Generate a synthetic ticket export in the same schema as the real one.

The real UTampa ITS ticket data is not included in this repository. This
script writes data/raw/tickets.csv with made-up tickets so the pipeline can
run end to end. Numbers produced from this file will not match the results in
the README.

Usage:
    python3 scripts/make_sample_data.py            # 1,200 tickets
    python3 scripts/make_sample_data.py --rows 5000
"""

import argparse
import csv
import math
import os
import random
from datetime import datetime, timedelta

COLUMNS = [
    "State", "Category", "Subcategory", "Assignee Name",
    "Created At (Timestamp)", "Updated At (Timestamp)",
    "To Assignment", "Service Time To Assignment",
    "Last Assignee change at", "Last State change at",
    "To First Response (Business)", "To First Response (Elapsed)",
    "To Resolution (Business)", "To Resolution (Elapsed)",
    "To Closure", "Service Time To Closure",
    "# Reassignments", "Resolved At", "Closed At", "Year", "Month",
]

# (team, share of tickets, extra breach risk)
TEAMS = [
    ("Service Desk", 0.46, 0.0),
    ("Front Desk Queue", 0.19, 0.3),
    ("Academic Solutions", 0.11, 0.2),
    ("Classroom Media Services", 0.05, 1.6),
    ("Technical Support", 0.04, 0.6),
    ("Asset Management", 0.04, 0.5),
    ("System Services", 0.03, 0.9),
    ("Network Services", 0.03, 1.0),
    ("Information Security", 0.03, 1.1),
    ("Enterprise Applications", 0.02, 0.8),
]

CATEGORIES = {
    "Accounts & Access": ["Password Reset", "MFA", "Account Setup", "Other", ""],
    "Software and Applications": ["Office365", "Licensing", "Other", ""],
    "General Inquiry": ["SPAM", "Admissions", ""],
    "Internet & Connectivity": ["Wired", "Wireless", ""],
    "Email Requests": [""],
    "Learning Platform": ["Course Access", ""],
    "Classroom Support": ["Hardware", "Projector", ""],
    "Printing": ["Print Release", ""],
    "Hardware": ["Peripheral", "Laptop", ""],
}

TS = "%Y-%m-%d %H:%M:%S"


def fmt_duration(minutes):
    """Match the export: under a day is HH:MM:SS, a day or more is decimal days."""
    if minutes < 1440:
        total = int(minutes * 60)
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    return f"{minutes / 1440:.9f}"


def pick_team(rng):
    r, acc = rng.random(), 0.0
    for name, share, risk in TEAMS:
        acc += share
        if r <= acc:
            return name, risk
    return TEAMS[0][0], TEAMS[0][2]


def make_ticket(rng):
    created = datetime(2023, 1, 1) + timedelta(minutes=rng.randint(0, 365 * 24 * 60 - 1))
    team, team_risk = pick_team(rng)
    category = rng.choice(list(CATEGORIES))
    subcategory = rng.choice(CATEGORIES[category])

    reassignments = min(int(rng.expovariate(1.4)) + 1, 9)
    to_assign = rng.lognormvariate(3.0, 1.4)            # minutes
    to_first_response = to_assign + rng.lognormvariate(3.2, 1.6)
    weekend = created.weekday() >= 5

    # Breach risk rises with slow first response, handoffs, weekend creation and team.
    z = (-4.1 + team_risk + 0.55 * (reassignments - 1)
         + 0.45 * math.log1p(to_first_response / 60) + (0.4 if weekend else 0.0))
    breached = rng.random() < 1 / (1 + math.exp(-z))

    # The label in src/preprocess.py is "closed more than 14 days after creation".
    close_days = rng.uniform(14.5, 45) if breached else rng.uniform(0.5, 13.5)
    resolution_days = max(0.003, close_days - rng.uniform(1.5, 7))
    resolved = created + timedelta(days=resolution_days)
    closed = created + timedelta(days=close_days)
    last_assignee_change = created + timedelta(minutes=to_assign + rng.uniform(0, 600))

    return {
        "State": "Closed",
        "Category": category,
        "Subcategory": subcategory,
        "Assignee Name": team,
        "Created At (Timestamp)": created.strftime(TS),
        "Updated At (Timestamp)": closed.strftime(TS),
        "To Assignment": fmt_duration(to_assign),
        "Service Time To Assignment": "00:00:00",
        "Last Assignee change at": last_assignee_change.strftime(TS),
        "Last State change at": closed.strftime(TS),
        "To First Response (Business)": fmt_duration(to_first_response * 0.4),
        "To First Response (Elapsed)": fmt_duration(to_first_response),
        "To Resolution (Business)": fmt_duration(resolution_days * 1440 * 0.4),
        "To Resolution (Elapsed)": fmt_duration(resolution_days * 1440),
        "To Closure": fmt_duration((closed - created).total_seconds() / 60),
        "Service Time To Closure": f"{(closed - created).days * 0.33:.4f}",
        "# Reassignments": str(reassignments),
        "Resolved At": resolved.strftime(TS),
        "Closed At": closed.strftime(TS),
        "Year": str(created.year),
        "Month": str(created.month),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/raw/tickets.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [make_ticket(rng) for _ in range(args.rows)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    breaches = sum(
        1 for r in rows
        if (datetime.strptime(r["Closed At"], TS) - datetime.strptime(r["Created At (Timestamp)"], TS)).days > 14
    )
    print(f"Wrote {len(rows)} synthetic tickets to {args.out} ({breaches} past 14 days)")


if __name__ == "__main__":
    main()

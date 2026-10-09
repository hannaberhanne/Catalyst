# preprocess.py — Catalyst Early‑Life Feature Engineering
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/tickets.csv"
OUTPUT_PATH = "data/processed/tickets_cleaned.csv"

def safe_timedelta(col):
    return pd.to_timedelta(col, errors="coerce").dt.total_seconds() / 60

def main():
    df = pd.read_csv(RAW_PATH)

    # ----------------------------------------
    # Parse timestamps
    # ----------------------------------------
    df["Created"] = pd.to_datetime(df["Created At (Timestamp)"], errors="coerce")
    df["Updated"] = pd.to_datetime(df["Updated At (Timestamp)"], errors="coerce")
    df["LastStateChange"] = pd.to_datetime(df["Last State change at"], errors="coerce")
    df["LastAssigneeChange"] = pd.to_datetime(df["Last Assignee change at"], errors="coerce")

    # ----------------------------------------
    # EARLY‑LIFE FEATURES (Catalyst)
    # ----------------------------------------
    # Time to first assignment
    if "To Assignment" in df.columns:
        df["ToAssignment_min"] = safe_timedelta(df["To Assignment"]).fillna(0)
    else:
        df["ToAssignment_min"] = 0

    # Time to first response
    if "To First Response (Elapsed)" in df.columns:
        df["ToFirstResponse_min"] = safe_timedelta(df["To First Response (Elapsed)"]).fillna(0)
    else:
        df["ToFirstResponse_min"] = 0

    df["IsWeekend"] = df["Created"].dt.dayofweek >= 5

    df["ServiceDeskEarlyRoute"] = (
        df["Assignee Name"].fillna("")
        .str.contains("Service Desk", case=False)
        & (df["ToAssignment_min"] <= 4320)
    ).astype(int)

    # Time waiting in approval BEFORE any work
    if "State" in df.columns:
        df["ApprovalWaitFlag"] = df["State"].str.contains("Approval", case=False, na=False).astype(int)
    else:
        df["ApprovalWaitFlag"] = 0

    # ----------------------------------------
    # REASSIGNMENT FEATURES (Full-life + Early-life)
    # ----------------------------------------

    # Full-life reassignment count (raw signal from dataset)
    if "# Reassignments" in df.columns:
        df["ReassignCount_Full"] = df["# Reassignments"].fillna(0).astype(int)
    else:
        df["ReassignCount_Full"] = 0

    # Early-life reassignment count (first 3 days from creation)
    if "Assignment History" in df.columns:
        # Expecting format like: "TeamA|2024-01-03;TeamB|2024-01-04;TeamC|2024-01-10"
        def count_early_reassign(history, created):
            if pd.isna(history) or pd.isna(created):
                return 0
            events = history.split(";")
            early = 0
            for e in events:
                parts = e.split("|")
                if len(parts) != 2:
                    continue
                team, ts = parts
                try:
                    t = pd.to_datetime(ts, errors="coerce")
                    if pd.notna(t) and (t - created).days <= 3:
                        early += 1
                except:
                    continue
            # Reassignments = number of changes → events minus first assignment
            return max(early - 1, 0)

        df["ReassignCount_Early3d"] = df.apply(
            lambda row: count_early_reassign(row.get("Assignment History"), row.get("Created")),
            axis=1
        )
    else:
        df["ReassignCount_Early3d"] = 0

    df["TeamSwitchCount3d"] = df["ReassignCount_Early3d"]
    df["TeamSwitchFlag3d"] = (df["TeamSwitchCount3d"] > 0).astype(int)

    # SLA Target (label)
    df["Closed"] = pd.to_datetime(df["Closed At"], errors="coerce")
    df["Resolution_days"] = (df["Closed"] - df["Created"]).dt.total_seconds() / 86400
    df["SLA_exceeded"] = (df["Resolution_days"] > 14).astype(int)

    # ----------------------------------------
    # REMOVE LEAKAGE FEATURES
    # ----------------------------------------
    leakage = [
        "Resolution_days",
        "To Resolution (Business)",
        "To Resolution (Elapsed)",
        "Service Time To Closure",
        "To Closure",
        "Closed At",
        "Resolved At",
        "Closed",
        "Resolved"
    ]
    df = df.drop(columns=[c for c in leakage if c in df.columns], errors="ignore")

    # Drop raw duration strings / historical fields we no longer reference
    duration_cols = [
        "To Assignment",
        "Service Time To Assignment",
        "To First Response (Business)",
        "To First Response (Elapsed)",
        "To Closure",
        "Assignment History"
    ]
    df = df.drop(columns=[c for c in duration_cols if c in df.columns], errors="ignore")

    # ----------------------------------------
    # Keep safe categorical features
    # ----------------------------------------
    cat_cols = ["State", "Category", "Subcategory", "Assignee Name", "Month", "Year"]
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "Unknown"
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # One-hot encoding
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ----------------------------------------
    # Drop metadata/non-ML columns
    # ----------------------------------------
    drop_cols = [
        "Created At (Timestamp)",
        "Updated At (Timestamp)",
        "Last Assignee change at",
        "Last State change at",
        "Unnamed: 21",
        "Created",
        "Updated",
        "LastStateChange",
        "LastAssigneeChange"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ----------------------------------------
    # Final export
    # ----------------------------------------
    df.to_csv(OUTPUT_PATH, index=False)
    print("Saved Catalyst-cleaned dataset →", OUTPUT_PATH)


if __name__ == "__main__":
    main()

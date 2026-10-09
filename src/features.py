# features.py
import pandas as pd
import numpy as np

def add_resolution_time(df):
    df['Resolution_days'] = (df['Closed'] - df['Created']).dt.total_seconds() / 86400
    return df


def add_sla_flag(df, threshold_days=30):
    df['SLA_exceeded'] = (df['Resolution_days'] > threshold_days).astype(int)
    return df


def convert_duration_columns(df, cols):
    for col in cols:
        df[col] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds() / 60
        df[col] = df[col].fillna(0)
    return df


def clean_reassignments(df):
    df['# Reassignments'] = df['# Reassignments'].fillna(0)
    return df


def encode_categoricals(df, cat_cols):
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def drop_unused_columns(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors='ignore')
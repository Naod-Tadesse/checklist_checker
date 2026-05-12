import pandas as pd
import re
import numpy as np

# Columns that must be globally consistent
CONSISTENCY_COLS = [
    "Checklist Type",
    "Program Category",
    "Program Type",
    "Option Type",
    "Minimum",
    "Maximum"
]

STANDARD_PATTERN = re.compile(r"^\d+(\.\d+)*$")

# Columns that must not be missing
REQUIRED_COLS = [
    "Standard",
    "Checklist Title",
    "Checklist Type",
    "Program Category",
    "Program Type",
    "Option Type",
    "Minimum",
    "Maximum"
]

def parse_weight(val):
    """
    Converts Excel or string percentage to float (0-100 scale)
    """
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        # Excel numeric percent: 0.1 -> 10
        return val * 100
    val = str(val).strip()
    if val.endswith("%"):
        try:
            return float(val.replace("%", ""))
        except ValueError:
            return None
    return None

def validate_checklist(df: pd.DataFrame) -> pd.DataFrame:
    errors = []

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Normalize the Standard column: strip whitespace so hierarchy checks
    # (is_leaf / parent) compare clean values like "1.1.1", not "1.1.1 ".
    if "Standard" in df.columns:
        df["Standard"] = df["Standard"].apply(
            lambda v: str(v).strip() if pd.notna(v) else v
        )

    # 0. Missing required column values
    for col in REQUIRED_COLS:
        if col not in df.columns:
            errors.append({
                "Standard": "N/A",
                "Error": f"Missing column: '{col}'"
            })
            continue

        missing_rows = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        for _, r in missing_rows.iterrows():
            errors.append({
                "Standard": r.get("Standard", "N/A"),
                "Error": f"Missing value in column '{col}'"
            })

    # Stop further processing if key columns missing
    if any(col not in df.columns for col in REQUIRED_COLS):
        return pd.DataFrame(errors).drop_duplicates(subset=["Standard", "Error"])

    # Helper functions
    all_standards = df["Standard"].astype(str).tolist()

    def is_leaf(s):
        s = str(s)
        return not any(v.startswith(s + ".") for v in all_standards)

    def parent_standard(s):
        s = str(s)
        return ".".join(s.split(".")[:-1]) if "." in s else None

    # 1. Standard format check
    for _, row in df.iterrows():
        val = str(row["Standard"]).strip()
        if not STANDARD_PATTERN.match(val):
            errors.append({
                "Standard": row["Standard"],
                "Error": "Invalid Standard format"
            })

    # 2. Global consistency check (case-sensitive)
    for col in CONSISTENCY_COLS:
        values = df[col].astype(str).str.strip().replace("nan", pd.NA).dropna()
        unique_values = values.unique()
        if len(unique_values) > 1:
            expected_value = unique_values[0]
            for _, row in df.iterrows():
                cell_value = str(row[col]).strip()
                if pd.isna(row[col]) or cell_value != expected_value:
                    errors.append({
                        "Standard": row["Standard"],
                        "Error": f"Inconsistent value in '{col}'. Expected '{expected_value}', found '{row[col]}'"
                    })

    # 3. Leaf standards must not have weight
    for _, row in df.iterrows():
        if is_leaf(row["Standard"]) and pd.notna(row.get("Weight (Percentage)", None)):
            errors.append({
                "Standard": row["Standard"],
                "Error": "Leaf standard should not have Weight (%)"
            })

    # 4. Non-leaf standards must have weight
    for _, row in df.iterrows():
        if not is_leaf(row["Standard"]):
            weight_num = parse_weight(row.get("Weight (Percentage)", None))
            if weight_num is None:
                errors.append({
                    "Standard": row["Standard"],
                    "Error": "Non-leaf standard must have a valid percentage weight"
                })

    # 5. Weight sum = 100% per parent
    df_non_leaf = df.copy()
    df_non_leaf["Parent"] = df_non_leaf["Standard"].astype(str).apply(parent_standard)
    df_non_leaf["WeightNum"] = df_non_leaf["Weight (Percentage)"].apply(parse_weight)

    # Only consider valid weights with a parent
    df_non_leaf = df_non_leaf[df_non_leaf["Parent"].notna() & df_non_leaf["WeightNum"].notna()]

    sums = df_non_leaf.groupby("Parent")["WeightNum"].sum()
    for parent, total in sums.items():
        if not np.isclose(total, 100.0, atol=0.01):
            bad_rows = df_non_leaf[df_non_leaf["Parent"] == parent]
            for _, r in bad_rows.iterrows():
                errors.append({
                    "Standard": r["Standard"],
                    "Error": f"Weight sum under parent '{parent}' is {total:.2f}% (must be 100%)"
                })

    return pd.DataFrame(errors).drop_duplicates(subset=["Standard", "Error"])

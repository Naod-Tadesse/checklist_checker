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
                "Error": f"The file is missing a required column called '{col}'."
            })
            continue

        missing_rows = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        for _, r in missing_rows.iterrows():
            errors.append({
                "Standard": r.get("Standard", "N/A"),
                "Error": f"This row has no value in the '{col}' column — please fill it in."
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
                "Error": f"'{val}' is not a valid item number. It should contain only "
                         f"numbers and dots, like 1, 1.2, or 1.2.3."
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
                        "Error": f"The '{col}' column should be the same in every row. "
                                 f"Most rows have '{expected_value}', but this row has '{row[col]}'."
                    })

    # 3. Leaf standards must not have weight
    for _, row in df.iterrows():
        if is_leaf(row["Standard"]) and pd.notna(row.get("Weight (Percentage)", None)):
            errors.append({
                "Standard": row["Standard"],
                "Error": "This is a lowest-level item (it has no sub-items), so it should "
                         "not have a weight percentage."
            })

    # 4. Non-leaf standards must have weight
    for _, row in df.iterrows():
        if not is_leaf(row["Standard"]):
            weight_num = parse_weight(row.get("Weight (Percentage)", None))
            if weight_num is None:
                errors.append({
                    "Standard": row["Standard"],
                    "Error": "This item has sub-items, so it must have a valid weight "
                             "percentage (for example, 25%)."
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
                    "Error": f"The weights of all items under '{parent}' should add up to "
                             f"100%, but they currently total {total:.2f}%."
                })

    # 6. Sibling order check: within each parent, the final segment of the
    #    children must run sequentially 1, 2, 3, ... in document order with no
    #    gaps, duplicates, or reordering.
    #    e.g. 10.1 -> 10.2 -> 10.3 -> 10.4 is correct;
    #         10.1 -> 10.4 is incorrect (10.2 and 10.3 are skipped).
    next_expected = {}  # parent string -> next expected final segment
    for _, row in df.iterrows():
        val = str(row["Standard"]).strip()
        if not STANDARD_PATTERN.match(val):
            continue  # malformed standards are already flagged by check #1

        parts = val.split(".")
        last = int(parts[-1])
        parent = ".".join(parts[:-1])  # "" for top-level standards
        expected = next_expected.get(parent, 1)

        if last != expected:
            prefix = f"{parent}." if parent else ""
            errors.append({
                "Standard": row["Standard"],
                "Error": f"This item is out of order. The next item here should be "
                         f"'{prefix}{expected}', but found '{val}'. Items must be listed "
                         f"in counting order ({prefix}1, then {prefix}2, then {prefix}3, ...)."
            })

        # Advance from the value actually seen so a single gap doesn't
        # cascade into flagging every following sibling.
        next_expected[parent] = last + 1

    return pd.DataFrame(errors).drop_duplicates(subset=["Standard", "Error"])

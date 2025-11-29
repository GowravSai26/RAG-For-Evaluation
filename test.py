import pandas as pd
import re

# ----------------------------
# CONFIG
# ----------------------------
OUTPUT_FILE = "data/esg_scored_out.csv"

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(OUTPUT_FILE, dtype=str).fillna("")
columns = df.columns.tolist()

print("\n=== STEP 1: DETECT QUESTION BASES ===")

def detect_bases(columns):
    bases = []

    for col in columns:
        if not col.startswith("Question_"):
            continue

        # ignore already known suffixes
        if (
            col.endswith("_Response")
            or col.endswith("_LGBT_Feedback")
            or col.endswith("_OpenAI_Response")
            or col.endswith("_Similarity_in_Insights")
        ):
            continue

        # Case A: direct base → must have matching _Response
        if f"{col}_Response" in columns:
            bases.append(col)
            continue

        # Case B: repeated questions like Question_5.1_3
        m = re.match(r"^(Question_[0-9]+(?:\.[0-9]+)*)_(\d+)$", col)
        if m:
            base_prefix, idx = m.group(1), m.group(2)
            resp_col = f"{base_prefix}_Response_{idx}"
            if resp_col in columns:
                bases.append(col)

    return sorted(bases)

bases = detect_bases(columns)
print("Detected Bases:", bases)

# ----------------------------
# STEP 2 — Verify required output columns
# ----------------------------
print("\n=== STEP 2: CHECK REQUIRED OUTPUT COLUMNS ===")

missing_required = []

for base in bases:
    ai_col = f"{base}_OpenAI_Response"
    sim_col = f"{base}_Similarity_in_Insights"

    if ai_col not in columns:
        missing_required.append(ai_col)
    if sim_col not in columns:
        missing_required.append(sim_col)

if missing_required:
    print("\n❌ Missing required columns:")
    for col in missing_required:
        print(" -", col)
else:
    print("\n✅ All required columns exist.")

# ----------------------------
# STEP 3 — Check completeness
# ----------------------------
print("\n=== STEP 3: COMPLETENESS CHECK ===")

missing_ai = []
missing_sim = []

for base in bases:
    ai_col = f"{base}_OpenAI_Response"
    sim_col = f"{base}_Similarity_in_Insights"

    # Count missing values
    ai_missing = (df[ai_col].str.strip() == "").sum()
    sim_missing = (df[sim_col].str.strip() == "").sum()

    if ai_missing > 0:
        missing_ai.append((ai_col, ai_missing))

    if sim_missing > 0:
        missing_sim.append((sim_col, sim_missing))

if missing_ai:
    print("\n❌ Missing AI Responses:")
    for col, count in missing_ai:
        print(f" - {col}: {count} missing")
else:
    print("\n✅ All AI Responses filled.")

if missing_sim:
    print("\n❌ Missing Similarity Scores:")
    for col, count in missing_sim:
        print(f" - {col}: {count} missing")
else:
    print("\n✅ All Similarity Scores filled.")

# ----------------------------
# FINAL SUMMARY
# ----------------------------
print("\n=== SUMMARY ===")

if not missing_required and not missing_ai and not missing_sim:
    print("🎉 All checks passed! Your output file is complete and valid.")
else:
    print("⚠️ Issues detected. Please review the missing counts above.")

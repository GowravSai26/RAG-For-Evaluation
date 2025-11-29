"""
ESG Survey Scoring Script

- Detects question groups dynamically
- Generates AI insights using:
    - GPT-4.1-mini
    - GPT-5.1       
- Computes similarity against LGBT feedback
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import time
from typing import Dict, List, Tuple, Optional

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------
load_dotenv(find_dotenv(usecwd=True))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

DEFAULT_MODEL = "gpt-4.1-mini"
GPT5_MODEL = "gpt-5.1"

AI_INSIGHT_SYSTEM_PROMPT = "Check chat"

GPT5_INSIGHT_SYSTEM_PROMPT = """
You are an ESG analysis model. Provide a clear, structured, detailed summarization
of LGBTQ+ DE&I responsibilities based on the employee response. Output must be a
well-formatted explanation.
""".strip()

SCORING_SYSTEM_PROMPT = """
Act as an ESG reviewer. Your task is to review insights given by AI against the provided LGBT Feedback.
If the insight contains the same analysis as the LGBT Feedback, give a score of 100%.
If partial, give a score based on semantic match.
If no match, give 0%.
Return ONLY the percentage number.
""".strip()

SCORING_USER_TEMPLATE = """
Input Data:
- Question: {question}
- Response: {response}
- AI Response: {ai}
- LGBT Feedback: {feedback}
""".strip()


# -------------------------
# Column Detection
# -------------------------
def detect_question_bases(columns: List[str]) -> List[str]:
    bases = []
    colset = set(columns)
    pattern = re.compile(r"^Question_[0-9]+(?:\.[0-9]+)*(?:_[0-9]+)*$")

    def has_response(base: str) -> bool:
        if f"{base}_Response" in colset:
            return True
        m = re.match(r"^(Question_[0-9]+(?:\.[0-9]+)*?)_(\d+)$", base)
        if m:
            prefix, idx = m.group(1), m.group(2)
            return f"{prefix}_Response_{idx}" in colset
        return False

    for c in columns:
        if c.startswith("Question_") and not any(
            c.endswith(s) for s in (
                "_Response", "_LGBT_Feedback",
                "_OpenAI_Response", "_Similarity_in_Insights",
                "_GPT5_Response", "_Similarity_in_Insights_5"
            )
        ):
            if pattern.match(c) and has_response(c):
                bases.append(c)

    return sorted(bases)


# -------------------------
# Column Insertion
# -------------------------
def insert_after(df: pd.DataFrame, after_col: str, new_cols: List[str]) -> None:
    try:
        pos = list(df.columns).index(after_col) + 1
    except ValueError:
        pos = len(df.columns)

    for col in new_cols:
        if col not in df.columns:
            df.insert(pos, col, [""] * len(df))
            pos += 1


def ensure_output_cols(df: pd.DataFrame, base: str):
    ai4 = f"{base}_OpenAI_Response"
    sim4 = f"{base}_Similarity_in_Insights"
    ai5 = f"{base}_GPT5_Response"
    sim5 = f"{base}_Similarity_in_Insights_5"

    if ai4 in df.columns and sim4 in df.columns and ai5 in df.columns and sim5 in df.columns:
        return ai4, sim4, ai5, sim5

    fb = f"{base}_LGBT_Feedback"
    resp = f"{base}_Response"
    insert_after_col = fb if fb in df.columns else resp

    if insert_after_col is None:
        insert_after_col = df.columns[-1]

    insert_after(df, insert_after_col, [ai4, sim4, ai5, sim5])
    return ai4, sim4, ai5, sim5


# -------------------------
# OpenAI Utilities
# -------------------------
def call_openai(client, messages, max_tokens=300, model_override=None) -> str:
    model = model_override or DEFAULT_MODEL
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def call_gpt5_with_retry(client, messages, retries=2):
    for _ in range(retries):
        out = call_openai(
            client,
            messages,
            max_tokens=1000,            # GPT-5 needs more space
            model_override=GPT5_MODEL
        )
        if out.strip():
            return out
        time.sleep(0.5)
    return ""


def extract_percent(text: Optional[str]) -> str:
    if not text:
        return "N/A"
    m = re.search(r"(\d{1,3})\s*%", text)
    if m:
        v = max(0, min(100, int(m.group(1))))
        return f"{v}%"
    m2 = re.search(r"\b(\d{1,3})\b", text)
    if m2:
        v = int(m2.group(1))
        if 0 <= v <= 100:
            return f"{v}%"
    return "N/A"


# -------------------------
# Main Processing
# -------------------------
def process_csv(input_path: str, output_path: str, dry_run=False, delay=0.2, log_path="data/esg_scoring_log.csv"):

    df = pd.read_csv(input_path, dtype=str).fillna("")
    bases = detect_question_bases(df.columns.tolist())
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    log: List[Dict[str, str]] = []

    for row in tqdm(range(len(df)), desc="Processing"):
        for base in bases:
            ai4_col, sim4_col, ai5_col, sim5_col = ensure_output_cols(df, base)

            resp_col = f"{base}_Response"
            fb_col = f"{base}_LGBT_Feedback"

            q = df.at[row, base].strip()
            r = df.at[row, resp_col].strip() if resp_col in df.columns else ""
            fb = df.at[row, fb_col].strip() if fb_col in df.columns else ""

            entry = {"row": row, "base": base}

            if not r:
                df.at[row, ai4_col] = ""
                df.at[row, sim4_col] = "N/A"
                df.at[row, ai5_col] = ""
                df.at[row, sim5_col] = "N/A"
                log.append(entry)
                continue

            if dry_run or client is None:
                df.at[row, ai4_col] = f"[DRY RUN]"
                df.at[row, sim4_col] = "N/A"
                df.at[row, ai5_col] = f"[DRY RUN GPT5]"
                df.at[row, sim5_col] = "N/A"
                continue

            # ---------------- GPT-4 INSIGHT ----------------
            ai4_out = call_openai(client, [
                {"role": "system", "content": AI_INSIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {q}\n\nResponse: {r}"}
            ])

            score4_raw = call_openai(client, [
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": SCORING_USER_TEMPLATE.format(
                    question=q, response=r, ai=ai4_out, feedback=fb
                )}
            ], max_tokens=60)

            sim4 = extract_percent(score4_raw)

            # ---------------- GPT-5 INSIGHT (FIXED) ----------------
            ai5_out = call_gpt5_with_retry(client, [
                {"role": "system", "content": GPT5_INSIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {q}\n\nResponse: {r}"}
            ])

            score5_raw = call_gpt5_with_retry(client, [
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": SCORING_USER_TEMPLATE.format(
                    question=q, response=r, ai=ai5_out, feedback=fb
                )}
            ])

            sim5 = extract_percent(score5_raw)

            df.at[row, ai4_col] = ai4_out
            df.at[row, sim4_col] = sim4
            df.at[row, ai5_col] = ai5_out
            df.at[row, sim5_col] = sim5

            if delay:
                time.sleep(delay)

    df.to_csv(output_path, index=False)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--out", default="data/esg_scored_out.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--log", default="data/esg_scoring_log.csv")
    args = parser.parse_args()

    process_csv(args.input, args.out, dry_run=args.dry_run, delay=args.delay, log_path=args.log)


if __name__ == "__main__":
    main()

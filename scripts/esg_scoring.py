#!/usr/bin/env python3
"""
ESG Survey Scoring Script

- Detects question groups dynamically (Question_2, Question_4_1, Question_5.1_3, etc.)
- Generates AI insights using OpenAI
- Computes similarity against LGBT feedback
- Writes output columns next to the related question block
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import time
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------
load_dotenv(find_dotenv(usecwd=True))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

DEFAULT_MODEL = "gpt-4.1-mini"

AI_INSIGHT_SYSTEM_PROMPT = "Check chat"

SCORING_SYSTEM_PROMPT = """
Act as an ESG reviewer. Your task is to review insights given by AI against the provided LGBT Feedback.
If the insight contains the same analysis as the LGBT Feedback, give a score of 100%.
If partial, give a score based on semantic match.
If no match, give 0%.
Return ONLY the percentage number (e.g., "85%").
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
            c.endswith(s) for s in ("_Response", "_LGBT_Feedback", "_OpenAI_Response", "_Similarity_in_Insights")
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


def ensure_output_cols(df: pd.DataFrame, base: str) -> Tuple[str, str]:
    ai_col = f"{base}_OpenAI_Response"
    sim_col = f"{base}_Similarity_in_Insights"

    if ai_col in df.columns and sim_col in df.columns:
        return ai_col, sim_col

    fb = f"{base}_LGBT_Feedback"
    resp = f"{base}_Response"
    insert_after_col = fb if fb in df.columns else resp

    if insert_after_col is None:
        insert_after_col = df.columns[-1]

    insert_after(df, insert_after_col, [ai_col, sim_col])
    return ai_col, sim_col


# -------------------------
# OpenAI Utilities
# -------------------------
def call_openai(client, messages, max_tokens=300) -> str:
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception:
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
            ai_col, sim_col = ensure_output_cols(df, base)

            resp_col = f"{base}_Response"
            fb_col = f"{base}_LGBT_Feedback"

            q = df.at[row, base].strip()
            r = df.at[row, resp_col].strip() if resp_col in df.columns else ""
            fb = df.at[row, fb_col].strip() if fb_col in df.columns else ""

            entry = {"row": row, "base": base, "ai": "no", "sim": "no", "error": ""}

            if not r:
                df.at[row, ai_col] = ""
                df.at[row, sim_col] = "N/A"
                log.append(entry)
                continue

            if df.at[row, ai_col].strip() and df.at[row, sim_col].strip() not in ("", "N/A"):
                entry["ai"] = entry["sim"] = "yes"
                log.append(entry)
                continue

            if dry_run or client is None:
                ai_out = f"[DRY RUN] {base}"
                sim_out = "N/A"
            else:
                ai_out = call_openai(client, [
                    {"role": "system", "content": AI_INSIGHT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {q}\n\nResponse: {r}"}
                ])

                score_raw = call_openai(client, [
                    {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": SCORING_USER_TEMPLATE.format(
                        question=q, response=r, ai=ai_out, feedback=fb
                    )}
                ], max_tokens=60)

                sim_out = extract_percent(score_raw)

            df.at[row, ai_col] = ai_out.strip()
            df.at[row, sim_col] = sim_out

            entry["ai"] = "yes"
            entry["sim"] = "yes"
            log.append(entry)

            if delay:
                time.sleep(delay)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row", "base", "ai", "sim", "error"])
        writer.writeheader()
        writer.writerows(log)


# -------------------------
# Entry
# -------------------------
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

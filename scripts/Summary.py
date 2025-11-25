import os
import json
import requests
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PPLX_API_KEY = os.getenv("PPLX_API_KEY")

SYSTEM_PROMPT = """
You are an AI system that:
1. Summarizes the answer clearly and concisely.
2. Extracts insights that relate directly to the user's question.
3. Avoids hallucination and relies strictly on provided text.
Return output in JSON with:
{
  "summary": "...",
  "insights": "..."
}
"""

# ----------------- OPENAI -----------------
def call_openai(question, answer):
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"Question: {question}\n\nAnswer: {answer}\n\nFollow system instructions."

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    try:
        return json.loads(res.choices[0].message.content)
    except:
        return {"summary": "", "insights": ""}

# ----------------- ANTHROPIC -----------------
def call_anthropic(question, answer):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    content = f"Question: {question}\n\nAnswer: {answer}\n\nFollow system instructions."

    res = client.messages.create(
        model="claude-3-5-sonnet-latest",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        max_tokens=300
    )

    try:
        return json.loads(res.content[0].text)
    except:
        return {"summary": "", "insights": ""}

# ----------------- PERPLEXITY -----------------
def call_perplexity(question, answer):
    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar-small-online",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"}
        ],
        "max_tokens": 300
    }

    res = requests.post(url, json=payload, headers=headers)
    try:
        out = res.json()["choices"][0]["message"]["content"]
        return json.loads(out)
    except:
        return {"summary": "", "insights": ""}

def Summary(question, answer, provider):
    provider = provider.strip().lower()

    if provider == "openai":
        return call_openai(question, answer)
    elif provider == "anthropic":
        return call_anthropic(question, answer)
    elif provider == "perplexity":
        return call_perplexity(question, answer)
    else:
        print("Invalid provider. Choose: openai / anthropic / perplexity")
        return {}

if __name__ == "__main__":
    print("\n=== QA Summary Generator ===\n")

    question = input("Enter Question:\n> ")
    answer = input("\nEnter Answer:\n> ")
    provider = input("\nChoose provider (openai / anthropic / perplexity):\n> ")

    print("\nProcessing...\n")

    output = Summary(question, answer, provider)
    print(json.dumps(output, indent=2))

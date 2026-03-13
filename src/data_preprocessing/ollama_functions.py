import asyncio
import re
from langchain_ollama import ChatOllama

CONTEXT_WINDOW = 4096
MAX_RETRIES = 1

# -----------------------------
# LLM configuration
# -----------------------------

llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0,
    num_predict=20,
    num_ctx=CONTEXT_WINDOW,
)


def build_conversation(rows):
    r = rows[0]  # single row

    text_content = f"""Evaluate sentiment for stock: {r['Stock_symbol']}

News:
Title: {r['Article_title']}
Summary: {r['summary']}"""[
        :CONTEXT_WINDOW
    ]  # truncate input to match context window

    conversation = [
        {
            "role": "system",
            "content": """
Forget previous instructions.

You are a financial expert with stock recommendation experience.

Score the stock sentiment from 1 to 5:
1 = negative
2 = somewhat negative
3 = neutral
4 = somewhat positive
5 = positive

Return ONLY a single number like '3'.

If the provided text is a generic market update, a list of multiple stocks without specific details for the target Stock Symbol, or contains no substantial news, default to a score of 3 (Neutral).""",
        },
        {"role": "user", "content": text_content},
    ]

    return conversation


def parse_scores(text):
    """Extract a single sentiment score from LLM output"""
    match = re.search(r"[1-5]", text)
    if not match:
        raise ValueError(f"No valid score found in LLM output: {text}")
    return int(match.group())


async def score_batch(rows):
    conversation = build_conversation(rows)  # rows will always be [row]

    for attempt in range(MAX_RETRIES):
        try:
            response = await llm.ainvoke(conversation)

            score = parse_scores(response.content)
            return [score]  # keep it as list to match the worker code

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                print(f"LLM failed: {e}")
                return None
            await asyncio.sleep(0.5)

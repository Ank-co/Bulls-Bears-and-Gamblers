"""The single prompt shared by every language model (zero-shot and fine-tuned)."""
from . import config

ANSWERS = {i: config.LABELS[i] for i in config.LABEL_IDS}   # 0 bearish, 1 bullish, 2 neutral

INSTRUCTION = (
    "Classify the market sentiment of this financial news tweet.\n"
    "bearish: negative for the price of the asset or the market.\n"
    "bullish: positive for the price of the asset or the market.\n"
    "neutral: no clear direction.\n"
    "Answer with one word: bearish, bullish or neutral.\n\n"
    "Tweet: {text}"
)


def user_message(text: str) -> str:
    return INSTRUCTION.format(text=text)


def messages(text: str) -> list[dict]:
    return [{"role": "user", "content": user_message(text)}]


# GBNF grammar for llama.cpp: the 35B model can only emit one of the three labels.
GRAMMAR = "root ::= " + " | ".join(f'"{a}"' for a in ANSWERS.values())

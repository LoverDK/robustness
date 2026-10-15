"""Prompt templates and answer extraction."""

from __future__ import annotations

import re

LOGIC_INSTRUCTION = (
    "Solve the logic problem. Use only the facts and rules given. Anything that "
    "cannot be derived from them is false.\n"
    "Reason step by step, then finish with a final line of the form "
    "'Answer: yes' or 'Answer: no'.\n\n"
)

ARITH_INSTRUCTION = (
    "Solve the math problem. Reason step by step, then finish with a final line "
    "of the form 'Answer: <number>'.\n\n"
)

_ANS = re.compile(r"answer\s*[:：]?\s*\**\s*(yes|no|true|false)\b", re.I)
_YN = re.compile(r"\b(yes|no|true|false)\b", re.I)
_NUM_ANS = re.compile(r"answer\s*[:：]?\s*\**\s*\$?(-?[\d,]*\.?\d+)", re.I)
_NUM = re.compile(r"-?[\d,]*\.?\d+")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def logic_prompt(text: str) -> str:
    return LOGIC_INSTRUCTION + text


def arith_prompt(text: str) -> str:
    return ARITH_INSTRUCTION + text


def extract_yes_no(generation: str) -> str:
    m = _ANS.findall(generation)
    if m:
        tok = m[-1].lower()
    else:
        tail = generation[-300:]
        m2 = _YN.findall(tail)
        if not m2:
            return ""
        tok = m2[-1].lower()
    return {"true": "yes", "false": "no"}.get(tok, tok)


def _norm_num(s: str) -> str:
    s = s.replace(",", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return ""
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.4f}".rstrip("0").rstrip(".")


def extract_number(generation: str) -> str:
    m = _NUM_ANS.findall(generation)
    if m:
        return _norm_num(m[-1])
    b = _BOXED.findall(generation)
    if b:
        nums = _NUM.findall(b[-1])
        if nums:
            return _norm_num(nums[-1])
    nums = _NUM.findall(generation[-200:])
    return _norm_num(nums[-1]) if nums else ""

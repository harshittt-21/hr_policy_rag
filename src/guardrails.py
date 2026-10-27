"""
Guardrails: cheap, rule-based safety checks around the RAG pipeline.

Everything here is plain Python: no LLM calls, no network, no extra cost,
and every check gives the same answer every time, so it is easy to test.

Three places in the pipeline use it:

1. BEFORE retrieval - `check_input` cleans and screens the question
   (empty, too long, a crisis message, a prompt-injection attempt, personal
   data to redact).
2. BETWEEN retrieval and the prompt - `filter_unsafe_documents` and
   `sanitize_context` stop a poisoned document from talking to the model.
3. AFTER the LLM - `check_output` checks citations, checks that every number
   in the answer appears in the documents, and catches prompt leaks.

Be honest about the limits. Regex checks catch mechanical failures. They
cannot judge whether a fluent sentence is subtly wrong (for example "may work
remotely" versus "may work remotely with manager approval"), and a determined
attacker can phrase an injection that no pattern list anticipates. Treat the
injection patterns as a speed bump; the structural defences (context wrapped
as data, output checks, a bot that only reads documents and takes no actions)
do the real work.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Set, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Settings (move to config.py if you want them in .env)
# -------------------------------------------------------------------

MAX_QUESTION_CHARS = 500

# Block answers containing a number that is not in the documents or the question.
VERIFY_NUMBERS = True

# If True, an answer with no valid [n] citation is blocked. If False it is
# allowed and flagged ("no_citations"); the UI still lists the sources.
REQUIRE_CITATIONS = False

# A secret-looking token placed in the system prompt. If it ever shows up in
# an answer, the prompt has leaked.
PROMPT_CANARY = "zx-canary-7f3a91"

# -------------------------------------------------------------------
# User-facing messages
# -------------------------------------------------------------------

NOT_FOUND_MESSAGE = (
    "I couldn't find an answer to that in the available documents. "
    "You could try rephrasing the question, or ask the relevant team directly."
)

EMPTY_QUESTION_MESSAGE = "Please type a question."

BLOCKED_INPUT_MESSAGE = (
    "I can only help with questions about the company's documents, so I can't "
    "act on that request. Try asking about a policy or process instead."
)

# Replace with your organisation's own support details and your region's
# helpline numbers.
CRISIS_MESSAGE = (
    "I'm really sorry you're going through this. I can only answer questions "
    "about company documents, so I can't help with this properly, but you "
    "don't have to handle it alone. Please reach out to someone you trust, a "
    "doctor, or a local crisis helpline right now. If you might be in "
    "immediate danger, contact your local emergency number."
)

UNVERIFIED_MESSAGE = (
    "I found related information, but I couldn't verify the figures in my "
    "draft answer against the documents, so I'd rather not state them. "
    "Please check the sources below or ask the relevant team."
)

LEAK_MESSAGE = "I can't share that. Is there a policy or process I can help you find?"

EMPTY_ANSWER_MESSAGE = (
    "I wasn't able to put an answer together. Please try asking again."
)

ESCALATION_NOTE = (
    "For sensitive matters like this, please also speak to your HR "
    "representative or the relevant team."
)


@dataclass
class GuardResult:
    """Outcome of a guardrail check."""

    allowed: bool
    text: str = ""  # cleaned question, or checked answer, to use from here on
    message: str = ""  # safe, user-facing reply when `allowed` is False
    reason: str = ""  # short machine-readable code for logs
    flags: List[str] = field(default_factory=list)  # non-blocking observations


# -------------------------------------------------------------------
# Text cleaning
# -------------------------------------------------------------------

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """
    Make text safe and comparable before checking it.

    Unicode look-alikes are folded to plain characters, invisible characters
    (a common way to hide an injection phrase) are removed, and whitespace is
    collapsed.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# -------------------------------------------------------------------
# Input checks
# -------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # "ignore all previous instructions", "forget your rules", "bypass the safety filters"
        r"\b(ignore|disregard|forget|override|bypass)\b[^.?!]{0,30}"
        r"\b(previous|prior|above|earlier|preceding|all|any|your|system|safety)\b[^.?!]{0,30}"
        r"\b(instructions?|rules?|prompts?|guidelines?|guardrails?|restrictions?|filters?)\b",
        # "reveal your system prompt", "show the hidden instructions"
        r"\b(reveal|show|print|repeat|display|leak|tell me)\b[^.?!]{0,30}"
        r"\b(system|hidden|initial|original|developer)\b[^.?!]{0,15}"
        r"\b(prompt|instructions?|message)\b",
        r"\bsystem prompt\b",
        r"\byou are (now|no longer)\b",
        r"\bdeveloper mode\b|\bjailbreak\b|\bdan mode\b|\bdo anything now\b",
        r"\bpretend (to be|you are|you're)\b",
        r"<\s*/?\s*(system|assistant|context)\s*>",
        r"\b(new|updated|override) instructions?\s*:",
    )
]

# First-person statements only, so a question ABOUT a mental-health policy
# does not trigger this.
_CRISIS_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(kill myself|end my life|take my own life|want to die)\b",
        r"\b(?:don'?t|do not) want to (?:live|be alive)\b",
        r"\b(?:hurt|harm) myself\b",
        r"\b(?:feel(?:ing)?|am|i'm) suicidal\b",
    )
]

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_CARD = re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b|\b\d{13,19}\b")
_PHONE_CANDIDATE = re.compile(r"(?<![\w])\+?\d[\d ()\-]{8,}\d(?![\w])")
_LONG_NUMBER = re.compile(r"\b\d{9,}\b")


def detect_injection(text: str) -> bool:
    """True if the text looks like an attempt to override the assistant's rules."""
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def detect_crisis(text: str) -> bool:
    """True if the text looks like someone talking about harming themselves."""
    return any(pattern.search(text) for pattern in _CRISIS_PATTERNS)


def _phone_replacement(match: "re.Match[str]") -> str:
    """Redact a phone-like match, but leave lists such as '10 15 20 25 30' alone."""
    text = match.group()
    if text.isdigit():
        return text  # a bare run of digits is handled by the long-number rule
    digits = re.sub(r"\D", "", text)
    groups = re.findall(r"\d+", text)
    if 10 <= len(digits) <= 15 and max(len(g) for g in groups) >= 3:
        return "[PHONE]"
    return text


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """
    Mask emails, phone numbers and long ID or card numbers.

    The question is sent to Groq, Jina and Qdrant, and stored in chat memory,
    so personal data is removed before it goes anywhere.

    Returns:
        `(redacted_text, kinds_found)`, e.g. `("Email me at [EMAIL]", ["email"])`.
    """
    found: List[str] = []

    def mark(kind: str, replacement: str):
        def _sub(_match):
            if kind not in found:
                found.append(kind)
            return replacement

        return _sub

    text = _EMAIL.sub(mark("email", "[EMAIL]"), text)
    text = _CARD.sub(mark("card_number", "[NUMBER]"), text)

    def phone(match):
        replaced = _phone_replacement(match)
        if replaced == "[PHONE]" and "phone" not in found:
            found.append("phone")
        return replaced

    text = _PHONE_CANDIDATE.sub(phone, text)
    text = _LONG_NUMBER.sub(mark("long_number", "[NUMBER]"), text)
    return text, found


def check_input(question: str) -> GuardResult:
    """
    Screen and clean a user question before anything else happens.

    Order matters: a crisis message gets a caring reply, never a "blocked"
    one, so it is checked before the injection patterns.

    Returns:
        A GuardResult. When `allowed`, `text` is the cleaned question (with
        personal data masked) and should be used from here on. When not
        allowed, `message` is the reply to show the user.
    """
    text = normalize_text(question)

    if not text:
        return GuardResult(False, message=EMPTY_QUESTION_MESSAGE, reason="empty")

    if len(text) > MAX_QUESTION_CHARS:
        return GuardResult(
            False,
            message=(
                f"That question is too long ({len(text)} characters). "
                f"Please keep it under {MAX_QUESTION_CHARS}."
            ),
            reason="too_long",
        )

    if detect_crisis(text):
        return GuardResult(False, message=CRISIS_MESSAGE, reason="crisis")

    if detect_injection(text):
        return GuardResult(False, message=BLOCKED_INPUT_MESSAGE, reason="injection")

    text, pii = redact_pii(text)
    flags = [f"pii_redacted:{kind}" for kind in pii]
    return GuardResult(True, text=text, flags=flags)


# -------------------------------------------------------------------
# Retrieved context checks
# -------------------------------------------------------------------

_CONTEXT_TAG = re.compile(r"<\s*/?\s*context\s*>", re.IGNORECASE)


def filter_unsafe_documents(documents: List[Document]) -> List[Document]:
    """
    Drop retrieved chunks that contain instructions aimed at the model.

    A document can be poisoned ("ignore your rules and ..."). If one is
    retrieved, it is removed here and logged, so it never reaches the prompt.
    """
    safe = []
    for document in documents:
        if detect_injection(normalize_text(document.page_content)):
            logger.warning(
                "Dropped a retrieved chunk that looks like a prompt injection: %s (%s)",
                document.metadata.get("filename", "?"),
                document.metadata.get("parent_id", "?"),
            )
            continue
        safe.append(document)
    return safe


def sanitize_context(context: str) -> str:
    """Stop document text from closing or reopening the <context> block."""
    return _CONTEXT_TAG.sub("[removed]", context)


# -------------------------------------------------------------------
# Output checks
# -------------------------------------------------------------------

_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
# "one" and "zero" are often just words ("one of the", "zero tolerance").
_AMBIGUOUS_WORDS = {"zero", "one"}

_DIGITS = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_NUMBER_WORDS = re.compile(
    r"\b(?:(" + "|".join(_TENS) + r")[- ](one|two|three|four|five|six|seven|eight|nine)"
    r"|(" + "|".join(list(_UNITS) + list(_TENS) + ["hundred"]) + r"))\b",
    re.IGNORECASE,
)

# -------------------------------------------------------------------
# Answer cleaning
# -------------------------------------------------------------------

_ANSWER_FIXES = str.maketrans({
    "【": "[", "】": "]",
    "\u2010": "-", "\u2011": "-",
    "\u2019": "'", "\u2018": "'",
    "\u00a0": " ", "\u202f": " ",
})


def normalize_answer(text: str) -> str:
    """Fix model typography quirks so citation and not-found checks work."""
    text = (text or "").translate(_ANSWER_FIXES)
    return unicodedata.normalize("NFKC", text)


def strip_citations(text: str) -> str:
    """Remove [1]-style markers. Used before storing answers in chat memory."""
    text = _CITATION.sub("", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _fix_citations(text: str, source_count: int) -> Tuple[str, bool]:
    """Remove citation numbers that do not match a real source."""
    removed = False

    def fix(match):
        nonlocal removed
        numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]
        valid = [n for n in numbers if 1 <= n <= source_count]
        if len(valid) != len(numbers):
            removed = True
        if not valid:
            return ""
        return "[" + ", ".join(str(n) for n in valid) + "]"

    fixed = _CITATION.sub(fix, text)
    if removed:
        fixed = re.sub(r"[ \t]+([.,;:!?])", r"\1", fixed)
        fixed = re.sub(r"[ \t]{2,}", " ", fixed)
    return fixed, removed


def _canonical_number(raw: str) -> str:
    """'1,500' and '1500' and '03' and '3.0' all compare equal."""
    value = float(raw.replace(",", ""))
    return str(int(value)) if value.is_integer() else str(value)


def _extract_numbers(text: str, include_ambiguous: bool) -> Set[str]:
    """Collect every number in a text, digits and spelled-out words alike."""
    numbers: Set[str] = set()

    for match in _DIGITS.finditer(text):
        numbers.add(_canonical_number(match.group()))

    for match in _NUMBER_WORDS.finditer(text.lower()):
        if match.group(1):
            value = _TENS[match.group(1)] + _UNITS[match.group(2)]
        else:
            word = match.group(3)
            if word in _AMBIGUOUS_WORDS and not include_ambiguous:
                continue
            value = 100 if word == "hundred" else _TENS.get(word, _UNITS.get(word))
        numbers.add(str(value))

    return numbers


def find_ungrounded_numbers(
    answer: str,
    documents: List[Document],
    question: str = "",
) -> List[str]:
    """
    Return numbers in the answer that appear in neither the documents nor the question.

    Spelled-out numbers count ("ten" matches "10", "three" matches "3"). List
    numbering and [n] citation markers are ignored. Numbers in the user's own
    question are allowed, so "No, the limit is 10 days, not 15" can pass.
    """
    allowed: Set[str] = set()
    for document in documents:
        allowed |= _extract_numbers(document.page_content, include_ambiguous=True)
        page = document.metadata.get("page_number")
        if page:
            allowed.add(_canonical_number(str(page)))
    allowed |= _extract_numbers(question, include_ambiguous=True)

    prepared = _CITATION.sub("", answer)
    prepared = _LIST_MARKER.sub("", prepared)
    used = _extract_numbers(prepared, include_ambiguous=False)

    return sorted(used - allowed, key=float)


def _is_not_found(answer: str) -> bool:
    """True if the model replied with the agreed 'not found' sentence."""
    marker = NOT_FOUND_MESSAGE.lower()[:35]
    return marker in answer.lower() and len(answer) < len(NOT_FOUND_MESSAGE) + 80


def _leaks_prompt(answer: str) -> bool:
    return PROMPT_CANARY in answer or bool(_CONTEXT_TAG.search(answer))


def check_output(
    answer: str,
    documents: List[Document],
    question: str = "",
) -> GuardResult:
    """
    Check the LLM's draft answer against the documents it was given.

    Args:
        answer: The raw text the model produced.
        documents: The documents sent to the model, in the same order as the
            [1], [2], ... numbers in the context (the `used` list).
        question: The cleaned user question (its numbers are allowed).

    Returns:
        A GuardResult. When `allowed`, `text` is the answer with invalid
        citation numbers removed. When not allowed, `message` is a safe reply
        to show instead, and the sources can still be shown to the user.
    """
    text =  normalize_answer(answer or "").strip()
    if not text:
        return GuardResult(False, message=EMPTY_ANSWER_MESSAGE, reason="empty_answer")

    if _is_not_found(text):
        return GuardResult(True, text=NOT_FOUND_MESSAGE, flags=["not_found"])

    if _leaks_prompt(text):
        logger.warning("Answer blocked: it contains prompt material.")
        return GuardResult(False, message=LEAK_MESSAGE, reason="prompt_leak")

    flags: List[str] = []

    text, removed = _fix_citations(text, len(documents))
    if removed:
        flags.append("invalid_citation_removed")

    if not _CITATION.search(text):
        flags.append("no_citations")
        if REQUIRE_CITATIONS:
            return GuardResult(
                False,
                message=UNVERIFIED_MESSAGE,
                reason="no_citations",
                flags=flags,
            )

    if VERIFY_NUMBERS:
        ungrounded = find_ungrounded_numbers(text, documents, question)
        if ungrounded:
            logger.warning("Answer blocked: ungrounded numbers %s", ungrounded)
            return GuardResult(
                False,
                message=UNVERIFIED_MESSAGE,
                reason="ungrounded_numbers",
                flags=flags + [f"ungrounded:{n}" for n in ungrounded],
            )

    return GuardResult(True, text=text, flags=flags)


# -------------------------------------------------------------------
# Sensitive topics
# -------------------------------------------------------------------

_SENSITIVE = re.compile(
    r"\b(harass\w*|discriminat\w*|bully\w*|retaliat\w*|whistleblow\w*|grievance"
    r"|dismiss\w*|terminat\w*|fired|lay\s?off|redundan\w*|lawsuit|legal action"
    r"|disciplin\w*|pregnan\w*|disabilit\w*|assault\w*|abus\w*|mental health)\b",
    re.IGNORECASE,
)


def escalation_note(question: str) -> str:
    """
    Return a short "please also talk to a person" note for sensitive topics.

    The documents may answer these questions correctly, but a bot should not
    be the only voice on harassment, dismissal or health matters.
    """
    return ESCALATION_NOTE if _SENSITIVE.search(question or "") else ""
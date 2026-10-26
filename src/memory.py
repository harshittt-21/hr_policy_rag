"""
Conversation memory, so follow-up questions make sense.

Without memory, every question starts from zero:

    User: What is TechNova's refund policy?
    User: Does that policy apply to enterprise customers?   <- "that" is lost

We use LangChain's `InMemoryChatMessageHistory`, which is simply a list of
chat messages held in memory for the length of the session. No database, no
files - closing the app clears everything, which is exactly what we want for
a learning project.

The history is used twice in this project:
1. When rewriting the question, so "that policy" becomes a searchable phrase.
2. In the answer prompt, so the model's wording follows the conversation.
"""

from typing import List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import config

# Previous answers are shortened in the rewrite prompt: the rewriter only
# needs the topic, not the full text.
MAX_ANSWER_CHARS_IN_HISTORY = 300


def create_memory() -> InMemoryChatMessageHistory:
    """
    Create an empty conversation history.

    Streamlit stores the returned object in `st.session_state`, which keeps it
    alive between button clicks but separate for every browser session.

    Returns:
        A fresh, empty LangChain chat message history.
    """
    return InMemoryChatMessageHistory()


def add_exchange(
    memory: InMemoryChatMessageHistory,
    question: str,
    answer: str,
) -> None:
    """
    Record one question-and-answer pair in the conversation history.

    Call this after the LLM has replied, so the next question can refer back
    to what was just discussed. Adding both together keeps the history in
    clean question/answer pairs even if answering fails part-way.

    Args:
        memory: The conversation history object.
        question: What the user asked.
        answer: What the assistant replied.
    """
    memory.add_user_message(question)
    memory.add_ai_message(answer)


def get_recent_messages(memory: InMemoryChatMessageHistory) -> List[BaseMessage]:
    """
    Return only the last few messages from the conversation.

    Sending an hour-long conversation to the model on every question would be
    slow and expensive, and old messages rarely help. A sliding window of the
    most recent exchanges gives nearly all the benefit at a fraction of the
    cost.

    Args:
        memory: The conversation history object.

    Returns:
        The most recent messages, oldest first. MEMORY_WINDOW counts
        question/answer pairs, so a window of 4 keeps 8 messages. A window of
        0 turns memory off and returns nothing.
    """
    window = config.MEMORY_WINDOW
    if window <= 0:
        # Without this guard, messages[-0:] would return the WHOLE history.
        return []

    messages = memory.messages[-(window * 2):]

    # If the cut landed mid-exchange, do not start with an orphaned answer.
    if messages and isinstance(messages[0], AIMessage):
        messages = messages[1:]

    return messages


def _shorten(text: str, limit: int) -> str:
    """Cut text to roughly `limit` characters at a word boundary."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def format_history(memory: InMemoryChatMessageHistory) -> str:
    """
    Turn the recent messages into plain text for the query-rewriting prompt.

    The rewriting prompt takes a single `{history}` string, so we flatten the
    message objects into labelled lines.

    Args:
        memory: The conversation history object.

    Returns:
        A string such as:
            User: What is the refund policy?
            Assistant: Refunds are available within 30 days...
        Returns "" when the conversation has not started yet.
    """
    lines = []
    for message in get_recent_messages(memory):
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(
                f"Assistant: {_shorten(message.content, MAX_ANSWER_CHARS_IN_HISTORY)}"
            )
    return "\n".join(lines)


def clear_memory(memory: InMemoryChatMessageHistory) -> None:
    """
    Forget the whole conversation.

    This is what the "Clear conversation" button in the sidebar calls when a
    learner wants to start a fresh topic.

    Args:
        memory: The conversation history object to empty.
    """
    memory.clear()
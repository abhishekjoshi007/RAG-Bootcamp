from __future__ import annotations


def build_query(unit: dict, include_topics: bool = True) -> str:
    pieces = [unit["title"], unit.get("description", "")]
    topics = unit.get("current_topics", [])
    if include_topics and topics:
        pieces.append("Topics: " + ", ".join(topics))
    return ". ".join(piece.strip() for piece in pieces if piece.strip())


def build_query_from_learner(learner_query) -> str:
    pieces = [
        learner_query.query_text,
        learner_query.goal,
        " ".join(learner_query.completed_skills),
        " ".join(learner_query.curriculum_unit_ids),
    ]
    text = ". ".join(piece.strip() for piece in pieces if piece and piece.strip())
    return text or learner_query.program


def build_hyde_prompt(unit: dict) -> str:
    return (
        "Write a brief hypothetical source paragraph about this CS curriculum topic. "
        "Use concrete technical language and avoid recommendations.\n\n"
        f"Topic: {unit['title']}\n"
        f"Description: {unit.get('description', '')}"
    )

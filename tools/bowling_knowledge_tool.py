#!/usr/bin/env python3
"""Native Hermes tool for Bowling Knowledge's grounded, cited answer
pipeline.

Thin self-registering wrapper (this repo's standard tools/ convention --
see close_terminal_tool.py) around
skills/bowling-knowledge/scripts/hermes_tool.py's
``answer_bowling_knowledge_question()``. Holds no retrieval, generation,
or validation logic of its own -- see that module (and, beneath it,
answer_context.py / model_client.py / answer_generator.py, all
unchanged) for the actual pipeline.

Always answers from the one known production Bowling Knowledge database,
opened read-only. Never accepts a database path, source-library path,
model endpoint, or raw SQL from the caller -- those are exactly the
controls this tool is deliberately not allowed to expose.
"""

import importlib.util
import sys
from pathlib import Path

from tools.registry import registry, tool_error, tool_result

_ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills" / "bowling-knowledge" / "scripts" / "hermes_tool.py"
)

_adapter = None


def _get_adapter():
    """Lazily load the adapter module on first use (not at import time),
    so importing this tool file never has a side effect of touching the
    filesystem beyond Python's normal module machinery."""
    global _adapter
    if _adapter is None:
        spec = importlib.util.spec_from_file_location(
            "bowling_knowledge_hermes_tool_adapter", _ADAPTER_PATH,
        )
        if spec is None or spec.loader is None:  # pragma: no cover - environment failure
            raise RuntimeError("Bowling Knowledge adapter module could not be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _adapter = module
    return _adapter


def bowling_knowledge_answer_tool(question: str, top_k=None, corpus_only: bool = True) -> str:
    adapter = _get_adapter()
    try:
        result = adapter.answer_bowling_knowledge_question(
            question, top_k=top_k, corpus_only=corpus_only,
        )
    except adapter.BowlingKnowledgeToolError as exc:
        return tool_error(str(exc))
    except Exception:  # noqa: BLE001 - never let an internal detail leak to the model/user
        return tool_error("Bowling Knowledge could not answer that question right now.")
    return tool_result(result)


BOWLING_KNOWLEDGE_ANSWER_SCHEMA = {
    "name": "bowling_knowledge_answer",
    "description": (
        "Answer a bowling-technique question strictly from the curated, "
        "provenance-tracked Bowling Knowledge corpus (expert sources "
        "including Mo Pinel, Joe Slowinski, Mark Baker, and Kegel). Every "
        "claim in the answer is grounded in and cited to a specific "
        "sourced claim -- it never blends in general bowling knowledge "
        "unless corpus_only is explicitly set to false. If the corpus "
        "doesn't contain enough relevant sourced evidence, it says so "
        "instead of guessing. Name a specific expert in the question (e.g. "
        "'What does Mo Pinel say about RG versus differential?') to get "
        "that expert's answer only, or ask a comparison question "
        "('Compare X and Y on ...') to get each named expert's evidence "
        "kept separate, never blended."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The bowling-technique question to answer from the corpus.",
            },
            "top_k": {
                "type": "integer",
                "description": (
                    "How many evidence claims to consider, bounded 1-10. "
                    "Omit to use the corpus's own evidence-planning default."
                ),
                "minimum": 1,
                "maximum": 10,
            },
            "corpus_only": {
                "type": "boolean",
                "description": (
                    "Default true: answer strictly from the corpus, never "
                    "blending in general bowling knowledge not present in it."
                ),
                "default": True,
            },
        },
        "required": ["question"],
    },
}


registry.register(
    name="bowling_knowledge_answer",
    toolset="bowling_knowledge",
    schema=BOWLING_KNOWLEDGE_ANSWER_SCHEMA,
    handler=lambda args, **kw: bowling_knowledge_answer_tool(
        question=args.get("question", ""),
        top_k=args.get("top_k"),
        corpus_only=args.get("corpus_only", True),
    ),
    emoji="🎳",
)

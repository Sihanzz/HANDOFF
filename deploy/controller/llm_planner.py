"""LLM-based task decomposer — a genuine (if partial) stand-in for the
"molmo" agentic planner's task-decomposition half (paper Fig. 4, ~0.001 Hz).

This is NOT the paper's VLM visual-grounding half (SAM2 + FoundationStereoPose
object localization) — that pipeline was never included in the public repo
and is out of scope here (see prompt_node.py's module docstring). This module
only replaces the *language* side: turning a free-form instruction into an
ordered sequence of primitive robot actions, using a real LLM call instead of
regex keyword matching.

Requires ANTHROPIC_API_KEY (loaded from deploy/controller/.env, which is
git-ignored — never commit it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

_MODEL = "claude-haiku-4-5-20251001"

_VALID_ACTIONS = (
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "squat",
    "stand",
    "wave",
    "stop",
)

_SYSTEM_PROMPT = f"""You are the task-decomposition module of a humanoid robot controller.

The robot only understands these primitive actions, in this exact vocabulary:
{", ".join(_VALID_ACTIONS)}

- forward / backward: walk in place for a few seconds, then stop.
- turn_left / turn_right: rotate in place for a few seconds, then stop.
- strafe_left / strafe_right: side-step for a few seconds, then stop.
- squat: lower to a crouched height and hold.
- stand: return to nominal standing height and hold.
- wave: raise and wave the right hand briefly, then lower it.
- stop: immediately zero all velocities and cancel pending actions.

Given a free-form natural-language instruction (English or Chinese), decompose
it into an ordered list of these primitive actions that best accomplishes it.
Repeat an action if the instruction implies doing it multiple times or for
longer (e.g. "turn all the way around" -> turn_left, turn_left, turn_left,
turn_left as a rough 360). If the instruction doesn't map to anything the
robot can do (e.g. "pick up the box" — no grasping primitive exists), still
return your best-effort partial plan using whatever primitives are relevant,
and say so in the "note" field.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"steps": [{{"action": "<one of the primitives above>", "desc": "<short human-readable label, e.g. 'Walk forward'>"}}, ...], "note": "<optional one-sentence caveat, or empty string>"}}
"""


class LLMPlannerError(RuntimeError):
    """Raised when the LLM call fails or returns something unusable."""


def decompose(instruction: str) -> tuple[list[tuple[str, str]], str]:
    """Turn free-form text into [(action, desc), ...] plus an optional note.

    Raises LLMPlannerError on any failure (missing key, network error, bad
    JSON, unknown action) so the caller can fall back to the regex parser.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMPlannerError("ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError as e:
        raise LLMPlannerError(f"anthropic package not installed: {e}") from e

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": instruction}],
        )
        raw = resp.content[0].text.strip()
    except Exception as e:  # noqa: BLE001 - any SDK/network failure -> fallback
        raise LLMPlannerError(f"Anthropic API call failed: {e}") from e

    # Model sometimes wraps JSON in a ```json fence despite instructions.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMPlannerError(f"Non-JSON response: {raw[:200]!r}") from e

    steps_raw = parsed.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise LLMPlannerError(f"No steps in response: {parsed!r}")

    steps: list[tuple[str, str]] = []
    for item in steps_raw:
        action = item.get("action")
        desc = item.get("desc", action)
        if action not in _VALID_ACTIONS:
            raise LLMPlannerError(f"Unknown action from LLM: {action!r}")
        steps.append((action, str(desc)))

    note = str(parsed.get("note", "") or "")
    return steps, note

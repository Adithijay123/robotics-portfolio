"""
LLM task planner: natural language instruction + scene to validated action plan.
"""

import json
import os
import re

from google import genai
from google.genai import types

from .actions import ArmAction, InvalidActionError, VALID_ACTION_TYPES
from .scene import Scene


SYSTEM_PROMPT = f"""You are a task planner for a 3-DOF robotic arm.

You must convert a natural-language instruction into a JSON list of actions,
using ONLY the following action types: {sorted(VALID_ACTION_TYPES)}.

Action schemas:
- {{"type": "move_to", "x": <float>, "y": <float>, "z": <float>}}
- {{"type": "open_gripper"}}
- {{"type": "close_gripper"}}
- {{"type": "wait", "seconds": <float>}}
- {{"type": "go_home"}}

Rules:
- To pick up an object: move_to just above it, move_to its exact height, close_gripper, move_to a safe height.
- To place an object: move_to above the target location, move_to down to place height, open_gripper, move_to a safe height.
- Always use the object coordinates given in the scene description. Do not invent coordinates.
- "left of" means lower y value MINUS 0.10; "right of" means y value PLUS 0.10 (arm's y-axis convention).
- Output ONLY a JSON array of action objects. No prose, no markdown fences, no explanation.
"""


class PlanningError(Exception):
    pass


class LLMArmPlanner:
    def __init__(self, model: str = "gemini-3.6-flash", api_key: str = None):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise PlanningError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment "
                "variable, or pass api_key= explicitly."
            )
        self.client = genai.Client(api_key=resolved_key)
        self.model = model

    def plan(self, instruction: str, scene: Scene) -> list[ArmAction]:
        user_prompt = (
            f"Scene:\n{scene.to_prompt_context()}\n\n"
            f"Instruction: {instruction}\n\n"
            f"Output the JSON action list now."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )
        raw_text = (response.text or "").strip()

        return self._parse_and_validate(raw_text)

    @staticmethod
    def _parse_and_validate(raw_text: str) -> list[ArmAction]:
        # Strip accidental markdown fences just in case
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise PlanningError(f"LLM did not return valid JSON: {e}\nRaw output: {raw_text}")

        if not isinstance(data, list):
            raise PlanningError(f"Expected a JSON array of actions, got: {type(data)}")

        actions = []
        for i, item in enumerate(data):
            try:
                actions.append(ArmAction.from_dict(item))
            except InvalidActionError as e:
                raise PlanningError(f"Invalid action at index {i}: {e}")

        if not actions:
            raise PlanningError("LLM returned an empty action list")

        return actions

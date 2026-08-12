"""
Quick smoke test: runs a few fixed instructions through the planner
and mock executor, without needing interactive input.

Usage:
    export GEMINI_API_KEY=your_key_here   # PowerShell: $env:GEMINI_API_KEY = "your_key_here"
    python test_planner.py
"""

from llm_arm_planner.scene import get_demo_scene
from llm_arm_planner.planner import LLMArmPlanner, PlanningError
from llm_arm_planner.arm_interface import MockArmExecutor


TEST_INSTRUCTIONS = [
    "pick up the red block and place it left of the blue block",
    "pick up the green cup and move it to the home position",
    "close the gripper, wait 1 second, then go home",
]


def main():
    scene = get_demo_scene()
    planner = LLMArmPlanner()

    for instruction in TEST_INSTRUCTIONS:
        print(f"\n{'=' * 60}")
        print(f"Instruction: {instruction}")
        print("=" * 60)

        try:
            actions = planner.plan(instruction, scene)
        except PlanningError as e:
            print(f"FAILED to plan: {e}")
            continue

        for a in actions:
            print(f"  - {a}")

        executor = MockArmExecutor(step_delay=0.05)
        executor.execute_plan(actions)


if __name__ == "__main__":
    main()

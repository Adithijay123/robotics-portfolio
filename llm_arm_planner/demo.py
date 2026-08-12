"""
Demo: type a natural language instruction.
"""

from llm_arm_planner.scene import get_demo_scene
from llm_arm_planner.planner import LLMArmPlanner, PlanningError
from llm_arm_planner.arm_interface import MockArmExecutor


def main():
    scene = get_demo_scene()
    planner = LLMArmPlanner()
    executor = MockArmExecutor()

    print("LLM Arm Planner Demo ")
    print(scene.to_prompt_context())
    print()
    print("Type an instruction (e.g. 'pick up the red block and place it left of the blue block')")
    print("Type 'quit' to exit.\n")

    while True:
        instruction = input("> ").strip()
        if instruction.lower() in ("quit", "exit"):
            break
        if not instruction:
            continue

        try:
            actions = planner.plan(instruction, scene)
        except PlanningError as e:
            print(f"Planning failed: {e}\n")
            continue

        print(f"\nPlan ({len(actions)} steps):")
        for a in actions:
            print(f"  - {a}")
        print()

        executor.execute_plan(actions)
        print()


if __name__ == "__main__":
    main()

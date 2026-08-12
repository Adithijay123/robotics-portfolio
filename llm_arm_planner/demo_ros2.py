"""
it is the same as demo.py, but executes plans against the real ROS2 action server
(arm_sim_node) instead of the in-process MockArmExecutor.

Prerequisites:
    1. ROS2 installed and sourced (e.g. `source /opt/ros/humble/setup.bash`)
    2. The ros2_arm_sim workspace built:
           cd ros2_arm_sim
           colcon build
           source install/setup.bash
    3. In a separate terminal, the simulated arm running:
           ros2 run arm_sim arm_sim_node
    4. This project's dependencies installed:
           pip install google-genai
    5. GEMINI_API_KEY set in the environment

Then, in another terminal (same ROS2 workspace sourced):
    python demo_ros2.py
"""

import rclpy

from llm_arm_planner.scene import get_demo_scene
from llm_arm_planner.planner import LLMArmPlanner, PlanningError
from llm_arm_planner.arm_interface import Ros2ArmExecutor


def main():
    rclpy.init()
    node = rclpy.create_node('llm_planner_client')

    try:
        executor = Ros2ArmExecutor(node)
    except RuntimeError as e:
        print(f"Could not connect to arm_sim_node: {e}")
        rclpy.shutdown()
        return

    scene = get_demo_scene()
    planner = LLMArmPlanner()

    print(" LLM Arm Planner (ROS2 simulation) ")
    print(scene.to_prompt_context())
    print()
    print("Connected to arm_sim_node. Type an instruction, or 'quit' to exit.\n")

    try:
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
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

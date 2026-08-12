"""
Arm executor interface.

MockArmExecutor simulates execution in pure Python, no ROS2 required 
for quickly testing the LLM planning layer on its own.

Ros2ArmExecutor talks to the simulated arm ROS2 action server (arm_sim_node,
in the sibling ros2_arm_sim workspace) over real ROS2 actions, MoveTo and
Gripper. Same code path you'd use for real hardware later, only the node on
the other end of the action would change.
"""

import time
from abc import ABC, abstractmethod

from .actions import ArmAction


class ArmExecutor(ABC):
    @abstractmethod
    def execute(self, action: ArmAction) -> bool:
        """Execute a single action. Return True on success."""
        raise NotImplementedError

    def execute_plan(self, actions: list[ArmAction]) -> bool:
        for i, action in enumerate(actions):
            print(f"[{i + 1}/{len(actions)}] Executing: {action}")
            ok = self.execute(action)
            if not ok:
                print(f"  -> FAILED at step {i + 1}: {action}")
                return False
        print("Plan executed successfully.")
        return True


class MockArmExecutor(ArmExecutor):

    def __init__(self, step_delay: float = 0.3):
        self.step_delay = step_delay
        self.current_position = (0.0, 0.0, 0.2)
        self.gripper_state = "open"

    def execute(self, action: ArmAction) -> bool:
        time.sleep(self.step_delay)
        if action.type == "move_to":
            self.current_position = (action.x, action.y, action.z)
        elif action.type == "open_gripper":
            self.gripper_state = "open"
        elif action.type == "close_gripper":
            self.gripper_state = "closed"
        elif action.type == "wait":
            time.sleep(action.seconds)
        elif action.type == "go_home":
            self.current_position = (0.0, 0.0, 0.2)
        return True


class Ros2ArmExecutor(ArmExecutor):
    """
    Talks to the simulated arm over real ROS2 actions.

    Requires arm_sim_node to be running separately:
        ros2 run arm_sim arm_sim_node
    """

    MOVE_ACTION = 'arm_sim/move_to'
    GRIPPER_ACTION = 'arm_sim/gripper'
    ACTION_SERVER_TIMEOUT_SEC = 10.0

    def __init__(self, node):
        import rclpy
        from rclpy.action import ActionClient
        from arm_interfaces.action import MoveTo, Gripper

        self._rclpy = rclpy
        self.node = node
        self._move_client = ActionClient(node, MoveTo, self.MOVE_ACTION)
        self._gripper_client = ActionClient(node, Gripper, self.GRIPPER_ACTION)
        self._MoveTo = MoveTo
        self._Gripper = Gripper

        for client, name in (
            (self._move_client, self.MOVE_ACTION),
            (self._gripper_client, self.GRIPPER_ACTION),
        ):
            if not client.wait_for_server(timeout_sec=self.ACTION_SERVER_TIMEOUT_SEC):
                raise RuntimeError(
                    f"Action server '{name}' not available after "
                    f"{self.ACTION_SERVER_TIMEOUT_SEC}s. Is arm_sim_node running? "
                    f"(ros2 run arm_sim arm_sim_node)"
                )

        self.home_position = (0.0, 0.0, 0.20)

    def _send_and_wait(self, client, goal_msg, action_name: str):
        future = client.send_goal_async(goal_msg)
        self._rclpy.spin_until_future_complete(self.node, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.node.get_logger().error(f'{action_name} goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        self._rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result().result

        if not result.success:
            self.node.get_logger().error(f'{action_name} failed: {result.message}')
        return result.success

    def execute(self, action: ArmAction) -> bool:
        if action.type == 'move_to':
            goal = self._MoveTo.Goal(x=action.x, y=action.y, z=action.z)
            return self._send_and_wait(self._move_client, goal, 'move_to')

        if action.type == 'go_home':
            x, y, z = self.home_position
            goal = self._MoveTo.Goal(x=x, y=y, z=z)
            return self._send_and_wait(self._move_client, goal, 'go_home')

        if action.type == 'open_gripper':
            goal = self._Gripper.Goal(command='open')
            return self._send_and_wait(self._gripper_client, goal, 'open_gripper')

        if action.type == 'close_gripper':
            goal = self._Gripper.Goal(command='close')
            return self._send_and_wait(self._gripper_client, goal, 'close_gripper')

        if action.type == 'wait':
            time.sleep(action.seconds)
            return True

        self.node.get_logger().error(f'Unknown action type: {action.type}')
        return False

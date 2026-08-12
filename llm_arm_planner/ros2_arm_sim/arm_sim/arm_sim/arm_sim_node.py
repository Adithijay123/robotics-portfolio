"""
Simulated 3-DOF arm as a ROS2 action server.

Exposes two actions:
  - /arm_sim/move_to   (arm_interfaces/action/MoveTo)
  - /arm_sim/gripper   (arm_interfaces/action/Gripper)

No hardware required. Position is simulated by linear interpolation over a
fixed duration, with feedback published along the way, and the final pose
published as a sensor_msgs/JointState on /joint_states so you can watch it
move in rviz2 if you add a robot model, or just log it directly.

This mirrors exactly what a real SOLARIS-style ROS2 action server would look
like — swap the body of `_simulate_move` for real motor driver calls later
and the LLM planner / Ros2ArmExecutor on the Python side don't change at all.
"""

import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import JointState

from arm_interfaces.action import MoveTo, Gripper


MOVE_DURATION_SEC = 1.0
FEEDBACK_STEPS = 10


class ArmSimNode(Node):
    def __init__(self):
        super().__init__('arm_sim_node')

        self.current_position = [0.0, 0.0, 0.20]  # matches Scene.home_position
        self.gripper_state = 'open'

        cb_group = ReentrantCallbackGroup()

        self._move_server = ActionServer(
            self,
            MoveTo,
            'arm_sim/move_to',
            execute_callback=self._execute_move_to,
            callback_group=cb_group,
        )
        self._gripper_server = ActionServer(
            self,
            Gripper,
            'arm_sim/gripper',
            execute_callback=self._execute_gripper,
            callback_group=cb_group,
        )

        self._joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self._publish_joint_state()

        self.get_logger().info('arm_sim_node ready: /arm_sim/move_to, /arm_sim/gripper')

    def _publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # Reporting the simulated end-effector pose as three "joints" x, y, z
        # for easy inspection with `ros2 topic echo /joint_states`.
        msg.name = ['x', 'y', 'z']
        msg.position = list(self.current_position)
        self._joint_state_pub.publish(msg)

    def _execute_move_to(self, goal_handle):
        target = (goal_handle.request.x, goal_handle.request.y, goal_handle.request.z)
        start = tuple(self.current_position)
        self.get_logger().info(f'move_to: {start} -> {target}')

        for step in range(1, FEEDBACK_STEPS + 1):
            t = step / FEEDBACK_STEPS
            interp = [start[i] + (target[i] - start[i]) * t for i in range(3)]
            self.current_position = interp
            self._publish_joint_state()

            feedback = MoveTo.Feedback()
            feedback.current_x, feedback.current_y, feedback.current_z = interp
            feedback.progress = t
            goal_handle.publish_feedback(feedback)

            time.sleep(MOVE_DURATION_SEC / FEEDBACK_STEPS)

        goal_handle.succeed()
        result = MoveTo.Result()
        result.success = True
        result.message = f'Reached ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})'
        return result

    def _execute_gripper(self, goal_handle):
        command = goal_handle.request.command.lower().strip()
        self.get_logger().info(f'gripper: {command}')

        result = Gripper.Result()
        if command not in ('open', 'close'):
            goal_handle.abort()
            result.success = False
            result.message = f"Unknown gripper command '{command}', expected 'open' or 'close'"
            return result

        time.sleep(0.3)  # simulated actuation time
        self.gripper_state = 'open' if command == 'open' else 'closed'

        feedback = Gripper.Feedback()
        feedback.state = self.gripper_state
        goal_handle.publish_feedback(feedback)

        goal_handle.succeed()
        result.success = True
        result.message = f'Gripper {self.gripper_state}'
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ArmSimNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

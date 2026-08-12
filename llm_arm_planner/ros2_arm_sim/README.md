# ros2_arm_sim

A ROS2 simulation of a 3-DOF arm, exposing two actions that the LLM planner
project drives directly:

- `arm_sim/move_to` (`arm_interfaces/action/MoveTo`) — moves to an (x, y, z)
  position over ~1 second, publishing feedback and `/joint_states` along the way.
- `arm_sim/gripper` (`arm_interfaces/action/Gripper`) — opens/closes the gripper.

No real hardware involved — `arm_sim_node` simulates motion by linear
interpolation. This is a drop-in stand-in for what a real SOLARIS-style
action server would look like; only the *inside* of the callback would
change to talk to real motors.

## Prerequisites

ROS2 (Humble or newer) installed. On Windows, the standard route is
**WSL2 + Ubuntu 22.04**, then install ROS2 inside that Ubuntu environment —
ROS2 does not run natively on Windows in any well-supported way.

If you don't have ROS2 yet:
1. Install WSL2: `wsl --install` in an admin PowerShell, choose Ubuntu 22.04
2. Inside the Ubuntu terminal, follow the official ROS2 Humble install docs:
   https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
3. `sudo apt install python3-colcon-common-extensions`

## Build

```bash
cd ros2_arm_sim
colcon build
source install/setup.bash
```

You'll need to re-run `source install/setup.bash` in every new terminal you
use for this project (or add it to your `~/.bashrc`).

## Run the simulated arm

```bash
ros2 run arm_sim arm_sim_node
```

Leave this running. You should see:
```
[INFO] [arm_sim_node]: arm_sim_node ready: /arm_sim/move_to, /arm_sim/gripper
```

## Sanity-check it manually (optional)

In another sourced terminal:
```bash
ros2 action send_goal /arm_sim/move_to arm_interfaces/action/MoveTo "{x: 0.2, y: 0.1, z: 0.05}" --feedback
ros2 action send_goal /arm_sim/gripper arm_interfaces/action/Gripper "{command: 'close'}"
ros2 topic echo /joint_states
```

## Drive it from the LLM planner

From the `llm_arm_planner` project (in a terminal with this workspace
sourced, and the Python venv/deps for that project active):

```bash
python demo_ros2.py
```

This is identical to `demo.py` except it sends every planned action as a
real ROS2 action goal to `arm_sim_node` instead of simulating in-process.

## Workspace layout

```
ros2_arm_sim/
├── arm_interfaces/     # action message definitions (MoveTo, Gripper)
│   ├── action/
│   ├── CMakeLists.txt
│   └── package.xml
└── arm_sim/             # the simulated arm action server node
    ├── arm_sim/
    │   └── arm_sim_node.py
    ├── setup.py
    └── package.xml
```

## Going further

- **Visualise in rviz2**: add a URDF for a simple 3-link arm and remap
  `/joint_states` to drive it — `arm_sim_node` already publishes the right
  topic and message type.
- **Real hardware**: once you're ready to connect real motors, only the
  body of `_execute_move_to` / `_execute_gripper` in `arm_sim_node.py`
  changes — everything upstream (the LLM planner, the action interfaces,
  `Ros2ArmExecutor`) stays exactly the same.

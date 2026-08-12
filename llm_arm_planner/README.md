# LLM Arm Planner

A natural-language task planner for a robotic arm, using an LLM to translate
plain-English instructions into a validated sequence of low-level actions.

Architecture:

```
instruction (text) + scene (object positions)
         |
         v
   LLMArmPlanner  <- Gemini API, constrained to a fixed JSON action schema
         |
         v
   list[ArmAction]  <- validated, parsed, guaranteed-safe action objects
         |
         v
   ArmExecutor  <- MockArmExecutor (simulated) or Ros2ArmExecutor (real SOLARIS)
```

## Setup

```bash
pip install google-genai
```

Get a free API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then set it:

```bash
# macOS / Linux
export GEMINI_API_KEY=your key here

# Windows PowerShell
$env:GEMINI_API_KEY = "your key here"

# Windows cmd
set GEMINI_API_KEY=your key here
```

## Run the interactive demo (no hardware needed)

```bash
python demo.py
```

Try instructions like:
- "pick up the red block and place it left of the blue block"
- "move the green cup to the home position"

## Run the non-interactive smoke test

```bash
python test_planner.py
```

## Run against the ROS2 simulation (instead of the mock executor)

The `ros2_arm_sim/` workspace (sibling to this folder) contains a real ROS2
action server simulating the arm. See `ros2_arm_sim/README.md` for setup
(requires ROS2, typically via WSL2 on Windows). Once it's built and running:

```bash
ros2 run arm_sim arm_sim_node    # in one terminal
python demo_ros2.py              # in another
```

`demo_ros2.py` is identical to `demo.py`, but sends each planned action as a
real ROS2 action goal (`MoveTo` / `Gripper`) to `arm_sim_node` instead of
simulating in-process.

## Project layout

- `llm_arm_planner/scene.py` — workspace state (object positions). Currently
  hardcoded in `get_demo_scene()`. Replace this with a call into your
  existing OpenCV/MediaPipe pipeline to get live object positions from the
  camera feed used in SOLARIS.
- `llm_arm_planner/actions.py` — the fixed, validated action vocabulary
  (`move_to`, `open_gripper`, `close_gripper`, `wait`, `go_home`). Extend
  this as your arm gains more capabilities (e.g. `rotate_wrist`).
- `llm_arm_planner/planner.py` — calls the Gemini API with a system prompt
  that constrains output to the action schema, then parses and validates
  the JSON response. Raises `PlanningError` on malformed or invalid plans.
- `llm_arm_planner/arm_interface.py` — `MockArmExecutor` (pure Python, no
  ROS2 needed) and `Ros2ArmExecutor` (talks to the real `arm_sim_node` ROS2
  action server over `MoveTo` / `Gripper` actions — see `ros2_arm_sim/`).
- `demo_ros2.py` — same interactive loop as `demo.py`, but backed by
  `Ros2ArmExecutor` + the ROS2 simulation instead of the mock.

## Moving from simulation to real hardware later

`Ros2ArmExecutor` already speaks real ROS2 actions — it doesn't know or care
whether `arm_sim_node` is a simulation or real motors underneath. To point
it at real hardware eventually:

1. Write a new action server node (like `arm_sim_node.py`, same action
   types) that drives your real motors instead of interpolating in memory.
2. Run that node instead of `arm_sim_node` — same action names
   (`arm_sim/move_to`, `arm_sim/gripper`), or update the `MOVE_ACTION` /
   `GRIPPER_ACTION` constants in `Ros2ArmExecutor` to match new names.
3. Nothing on the Python planner side changes at all.

## Extending toward vision grounding

Once this works with hardcoded scenes, the natural next step (see the
"Multimodal Grounding" idea) is to replace `get_demo_scene()` with a
function that runs your OpenCV pipeline, detects objects, and returns
their real positions in the arm's coordinate frame — so instructions like
"pick up the small screwdriver" work on a live camera feed instead of a
hardcoded object list.

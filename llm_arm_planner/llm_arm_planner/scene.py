"""
Scene state for the arm workspace.

For now this is hardcoded. Later, qill replace `get_scene()` with a call into
your existing OpenCV/MediaPipe pipeline that returns real detected object
positions (e.g. from a calibrated camera-to-arm-frame transform).
"""

from dataclasses import dataclass, field


@dataclass
class SceneObject:
    name: str
    color: str
    x: float
    y: float
    z: float

    def to_dict(self):
        return {
            "name": self.name,
            "color": self.color,
            "position": [self.x, self.y, self.z],
        }


@dataclass
class Scene:
    objects: list = field(default_factory=list)
    gripper_state: str = "open"  # "open" | "closed"
    home_position: tuple = (0.0, 0.0, 0.2)

    def to_prompt_context(self) -> str:
        """Render the scene as compact text for the LLM prompt."""
        lines = [f"Home position: {self.home_position}",
                 f"Gripper state: {self.gripper_state}",
                 "Objects in workspace:"]
        for obj in self.objects:
            lines.append(
                f"  - {obj.color} {obj.name} at position "
                f"({obj.x:.2f}, {obj.y:.2f}, {obj.z:.2f})"
            )
        return "\n".join(lines)


def get_demo_scene() -> Scene:
    """Hardcoded demo scene. Replace with live vision data later."""
    return Scene(
        objects=[
            SceneObject(name="block", color="red", x=0.30, y=0.10, z=0.05),
            SceneObject(name="block", color="blue", x=0.30, y=-0.10, z=0.05),
            SceneObject(name="cup", color="green", x=0.20, y=0.00, z=0.08),
        ],
        gripper_state="open",
        home_position=(0.0, 0.0, 0.20),
    )

from dataclasses import dataclass
from typing import Optional


VALID_ACTION_TYPES = {
    "move_to",       # params: x, y, z
    "open_gripper",  # params: none
    "close_gripper", # params: none
    "wait",          # params: seconds
    "go_home",       # params: none
}


class InvalidActionError(ValueError):
    pass


@dataclass
class ArmAction:
    type: str
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    seconds: Optional[float] = None

    def __post_init__(self):
        if self.type not in VALID_ACTION_TYPES:
            raise InvalidActionError(
                f"'{self.type}' is not a valid action. "
                f"Must be one of: {sorted(VALID_ACTION_TYPES)}"
            )
        if self.type == "move_to":
            if None in (self.x, self.y, self.z):
                raise InvalidActionError("move_to requires x, y, z")
        if self.type == "wait" and self.seconds is None:
            raise InvalidActionError("wait requires seconds")

    @staticmethod
    def from_dict(d: dict) -> "ArmAction":
        if "type" not in d:
            raise InvalidActionError(f"Action missing 'type' field: {d}")
        return ArmAction(
            type=d["type"],
            x=d.get("x"),
            y=d.get("y"),
            z=d.get("z"),
            seconds=d.get("seconds"),
        )

    def __str__(self):
        if self.type == "move_to":
            return f"move_to(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"
        if self.type == "wait":
            return f"wait({self.seconds}s)"
        return f"{self.type}()"

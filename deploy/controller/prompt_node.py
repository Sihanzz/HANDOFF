"""LLM-planned natural-language controller node.

Bridges the viser "Prompt" text box (see deploy/sim/viser_bridge.py) to the
unified /g1/command topic. This replaces the *task-decomposition* half of the
real "molmo" agentic planner (paper Fig. 4) with a genuine LLM call (see
llm_planner.py) that turns free-form text into an ordered action sequence.

It is still NOT the paper's VLM *visual-grounding* half (SAM2 object
localization + FoundationStereoPose waypoint emission) — that pipeline was
never included in this public repo and reproducing it is out of scope here.
If the LLM call fails for any reason (no API key, network error, bad
response), this node falls back to the original regex keyword parser so a
demo never silently does nothing.

Subscribes:
  /molmo/ui/prompt_submit (std_msgs/String) — raw prompt text from viser.

Publishes:
  /g1/command            (Float32MultiArray) — same 18-float command every
                          other controller node (xbox/keyboard/dds_xr) writes.
  /molmo/ui/status_text   (std_msgs/String) — echoed into the viser "Status" panel.
  /molmo/ui/plan          (std_msgs/String) — newline-separated steps, shown
                          as the viser "Plan" panel.

Usage (with the sim already running, controller mode "none"):
  uv run --python 3.12 python deploy/controller/prompt_node.py
"""

import re
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String as StringMsg

from deploy.controller.llm_planner import LLMPlannerError, decompose as llm_decompose

from deploy.common.command import (
    CMD_HEIGHT,
    CMD_LEFT_HAND,
    CMD_RIGHT_HAND,
    CMD_VX,
    CMD_VY,
    CMD_YAW_RATE,
    COMMAND_TOPIC,
    make_command,
)
from wbc_mjlab.g1_constants_custom import (
    NOMINAL_COMMAND,
    NOMINAL_LEFT_HAND_BODY,
    NOMINAL_RIGHT_HAND_BODY,
)
from teleop_common import (
    DEFAULT_HAND_X,
    DEFAULT_HAND_Y,
    DEFAULT_HAND_Z,
    HAND_NEG_LIMIT_XYZ,
    HAND_POS_LIMIT_XYZ,
    HEIGHT_MAX,
    HEIGHT_MIN,
    MAX_VX,
    MAX_VY,
    MAX_YAW,
    PUBLISH_RATE_HZ,
    VIZ_QOS,
)

# Walking/turning/strafing commands auto-revert to zero after this many
# seconds so a stray prompt doesn't send the robot walking indefinitely.
LOCOMOTION_HOLD_S = 3.0
WAVE_HOLD_S = 2.0
WALK_SPEED = 0.5
STRAFE_SPEED = 0.4
YAW_SPEED = 0.6
SQUAT_HEIGHT = 0.55
WAVE_LIFT_Z = 0.20

# (regex patterns, handler name, human-readable plan step)
_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"forward|walk forward|往前|前进|走过去|走"), "forward", "Walk forward"),
    (re.compile(r"backward|back up|后退|往后|倒退"), "backward", "Walk backward"),
    (re.compile(r"turn left|左转"), "turn_left", "Turn left"),
    (re.compile(r"turn right|右转"), "turn_right", "Turn right"),
    (re.compile(r"strafe left|move left|左移"), "strafe_left", "Strafe left"),
    (re.compile(r"strafe right|move right|右移"), "strafe_right", "Strafe right"),
    (re.compile(r"squat|crouch|蹲下|蹲"), "squat", "Squat down"),
    (re.compile(r"stand up|stand|站起来|起立|站直"), "stand", "Stand up"),
    (re.compile(r"wave|raise.*hand|挥手|举手|打招呼"), "wave", "Wave right hand"),
    (re.compile(r"stop|halt|reset|停止|停下|复位"), "stop", "Stop and hold"),
]


class PromptNode(Node):
    def __init__(self):
        super().__init__("prompt_node")

        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0
        self._height = float(NOMINAL_COMMAND[CMD_HEIGHT])
        self._right_hand = np.array(
            [DEFAULT_HAND_X, DEFAULT_HAND_Y, DEFAULT_HAND_Z], dtype=np.float32
        )

        # Step queue for sequential plan execution: a multi-step LLM plan
        # (e.g. forward -> turn_left -> wave) runs one step at a time, each
        # held for its own duration, instead of all steps being applied
        # concurrently. See _advance_step / _tick.
        self._step_queue: list[tuple[str, str]] = []
        self._current_step_until: float = 0.0

        self._cmd_pub = self.create_publisher(Float32MultiArray, COMMAND_TOPIC, VIZ_QOS)
        self._status_pub = self.create_publisher(StringMsg, "/molmo/ui/status_text", VIZ_QOS)
        self._plan_pub = self.create_publisher(StringMsg, "/molmo/ui/plan", VIZ_QOS)

        self.create_subscription(
            StringMsg, "/molmo/ui/prompt_submit", self._on_prompt, VIZ_QOS
        )

        self._timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._tick)

        self.get_logger().info(
            "prompt_node ready — LLM task decomposer (regex fallback), not the real molmo planner."
        )
        self._publish_status("Ready. Describe what you want the robot to do.")

    # ---------------------------------------------------------------------
    # Prompt parsing
    # ---------------------------------------------------------------------
    def _on_prompt(self, msg: StringMsg) -> None:
        text = (msg.data or "").strip()
        if not text:
            return

        matched: list[str] = []
        steps: list[str] = []
        source = "llm"
        note = ""
        try:
            plan = llm_decompose(text)
            for action, desc in plan[0]:
                matched.append(action)
                steps.append(desc)
            note = plan[1]
        except LLMPlannerError as e:
            self.get_logger().warn(f"LLM planner failed ({e}); falling back to regex rules.")
            source = "regex-fallback"
            matched, steps = self._parse_with_rules(text)

        if not matched:
            self._publish_status(f'Not recognized: "{text}"')
            self._publish_plan(
                "No plan produced. Recognized primitives:\n"
                "forward / backward / turn left / turn right / strafe left / "
                "strafe right / squat / stand / wave / stop"
            )
            return

        self.get_logger().info(f'Prompt "{text}" -> {matched} (source={source})')
        status = f'Executing ({source}): {", ".join(matched)}'
        if note:
            status += f" — {note}"
        self._publish_status(status)
        self._publish_plan("\n".join(steps))

        # Run the plan as a sequence, not all steps at once: replace whatever
        # was queued/in-flight before with the new queue.
        self._step_queue = list(zip(matched, steps))
        self._advance_step()

    def _parse_with_rules(self, text: str) -> tuple[list[str], list[str]]:
        """Original regex keyword matcher — fallback when the LLM call fails."""
        text_lower = text.lower()
        matched: list[str] = []
        steps: list[str] = []
        for pattern, action, step_desc in _RULES:
            if pattern.search(text_lower):
                matched.append(action)
                steps.append(step_desc)
        return matched, steps

    def _advance_step(self) -> None:
        """Pop and start the next queued step. Zeroes transient velocities
        first so a step never inherits leftover motion from the previous one
        (that's what made concurrent-application look sequential by
        accident for same-axis repeats, but broke cross-axis sequences like
        forward -> turn -> wave)."""
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0
        self._right_hand[2] = DEFAULT_HAND_Z

        if not self._step_queue:
            self._current_step_until = float("inf")
            self._clamp_state()
            return

        action, _desc = self._step_queue.pop(0)
        now = time.time()
        hold = 0.0  # 0 = instantaneous (state-change actions), advance next tick.

        if action == "forward":
            self._vx = WALK_SPEED
            hold = LOCOMOTION_HOLD_S
        elif action == "backward":
            self._vx = -WALK_SPEED
            hold = LOCOMOTION_HOLD_S
        elif action == "strafe_left":
            self._vy = STRAFE_SPEED
            hold = LOCOMOTION_HOLD_S
        elif action == "strafe_right":
            self._vy = -STRAFE_SPEED
            hold = LOCOMOTION_HOLD_S
        elif action == "turn_left":
            self._yaw = YAW_SPEED
            hold = LOCOMOTION_HOLD_S
        elif action == "turn_right":
            self._yaw = -YAW_SPEED
            hold = LOCOMOTION_HOLD_S
        elif action == "squat":
            self._height = SQUAT_HEIGHT
        elif action == "stand":
            self._height = float(NOMINAL_COMMAND[CMD_HEIGHT])
        elif action == "wave":
            self._right_hand[2] = DEFAULT_HAND_Z + WAVE_LIFT_Z
            hold = WAVE_HOLD_S
        elif action == "stop":
            self._step_queue.clear()  # abort the rest of the plan; velocities already zeroed above

        self._current_step_until = now + hold
        self._clamp_state()

        if hold <= 0.0:
            # Instantaneous action (squat/stand/stop) — start the next
            # queued step right away instead of waiting a tick.
            self._advance_step()

    def _clamp_state(self) -> None:
        self._vx = float(np.clip(self._vx, -MAX_VX, MAX_VX))
        self._vy = float(np.clip(self._vy, -MAX_VY, MAX_VY))
        self._yaw = float(np.clip(self._yaw, -MAX_YAW, MAX_YAW))
        self._height = float(np.clip(self._height, HEIGHT_MIN, HEIGHT_MAX))
        self._right_hand = np.clip(self._right_hand, HAND_NEG_LIMIT_XYZ, HAND_POS_LIMIT_XYZ)

    # ---------------------------------------------------------------------
    # Publish loop
    # ---------------------------------------------------------------------
    def _tick(self) -> None:
        now = time.time()
        if now >= self._current_step_until:
            self._advance_step()

        cmd = make_command()
        cmd[CMD_VX] = self._vx
        cmd[CMD_VY] = self._vy
        cmd[CMD_YAW_RATE] = self._yaw
        cmd[CMD_HEIGHT] = self._height
        # Mirror left hand from right (matches keyboard_node's default mode):
        # same x/z, negated y.
        left_hand = np.array(
            [self._right_hand[0], -self._right_hand[1], self._right_hand[2]], dtype=np.float32
        )
        cmd[CMD_LEFT_HAND:CMD_LEFT_HAND + 3] = NOMINAL_LEFT_HAND_BODY + left_hand
        cmd[CMD_RIGHT_HAND:CMD_RIGHT_HAND + 3] = NOMINAL_RIGHT_HAND_BODY + self._right_hand

        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        self._cmd_pub.publish(msg)

    def _publish_status(self, text: str) -> None:
        msg = StringMsg()
        msg.data = text
        self._status_pub.publish(msg)

    def _publish_plan(self, text: str) -> None:
        msg = StringMsg()
        msg.data = text
        self._plan_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PromptNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

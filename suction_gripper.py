"""
Suction-cup gripper: replaces the two-finger Panda gripper with a vacuum pad.

Physics model
─────────────
MuJoCo has no native vacuum physics.  We approximate suction in two layers:

Layer 1 – High-friction pad (always active)
    The suction_pad_collision geom uses condim=6 and large friction coefficients
    so the pad resists sliding when it is pressed against an object surface.

Layer 2 – Equality weld constraint (activated by SuctionEnvWrapper)
    When action > 0 (suction ON), an external wrapper
    (see libero/envs/suction_env_wrapper.py) creates a "weld" equality
    constraint between the grip_site and the nearest contacted body.
    When action ≤ 0 (suction OFF), the constraint is removed.

    This two-layer design means the gripper works even WITHOUT the wrapper
    (objects will stick somewhat due to high friction) but performs much
    better with it.

Action space
────────────
    dof = 1
    action ∈ [-1, 1]
    > 0  → suction ON  (dummy actuator drives joint to +max)
    ≤ 0  → suction OFF (dummy actuator drives joint to -max / 0)
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class SuctionGripper(GripperModel):
    """
    Vacuum / suction-cup end-effector.

    Args:
        idn (int or str): unique ID string for this gripper instance.
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/suction_gripper.xml"), idn=idn)

    # ------------------------------------------------------------------
    # GripperModel interface
    # ------------------------------------------------------------------

    def format_action(self, action):
        """
        Map a single scalar action ∈ [-1, 1] to the dummy actuator command.

        The actuator's ctrlrange is [-1, 1].  We pass the raw action through
        so that external wrappers can read 'current_action' to detect
        whether suction is requested.

        Args:
            action (np.array): shape (1,)

        Returns:
            np.array: actuator command, shape (1,)
        """
        assert len(action) == self.dof, (
            f"SuctionGripper expects dof={self.dof} actions, got {len(action)}"
        )
        # Clamp to valid range and store
        self.current_action = np.clip(np.array(action), -1.0, 1.0)
        return self.current_action

    @property
    def init_qpos(self):
        """Initial position of the dummy joint (locked at 0)."""
        return np.array([0.0])

    @property
    def speed(self):
        """Suction cup has no opening speed."""
        return 0.2  # how fast the valve "signal" changes

    @property
    def dof(self):
        """One control dimension: vacuum on/off."""
        return 1

    # ------------------------------------------------------------------
    # Important contact geoms (used by robosuite for grasp detection)
    # ------------------------------------------------------------------

    @property
    def _important_geoms(self):
        return {
            "suction_pad": ["suction_pad_collision"],
            "left_finger": ["suction_pad_collision"],   # compatibility alias
            "right_finger": ["suction_pad_collision"],  # compatibility alias
            "left_fingerpad": ["suction_pad_collision"],
            "right_fingerpad": ["suction_pad_collision"],
        }

    # ------------------------------------------------------------------
    # Helpers used by SuctionEnvWrapper
    # ------------------------------------------------------------------

    @property
    def suction_active(self):
        """True if the current action requests vacuum ON."""
        return float(self.current_action[0]) > 0.0

    @property
    def contact_site_name(self):
        """Name of the site used to detect object contact."""
        return "suction_site"

    @property
    def indicator_site_name(self):
        """Name of the site used for suction on/off visual feedback."""
        return "suction_indicator"

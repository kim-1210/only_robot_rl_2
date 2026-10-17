from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class ReachImitationSceneCfg(InteractiveSceneCfg):
    ground: AssetBaseCfg = MISSING
    light: AssetBaseCfg = MISSING
    table: AssetBaseCfg = MISSING
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING


@configclass
class ReachImitationCommandCfg:
    pos_x = (-0.2, 0.2)
    pos_y = (-0.7, -0.4)
    pos_z = (0.15, 0.5)
    roll = (0.0, 0.0)
    pitch = (3.141592, 3.141592)
    yaw = (-3.14, 3.14)
    resample_interval_s = 1e10
    move_range_x = 0.4
    move_range_y = 0.4
    move_range_z = 0.4


@configclass
class ReachImitationActionCfg:
    arm_joint_names = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")
    actuated_joint_names = (
        "base",
        "shoulder",
        "elbow",
        "wrist1",
        "wrist2",
        "wrist3",
        "virual_ee_link",
        "robotiq_hande_left_finger_joint",
        "robotiq_hande_right_finger_joint",
    )
    left_finger_joint_name = "robotiq_hande_left_finger_joint"
    right_finger_joint_name = "robotiq_hande_right_finger_joint"
    arm_action_scale = 0.01
    gripper_close_threshold = 0.012


@configclass
class ReachImitationObservationCfg:
    action_space = 9
    observation_space = 25
    state_space = 0
    success_threshold = 0.04
    target_marker_radius = 0.01


@configclass
class ReachImitationEnvCfg(DirectRLEnvCfg):
    scene: ReachImitationSceneCfg = ReachImitationSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=2)
    commands: ReachImitationCommandCfg = ReachImitationCommandCfg()
    actions: ReachImitationActionCfg = ReachImitationActionCfg()
    observations: ReachImitationObservationCfg = ReachImitationObservationCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 12.0
        self.sim.render_interval = self.decimation
        # Keep the reach_lift-style nested config while remaining compatible
        # with older direct-env launch code that expects flat fields.
        self.action_space = self.observations.action_space
        self.observation_space = self.observations.observation_space
        self.state_space = self.observations.state_space
        self.success_threshold = self.observations.success_threshold

        self.arm_joint_names = self.actions.arm_joint_names
        self.actuated_joint_names = self.actions.actuated_joint_names
        self.left_finger_joint_name = self.actions.left_finger_joint_name
        self.right_finger_joint_name = self.actions.right_finger_joint_name
        self.arm_action_scale = self.actions.arm_action_scale
        self.gripper_close_threshold = self.actions.gripper_close_threshold

        self.command_pos_x = self.commands.pos_x
        self.command_pos_y = self.commands.pos_y
        self.command_pos_z = self.commands.pos_z
        self.command_roll = self.commands.roll
        self.command_pitch = self.commands.pitch
        self.command_yaw = self.commands.yaw
        self.command_resample_interval_s = self.commands.resample_interval_s
        self.command_move_range_x = self.commands.move_range_x
        self.command_move_range_y = self.commands.move_range_y
        self.command_move_range_z = self.commands.move_range_z
        self.target_marker_radius = self.observations.target_marker_radius

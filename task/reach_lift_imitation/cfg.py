from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


@configclass
class ReachLiftImitationSceneCfg(InteractiveSceneCfg):
    ground: AssetBaseCfg = MISSING
    light: AssetBaseCfg = MISSING
    table: AssetBaseCfg = MISSING
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    obj: RigidObjectCfg = MISSING
    obj_frame: FrameTransformerCfg = MISSING


@configclass
class ReachLiftImitationCommandCfg:
    pos_x = (-0.2, 0.2)
    pos_y = (-0.7, -0.3)
    pos_z = (0.2, 0.5)
    roll = (0.0, 0.0)
    pitch = (3.141592, 3.141592)
    yaw = (-3.14, 3.14)


@configclass
class ReachLiftImitationRandomizationCfg:
    object_reset_pos_x = (-0.2, 0.2)
    object_reset_pos_y = (-0.2, 0.2)
    object_reset_pos_z = 1.066
    object_reset_yaw = (0.0, 1.57)
    object_resample_interval_s = 1e5


@configclass
class ReachLiftImitationActionCfg:
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
    arm_action_scale = 0.5
    gripper_close_threshold = 0.013


@configclass
class ReachLiftImitationObservationCfg:
    action_space = 9
    observation_space = 28
    state_space = 0


@configclass
class ReachLiftImitationTerminationCfg:
    minimum_object_height = 1.03
    success_object_height = 1.14


@configclass
class ReachLiftImitationEnvCfg(DirectRLEnvCfg):
    scene: ReachLiftImitationSceneCfg = ReachLiftImitationSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
            friction_combine_mode="max",
        ),
    )
    commands: ReachLiftImitationCommandCfg = ReachLiftImitationCommandCfg()
    randomization: ReachLiftImitationRandomizationCfg = ReachLiftImitationRandomizationCfg()
    actions: ReachLiftImitationActionCfg = ReachLiftImitationActionCfg()
    observations: ReachLiftImitationObservationCfg = ReachLiftImitationObservationCfg()
    terminations: ReachLiftImitationTerminationCfg = ReachLiftImitationTerminationCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 30.0
        self.sim.render_interval = self.decimation
        # Keep the reach_lift-style nested config while remaining compatible
        # with older direct-env launch code that expects flat fields.
        self.action_space = self.observations.action_space
        self.observation_space = self.observations.observation_space
        self.state_space = self.observations.state_space
        self.minimum_object_height = self.terminations.minimum_object_height
        self.success_object_height = self.terminations.success_object_height

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

        self.object_reset_pos_x = self.randomization.object_reset_pos_x
        self.object_reset_pos_y = self.randomization.object_reset_pos_y
        self.object_reset_pos_z = self.randomization.object_reset_pos_z
        self.object_reset_yaw = self.randomization.object_reset_yaw
        self.object_resample_interval_s = self.randomization.object_resample_interval_s

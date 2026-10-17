"""
PYTHONPATH=$PWD:/home/kim/Desktop/only_robot_rl \
isaaclab -p /home/kim/Desktop/only_robot_rl/launch/rsl_rl_train.py \
--task reach --num_envs 2000 --max_iterations 10000 --seed 42 --ui
"""

from dataclasses import MISSING

import torch 

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

from isaaclab.sensors import CameraCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.scene import InteractiveSceneCfg

from . import mdp
from robot.rb5_gripper import RB5_850E_HAND_E_CFG

@configclass
class MyScene(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -0.6, 1.05),
        ),
    )

    robot = RB5_850E_HAND_E_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.05),  
            joint_pos={
                "base": -1.57, "shoulder": 0.0, "elbow": 0.7,
                "wrist1": 0.87, "wrist2": 1.57, "wrist3": 1.57,
                "robotiq_hande_left_finger_joint": 0.025,
                "robotiq_hande_right_finger_joint": 0.025,
            },
        ),
        # Drive only the left finger directly and let the right finger
        # follow via the asset's mimic relationship.
        actuators={
            **{name: cfg for name, cfg in RB5_850E_HAND_E_CFG.actuators.items() if name != "gripper"},
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["robotiq_hande_left_finger_joint"],
                effort_limit=200.0,
                velocity_limit=2.0,
                stiffness=2500.0,
                damping=450.0,
            ),
        },
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                name="ee",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.135],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/robotiq_hande_right_finger",
                name="right_finger",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.00465],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/robotiq_hande_left_finger",
                name="left_finger",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0465],
                ),
            ),
        ],
    )

@configclass
class EventCfg:
    # reset_robot_pose = EventTerm(
        # func=mdp.reset_robot_with_init_pose,
        # mode="reset",
    # )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class TerminationCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )
    success = DoneTerm(
        func=mdp.success_cube,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
            "command_name": "ee_pose",
        },
    )

@configclass
class CommandCfg:
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=["ee_link"],
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.2, 0.2),
            pos_y=(-0.7, -0.4),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(3.141592, 3.141592),
            yaw=(-3.14, 3.14),
        ),
    )

@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={
            "term_name": "action_rate", 
            "weight": -0.005, 
            "num_steps": 4500
        }
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={
            "term_name": "joint_vel", 
            "weight": -0.001, 
            "num_steps": 4500
        }
    )

@configclass
class ActionCfg:
    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name = "robot",
        joint_names=[
            "base",
            "shoulder",
            "elbow",
            "wrist1",
            "wrist2",
            "wrist3",
        ],
        scale=0.01,
        use_default_offset=True
    )

    gripper_joint_pos = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "robotiq_hande_left_finger_joint",
        ],
        open_command_expr={
            "robotiq_hande_left_finger_joint": 0.025,
        },
        close_command_expr={
            "robotiq_hande_left_finger_joint": 0.0,
        },
    )

@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm( # 0~8
            func= mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm( # 9~17
            func= mdp.joint_vel,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        pose_command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "ee_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardCfg:
    # task terms
    end_effector_position_tracking = RewardTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
            "command_name": "ee_pose",
        },
    )
    end_effector_position_tracking_fine_grained = RewardTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]), 
            "std": 0.1, 
            "command_name": "ee_pose",
        },
    )
    end_effector_orientation_tracking = RewardTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]), 
            "command_name": "ee_pose",
        },
    )

    # action penalty
    action_rate = RewardTerm(
        func=mdp.action_rate_l2, 
        weight=-0.0001
    )
    joint_vel = RewardTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class ReachCfg(ManagerBasedRLEnvCfg):
    scene:MyScene = MyScene()
    events:EventCfg = EventCfg()
    terminations:TerminationCfg = TerminationCfg()
    curriculums:CurriculumCfg = CurriculumCfg()
    actions:ActionCfg = ActionCfg()
    observations:ObservationCfg = ObservationCfg()
    rewards:RewardCfg = RewardCfg()
    commands:CommandCfg = CommandCfg()

    def __post_init__(self,):
        super().__post_init__()

        self.gripper_joint_names = ["robotiq_hande_left_finger_joint"]
        self.gripper_open_val = 0.025

        self.decimation = 2
        self.sim.render_interval = self.decimation
        # self.episode_length_s = (150.0)
        self.episode_length_s = 20.0
        # self.sim.dt = 0.01
        self.sim.dt = 1.0 / 60.0
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

class ReachEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 0: approch, 1: grasp, 2: lift, 3: target move, 4: drop

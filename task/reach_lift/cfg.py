"""
PYTHONPATH=$PWD:/media/kim/hdd1tb/master_study/only_robot_rl \
isaaclab -p ./launch/rsl_rl_train.py \
--task reach_lift --num_envs 8000 --max_iterations 20000 --seed 42
"""
from dataclasses import MISSING

import torch 

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from isaaclab.managers import ActionTermCfg as ActionTerm
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

    robot:ArticulationCfg = MISSING

    ee_frame:FrameTransformerCfg = MISSING

    obj:RigidObjectCfg | DeformableObjectCfg = MISSING

    obj_frame:FrameTransformerCfg = MISSING

@configclass
class EventCfg:
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default, 
        mode="reset"
    )

    obj_random_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("obj"),
            "pose_range":{
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0, 0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 1.57),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
    )

    # reset_phase = EventTerm(
    #     func=mdp.reset_phase,
    #     mode="reset"
    # )

    # change_phase = EventTerm(
    #     func=mdp.change_phase,
    #     mode="step",
    #     params={
    #         "obj_cfg": SceneEntityCfg("obj"),
    #         "threshold_1": 0.01,
    #     }
    # )

    # zero_phase = EventTerm(
    #     func=mdp.zero_phase,
    #     mode="step",
    #     params={
    #         "obj_cfg":SceneEntityCfg("obj")
    #     }
    # )

@configclass
class TerminationCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={
            "minimum_height": 1.03, 
            "asset_cfg": SceneEntityCfg("obj"),
        }
    )
    success = DoneTerm(
        func=mdp.object_goal_reached,
        params={
            "threshold": 0.005,
            "minimal_height": 1.1,
            "command_name": "obj_pose",
        },
    )

@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={
            "term_name": "action_rate", 
            "weight": -1e-3, 
            "num_steps": 10000
        }
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={
            "term_name": "joint_vel", 
            "weight": -1e-3, 
            "num_steps": 10000
        }
    )

@configclass
class CommandCfg:
    # lift 위치
    obj_pose = mdp.UniformPoseCommandCfg(
        asset_name= "robot",
        body_name=MISSING,
        resampling_time_range= (7.0, 10.0),
        debug_vis= False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.2, 0.2),
            pos_y=(-0.7, -0.3),
            pos_z=(0.2, 0.5),
            roll=(0.0, 0.0),
            pitch=(3.141592, 3.141592),
            yaw=(-3.14, 3.14),
        )
    )

@configclass
class ActionCfg:
    arm_joint:ActionTerm = MISSING

    gripper:ActionTerm | None = None

@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(
            func=mdp.obj_pos_in_robot_root_frame
        )
        target_object_position = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "obj_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy:PolicyCfg = PolicyCfg()

@configclass
class RewardCfg:
    reaching_object = RewardTerm(
        func=mdp.object_ee_distance, 
        params={"std": 0.1}, weight=1.0
    )

    lifting_object = RewardTerm(
        func=mdp.object_is_lifted, 
        params={"minimal_height": 1.1}, 
        weight=15.0
    )

    object_goal_tracking = RewardTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3, "minimal_height": 1.1, 
            "command_name": "obj_pose"
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewardTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05, "minimal_height": 1.1, 
            "command_name": "obj_pose"
        },
        weight=5.0,
    )

    goal_success = RewardTerm(
        func=mdp.object_goal_success,
        params={
            "threshold": 0.005,
            "minimal_height": 1.1,
            "command_name": "obj_pose",
        },
        weight=10.0,
    )

    # action penalty
    action_rate = RewardTerm(
        func=mdp.action_rate_l2, 
        weight=-1e-4
    )

    joint_vel = RewardTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
@configclass
class ReachLiftCfg(ManagerBasedRLEnvCfg):
    scene:MyScene = MyScene(num_envs=4096, env_spacing=2.5)
    rewards:RewardCfg = RewardCfg()
    observations:ObservationCfg = ObservationCfg()
    events:EventCfg = EventCfg()
    actions:ActionCfg = ActionCfg()
    terminations:TerminationCfg = TerminationCfg()
    commands:CommandCfg = CommandCfg()
    curriculums:CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2 # 몇 step마다 action을 적용할지
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01
        self.sim.physics_material = RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
            friction_combine_mode="max",
        )
        
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

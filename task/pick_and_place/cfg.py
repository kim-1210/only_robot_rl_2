"""
PYTHONPATH=$PWD:/home/kdy1210/isaac/only_robot_rl \
isaaclab -p ./launch/rsl_rl_train.py \
--task pick_and_place --num_envs 4096 --max_iterations 50000 --seed 42
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.scene import InteractiveSceneCfg

from . import mdp as mdp
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
    obj1:RigidObjectCfg | DeformableObjectCfg = MISSING
    obj2:RigidObjectCfg | DeformableObjectCfg = MISSING

@configclass
class EventCfg():
    init_arm_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose":[
                -1.57, 0.0, 0.7, 0.87, 1.57, 1.57
            ]
        }
    )
    randomize_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    randomize_cube_positions = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_ranges": [
                {
                    "x": (-0.2, 0.2),
                    "y": (-0.7, -0.3),
                    "z": (1.066, 1.066),
                    "yaw": (-1.57, 1.57),
                },
                {
                    "x": (-0.2, 0.2),
                    "y": (-0.7, -0.3),
                    "z": (1.076, 1.076),
                    "yaw": (-1.57, 1.57),
                },
            ],
            "min_separation": 0.1,
            "asset_cfgs": [
                SceneEntityCfg("obj1"), 
                SceneEntityCfg("obj2")
            ],
        },
    )

@configclass
class TerminationCfg():
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out = True,
    )
    object1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={
            "minimum_height": 1.03, 
            "asset_cfg": SceneEntityCfg("obj1"),
        }
    )
    object2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={
            "minimum_height": 1.03, 
            "asset_cfg": SceneEntityCfg("obj2"),
        }
    )
    success = DoneTerm(func=mdp.cubes_stacked)

@configclass
class CurriculumCfg():
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name":"action_rate",
            "weight": -2e-2,
            "num_steps": 256000,
        }
    )
    joint_vel = CurrTerm(
        func = mdp.modify_reward_weight,
        params = {
            "term_name": "joint_vel",
            "weight": -2e-2,
            "num_steps": 256000,
        }
    )
    obj_ee_dist = CurrTerm(
        func = mdp.modify_reward_weight,
        params={
            "term_name": "reaching_obj",
            "weight": 0.1,
            "num_steps": 128000,
        }
    )
    obj_grasp = CurrTerm(
        func = mdp.modify_reward_weight,
        params={
            "term_name": "obj_grasp",
            "weight": 1.0,
            "num_steps": 128000,
        }
    )
    obj_lifting = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "obj_lifting",
            "weight": 1.5,
            "num_steps": 128000,
        },
    )
    objs_stacking = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "objs_stacking",
            "weight": 5.0,
            "num_steps": 320000,
        },
    )
    obj_height_alignment = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "obj_height_alignment",
            "weight": 6.0,
            "num_steps": 448000,
        },
    )
    stacking_success = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "stacking_success",
            "weight": 12.0,
            "num_steps": 576000,
        },
    )
    

@configclass
class ActionCfg():
    arm_joint:mdp.JointPositionActionCfg = MISSING
    gripper:mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationCfg():
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # world기준 obj들의 위치와 회전값
        obj_poses = ObsTerm(func=mdp.objs_pose)
        # world기준 ee의 위치
        ee_pose_w = ObsTerm(func=mdp.ee_pose)
        # each env 기준 ee의 위치
        ee_pose_env = ObsTerm(func=mdp.ee_pos_to_each_env)
        # ee obj1 obj2 dist
        ee_obj_dist = ObsTerm(func=mdp.ee_obj_dist)
        # gripper joint값
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardCfg():
    action_rate = RewardTerm(
        func=mdp.action_rate_l2, 
        weight=-0.005
    )
    joint_vel = RewardTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    reaching_obj = RewardTerm(
        func=mdp.obj_ee_dist_reward,
        params={
            "std": 0.2,
        },
        weight=0.5,
    )
    obj_grasp = RewardTerm(
        func=mdp.obj_grasp_dense_reward,
        params={
            "diff_threshold": 0.04,
            "gripper_close_scale": 0.02,
        },
        weight=1.0
    )
    obj_lifting = RewardTerm(
        func=mdp.lifting_obj_progress_reward,
        params={
            "minimal_height": 1.13,
            "base_height": 1.066,
        },
        weight=0.75
    )
    objs_stacking = RewardTerm(
        func=mdp.obj_stacking_reward,
        params={
            "std": 0.05,
            "minimal_height": 1.13,
            "base_height": 1.066,
            "add_target_height": 0.04,
            "height_std": 0.03,
        },
        weight=1.5
    )
    obj_height_alignment = RewardTerm(
        func=mdp.obj_height_alignment_reward,
        params={
            "std": 0.03,
            "target_height": 0.04,
            "xy_std": 0.08,
            "minimal_height": 1.13,
            "base_height": 1.066,
        },
        weight=1.0
    )
    stacking_success = RewardTerm(
        func=mdp.stacking_success_reward,
        params={
            "xy_threshold": 0.04,
            "height_threshold": 0.005,
            "height_diff": 0.04,
            "xy_std": 0.06,
            "height_std": 0.015,
            "release_std": 0.015,
        },
        weight=2.0
    )

@configclass
class PickAndPlaceCfg(ManagerBasedRLEnvCfg):
    scene:MyScene = MyScene(num_envs=4096, env_spacing=2.5)
    rewards:RewardCfg = RewardCfg()
    observations:ObservationCfg = ObservationCfg()
    events:EventCfg = EventCfg()
    actions:ActionCfg = ActionCfg()
    terminations:TerminationCfg = TerminationCfg()
    curriculums:CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 5 # 몇 step마다 action을 적용할지
        self.sim.render_interval = 2
        self.episode_length_s = 40.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01

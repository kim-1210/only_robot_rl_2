import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg
)
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from robot.rb5_gripper import RB5_850E_HAND_E_CFG
from task.pick_and_place.cfg import PickAndPlaceCfg

@configclass
class PickAndPlaceCFG(PickAndPlaceCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = RB5_850E_HAND_E_CFG.replace(
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
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link0",
            debug_vis= False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.143),
                    ),
                ),
            ],
        )

        self.scene.obj1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obj1",
            spawn=sim_utils.CuboidCfg(
                size=(0.03,0.03,0.03),
        
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5.0,
                    solver_velocity_iteration_count=8,
                    solver_position_iteration_count=8,
                    disable_gravity=False,
                    max_angular_velocity=50.0,
                    max_linear_velocity=5.0,
                ),
        
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0
                ),
        
                physics_material=RigidBodyMaterialCfg(
                    static_friction=0.8,
                    dynamic_friction=0.6,
                    restitution=0.0,
                    friction_combine_mode="max"
                ),

                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, -0.5, 1.066),
                rot=(0.0,0.0,1.0,0.0),
            ),
        )

        self.scene.obj2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obj2",
            spawn=sim_utils.CuboidCfg(
                size=(0.05,0.05,0.05),
        
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5.0,
                    solver_velocity_iteration_count=8,
                    solver_position_iteration_count=8,
                    disable_gravity=False,
                    max_angular_velocity=50.0,
                    max_linear_velocity=5.0,
                ),
        
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0
                ),
        
                physics_material=RigidBodyMaterialCfg(
                    static_friction=0.8,
                    dynamic_friction=0.6,
                    restitution=0.0,
                    friction_combine_mode="max"
                ),

                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, -0.5, 1.076),
                rot=(0.0,0.0,1.0,0.0),
            ),
        )

        self.gripper_joint_names = [".*_finger_joint"]
        self.gripper_open_val = 0.025
        self.gripper_threshold = 0.013

        self.actions.arm_joint = mdp.JointPositionActionCfg(
            asset_name = "robot",
            joint_names=[
                "base",
                "shoulder",
                "elbow",
                "wrist1",
                "wrist2",
                "wrist3",
            ],
            scale=0.3,
            use_default_offset=True
        )
        self.actions.gripper = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "robotiq_hande_left_finger_joint",
                "robotiq_hande_right_finger_joint",
            ],
            open_command_expr={
                "robotiq_hande_left_finger_joint": 0.025,
                "robotiq_hande_right_finger_joint": 0.025,
            },
            close_command_expr={
                "robotiq_hande_left_finger_joint": 0.0,
                "robotiq_hande_right_finger_joint": 0.0,
            },
        )

@configclass
class PickAndPlaceCFG_Play(PickAndPlaceCFG):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
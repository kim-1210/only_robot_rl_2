import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robot.rb5_gripper import RB5_850E_HAND_E_CFG


@configclass
class ReachLiftStyleRobotSceneCfg(InteractiveSceneCfg):
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
                "base": -1.57,
                "shoulder": 0.0,
                "elbow": 0.7,
                "wrist1": 0.87,
                "wrist2": 1.57,
                "wrist3": 1.57,
                "robotiq_hande_left_finger_joint": 0.025,
                "robotiq_hande_right_finger_joint": 0.025,
            },
        ),
        actuators={
            **{name: cfg for name, cfg in RB5_850E_HAND_E_CFG.actuators.items() if name != "gripper"},
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["robotiq_hande_left_finger_joint"],
                effort_limit=450.0,
                velocity_limit=2.0,
                stiffness=9000.0,
                damping=1800.0,
            ),
        },
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link0",
        debug_vis=False,
        visualizer_cfg=None,
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


@configclass
class ReachLiftStyleSceneCfg(ReachLiftStyleRobotSceneCfg):
    obj = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obj",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=5.0,
                solver_velocity_iteration_count=16,
                solver_position_iteration_count=16,
                disable_gravity=False,
                max_angular_velocity=50.0,
                max_linear_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=2.5,
                dynamic_friction=2.0,
                restitution=0.0,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.8, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.5, 1.066),
            rot=(0.0, 0.0, 1.0, 0.0),
        ),
    )

    obj_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Obj",
        debug_vis=False,
        visualizer_cfg=None,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Obj",
                name="obj",
            ),
        ],
    )

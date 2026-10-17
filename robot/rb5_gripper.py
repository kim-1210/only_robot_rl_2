from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent

RB5_850E_HAND_E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/media/kim/hdd1tb2/master_study/urdf_ws/src/combine_urdf/urdf/test_gripper/test_gripper.usd",
        activate_contact_sensors=False,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "base": 1.57,
            "shoulder": 0.0,
            "elbow": 0.0,
            "wrist1": -1.57,
            "wrist2": -1.57,
            "wrist3": 1.57,

            "robotiq_hande_left_finger_joint": 0.025,
            "robotiq_hande_right_finger_joint": 0.025,
        },
    ),
    actuators={
        # BASE (무거움)
        "arm_base": ImplicitActuatorCfg(
            joint_names_expr=["base"],
            effort_limit=300.0,
            velocity_limit=2.0,
            stiffness=10000.0,
            damping=2000.0,
        ),

        # SHOULDER + ELBOW (무거움)
        "arm_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=["shoulder", "elbow"],
            effort_limit=200.0,
            velocity_limit=2.0,
            stiffness=8000.0,
            damping=1500.0,
        ),

        # WRIST 1-2 (중간)
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist[1-2]"],
            effort_limit=80.0,
            velocity_limit=3.0,
            stiffness=3500.0,
            damping=450.0,
        ),

        # WRIST 3 (가벼움)
        "arm_wrist3": ImplicitActuatorCfg(
            joint_names_expr=["wrist3"],
            effort_limit=60.0,
            velocity_limit=3.0,
            stiffness=2000.0,
            damping=250.0,
        ),

        # GRIPPER (접촉 많음)
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "robotiq_hande_left_finger_joint",
                "robotiq_hande_right_finger_joint",
            ],
            effort_limit=350.0,
            velocity_limit=2.0,
            stiffness=6000.0,
            damping=1200.0,
        ),
    }

    # actuators={
    #     "arm_base": ImplicitActuatorCfg(
    #         joint_names_expr=["base"],
    #         effort_limit=300.0, 
    #         velocity_limit=2.0,
    #         stiffness=8000.0,   
    #         damping=1500.0,      
    #     ),

    #     "arm_shoulder_elbow": ImplicitActuatorCfg(
    #         joint_names_expr=["shoulder", "elbow"],
    #         effort_limit=150.0,
    #         velocity_limit=2.0,
    #         stiffness=6000.0, 
    #         damping=1000.0,
    #     ),

    #     "arm_wrist": ImplicitActuatorCfg(
    #         joint_names_expr=["wrist[1-2]"],
    #         effort_limit=50.0,
    #         velocity_limit=3.0,  
    #         stiffness=3000.0,    
    #         damping=600.0,   
    #     ),

    #     "arm_wrist3": ImplicitActuatorCfg(
    #         joint_names_expr=["wrist3"],
    #         effort_limit=50.0,
    #         velocity_limit=3.0,  
    #         stiffness=1500.0,    
    #         damping=300.0,   
    #     ),

    #     "gripper": ImplicitActuatorCfg(
    #         joint_names_expr=[
    #             "robotiq_hande_left_finger_joint",
    #             "robotiq_hande_right_finger_joint",
    #         ],
    #         stiffness=3000.0,
    #         damping=500.0,
    #         effort_limit=200.0,
    #         velocity_limit=2.0,
    #     ),
    # },
)

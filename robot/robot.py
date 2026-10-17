from isaaclab_assets import UR10_CFG

# shoulder_pan_joint : 몸통 좌우 회전
# shoulder_lift_joint : 팔 들기/내리기
# elbow_joint : 팔 접기
# wrist_1_joint : 손목 회전 1
# wrist_2_joint : 손목 회전 2
# wrist_3_joint : 손목 회전 3
UR10 = UR10_CFG.copy()

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd",
    ),
)
# gripper : EE
UR10.attachments = {
    "gripper": ArticulationCfg.AttachmentCfg(
        asset_cfg=ROBOTIQ_CFG,
        parent_link_name="wrist_3_link",
        attach_pose=(0.0, 0.0, 0.08),
        attach_orientation=(0.0, 0.0, 0.0),
    )
}
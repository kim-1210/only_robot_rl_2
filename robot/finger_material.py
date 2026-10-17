from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, get_current_stage


FINGER_PRIM_SUFFIXES = (
    "/Robot/robotiq_hande_left_finger",
    "/Robot/robotiq_hande_right_finger",
)

RUBBER_VISUAL_MATERIAL_PATH = "/World/Looks/robot_finger_rubber"
RUBBER_PHYSICS_MATERIAL_PATH = "/World/Physics_Materials/robot_finger_rubber"

RUBBER_VISUAL_CFG = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.06, 0.06, 0.06),
    roughness=0.95,
    metallic=0.0,
)

RUBBER_PHYSICS_CFG = sim_utils.RigidBodyMaterialCfg(
    static_friction=4.0,
    dynamic_friction=3.5,
    restitution=0.0,
    friction_combine_mode="max",
)


def apply_rubber_finger_material() -> None:
    """Bind a high-friction matte material to all gripper finger prims on the stage."""

    stage = get_current_stage()

    if not stage.GetPrimAtPath(RUBBER_VISUAL_MATERIAL_PATH).IsValid():
        RUBBER_VISUAL_CFG.func(RUBBER_VISUAL_MATERIAL_PATH, RUBBER_VISUAL_CFG)
    if not stage.GetPrimAtPath(RUBBER_PHYSICS_MATERIAL_PATH).IsValid():
        RUBBER_PHYSICS_CFG.func(RUBBER_PHYSICS_MATERIAL_PATH, RUBBER_PHYSICS_CFG)

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.endswith(FINGER_PRIM_SUFFIXES):
            bind_visual_material(prim_path, RUBBER_VISUAL_MATERIAL_PATH, stage=stage)
            collision_path = f"{prim_path}/collisions"
            visual_path = f"{prim_path}/visuals"

            if stage.GetPrimAtPath(visual_path).IsValid():
                bind_visual_material(visual_path, RUBBER_VISUAL_MATERIAL_PATH, stage=stage)
            if stage.GetPrimAtPath(collision_path).IsValid():
                bind_physics_material(collision_path, RUBBER_PHYSICS_MATERIAL_PATH, stage=stage)
            else:
                bind_physics_material(prim_path, RUBBER_PHYSICS_MATERIAL_PATH, stage=stage)

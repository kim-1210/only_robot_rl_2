import gymnasium as gym

# ========== reach ==========
from task.reach.cfg import ReachCfg, ReachEnv
from task.reach.agent.rsl_rl_ppo_cfg import ReachPPORunnerCfg
gym.register(
    id="reach",
    entry_point="task.reach.cfg:ReachEnv",
    kwargs={
        "env_cfg_entry_point": "task.reach.cfg:ReachCfg",
        "rsl_rl_cfg_entry_point": "task.reach.agent.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
    },
)

# ========== reach_lift ==========
from task.reach_lift.cfg_settings import ReachLiftCFG, ReachLiftCFG_PLAY, ReachAndLiftEnv
from task.reach_lift.agent.rsl_rl_ppo_cfg import ReachLiftPPORunnerCfg
gym.register(
    id="reach_lift",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="task.reach_lift.cfg_settings:ReachAndLiftEnv",
    kwargs={
        "env_cfg_entry_point": "task.reach_lift.cfg_settings:ReachLiftCFG",
        "rsl_rl_cfg_entry_point": "task.reach_lift.agent.rsl_rl_ppo_cfg:ReachLiftPPORunnerCfg",
    },
)
import task.reach_lift.agent as reach_lift_agents
gym.register(
    id="reach_lift_skrl",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "task.reach_lift.cfg_settings:ReachLiftCFG",
        # "skrl_cfg_entry_point": f"{reach_lift_agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": "task.reach_lift.agent:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="reach_lift_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "task.reach_lift.cfg_settings:ReachLiftCFG_PLAY",
        "rsl_rl_cfg_entry_point": "task.reach_lift.agent.rsl_rl_ppo_cfg:ReachLiftPPORunnerCfg",
    },
)
# ========== pick_and_place ==========
from task.pick_and_place.cfg_settings import PickAndPlaceCFG, PickAndPlaceCFG_Play
from task.pick_and_place.agent.rsl_rl_ppo_cfg import PickandPlacePPORunnerCfg


gym.register(
    id="pick_and_place",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "task.pick_and_place.cfg_settings:PickAndPlaceCFG",
        "rsl_rl_cfg_entry_point": "task.pick_and_place.agent.rsl_rl_ppo_cfg:PickandPlacePPORunnerCfg",
    },
)
gym.register(
    id="pick_and_place_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "task.pick_and_place.cfg_settings:PickAndPlaceCFG_Play",
        "rsl_rl_cfg_entry_point": "task.pick_and_place.agent.rsl_rl_ppo_cfg:PickandPlacePPORunnerCfg",
    },
)

# ========== direct imitation play ==========
from task.reach_imitation.cfg_settings import ReachImitationCFG, ReachImitationEnv
from task.reach_lift_imitation.cfg_settings import ReachLiftImitationCFG, ReachLiftImitationEnv

gym.register(
    id="reach_imitation",
    entry_point="task.reach_imitation.cfg_settings:ReachImitationEnv",
    kwargs={
        "env_cfg_entry_point": "task.reach_imitation.cfg_settings:ReachImitationCFG",
    },
)

gym.register(
    id="reach_lift_imitation",
    entry_point="task.reach_lift_imitation.cfg_settings:ReachLiftImitationEnv",
    kwargs={
        "env_cfg_entry_point": "task.reach_lift_imitation.cfg_settings:ReachLiftImitationCFG",
    },
)

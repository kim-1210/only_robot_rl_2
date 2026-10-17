from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class ReachLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 수집할 데이터의 수
    num_steps_per_env = 128
    # 전체 학습 반복 횟수
    max_iterations = 1500
    save_interval = 200
    experiment_name = "rb5_850_reach_lift_task"
    run_name = ""
    # clip_actions = 10.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006, # 값을 높이면 강제로 탐색을 유도함
        num_learning_epochs=6, # 
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.014, # kl발산 허용치 -> 높이면 과감하게 변함
        max_grad_norm=1.0,
    )
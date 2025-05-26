import os
import random
import gym
import sconegym
import numpy as np
import torch.nn as nn
from gym.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from typing import Callable
from wrapper.H2190UnevenWrapper import H2190UnevenWrapper
from callback.UnevenCallback import UnevenCallback
from copy import deepcopy
import types

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ─── 2) Gym compatibility wrapper ─────────────────────────────────
class GymCompatibilityWrapper(gym.Wrapper):
    def step(self, action):
        result = self.env.step(action)
        # Gymnasium API: (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            # Classic Gym API: (obs, reward, done, info)
            obs, reward, done, info = result
            terminated = done
            truncated = False
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Gymnasium API: (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            # Classic Gym API: obs only
            obs, info = result, {}
        return obs, info
# ────────────────────────────────────────────────────────────────

# ─── 3) make_env 함수: env_id를 인자로 받아 환경 팩토리 반환 ────────────
def make_env():
    def _init():
        env = gym.make('sconewalk_h2190_terrain1-v1')
        env = H2190UnevenWrapper(
            env,
            syn_matrix=np.array([
                [ 0.00000000e+00, 0.00000000e+00, 1.43556358e-01, 0.00000000e+00, 2.28869292e-03, 6.31749443e-03, 2.62268060e-01, 0.00000000e+00, 0.00000000e+00, 1.38923007e-01, 3.39865776e-01, 4.13098342e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.14095654e-01, 1.09448373e-01, 5.07311192e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.53598609e-01, 9.25165204e-01, 7.77242013e-01, 6.50758412e-01, 6.01791772e-01, 7.28162919e-01, 1.13007200e+00, 8.63541856e-01, 1.15143489e+00, 4.89596844e-02, 1.00364258e-01, 0.00000000e+00, 0.00000000e+00, 6.64963671e-01, 1.39827542e-01, 2.14875035e-01, 2.76581938e-01, 5.15746701e-01 ],
                [ 4.37402971e-01, 1.45516695e-01, 6.34672996e-02, 2.99236860e-01, 2.64880304e-01, 1.92532228e-01, 9.17330152e-03, 2.57424251e-02, 1.77736390e-02, 2.74480435e-01, 4.62707680e-01, 1.45892196e-01, 2.77278371e-02, 3.09871884e-02, 2.69500392e-02, 1.23439873e-02, 6.30478173e-01, 7.84930533e-02, 3.26896359e-02, 0.00000000e+00, 0.00000000e+00, 3.61514731e-02, 1.06663312e+00, 6.97037945e-01, 1.98562243e-02, 1.12308837e-02, 9.68514023e-02, 1.03870978e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.58318284e-01, 7.63663269e-01, 9.32604491e-01, 6.82182560e-01, 2.28992797e-02, 2.84162115e-02, 7.63134127e-02, 0.00000000e+00, 0.00000000e+00 ],
                [ 9.64480737e-01, 6.44803091e-01, 6.73732081e-01, 2.73289746e-01, 2.95196366e-01, 2.97584650e-01, 2.40242361e-02, 1.53600734e-02, 2.86655813e-02, 0.00000000e+00, 6.09730492e-02, 0.00000000e+00, 8.49233613e-03, 1.37130777e-02, 1.40053016e-02, 2.34804769e-02, 2.49250236e-01, 0.00000000e+00, 0.00000000e+00, 4.26396847e-01, 3.83513820e-01, 1.84068228e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.69380205e-01, 3.09437958e-01, 3.22689957e-01, 2.53264828e-01, 1.98658943e-01, 0.00000000e+00, 1.98254798e-02, 0.00000000e+00, 4.70230754e-01, 2.93040652e-01, 1.37719406e-02, 0.00000000e+00, 5.51816296e-02, 8.40572715e-02 ],
                [ 1.85319143e-06, 7.04363492e-02, 2.81583300e-01, 0.00000000e+00, 8.63082780e-04, 2.92667040e-02, 5.55644197e-01, 4.18355319e-01, 5.12368773e-01, 8.15410907e-01, 2.62869819e-01, 2.98839074e-01, 2.00483142e-01, 2.56092040e-01, 2.55736307e-01, 2.35278552e-01, 0.00000000e+00, 8.91600226e-02, 2.31857920e-01, 1.89706737e-01, 3.18118440e-01, 3.18248725e-01, 3.02919723e-01, 1.69460985e-01, 2.89788167e-01, 4.31479543e-02, 2.05620429e-01, 0.00000000e+00, 5.67280446e-03, 0.00000000e+00, 0.00000000e+00, 7.57384077e-02, 1.30911957e-01, 0.00000000e+00, 0.00000000e+00, 6.90480543e-01, 2.09134205e-01, 3.99311658e-01, 2.63510950e-01, 6.44833112e-01 ],
            ]),
            n_syn=4,
            terrain_dir=r"C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\random_terrain",    # ← 추가
            n_terrains=100,      # ← 추가
            use_synergy=True,
            use_symmetry=False,
            init_activations_mean=0.01,
            init_activations_std=0.0,
            fall_penalty=100.0,
            step_size=0.025,
        )
        env = GymCompatibilityWrapper(env)
        env.render_mode = getattr(env, "render_mode", "rgb_array")
        return env
    return _init
# ────────────────────────────────────────────────────────────────



if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Windows 멀티프로세싱 용

    result_dir = r"C:\Users\IlseungPark\Documents\scone_sota\result\uneven"


    train_env = [ make_env() for i in range(0,20) ]
    train_env     = SubprocVecEnv(train_env)
    train_env = VecMonitor(train_env)                              # VecMonitor 적용
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)




    eval_env =[ make_env() for i in range(0,10) ]
    eval_env     = SubprocVecEnv(eval_env)
    eval_env = VecMonitor(eval_env)                               # VecMonitor 적용
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.obs_rms = deepcopy(train_env.obs_rms)   # :contentReference[oaicite:1]{index=1}
    eval_env.ret_rms = deepcopy(train_env.ret_rms)
    # ────────────────────────────────────────────────────────────────

    # ─── 7) 콜백 정의 ─────────────────────────────────────────────
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(result_dir, "checkpoints"),
        name_prefix="sac_model",
        save_vecnormalize=True,
        save_replay_buffer=False
    )
    uneven_cb = UnevenCallback(
        train_env=train_env,                     # VecNormalize된 학습 환경
        eval_episodes=10,                        # 한번에 돌릴 평가 에피소드 수
        eval_freq=10_000,                        # 몇 step마다 평가할지
        save_path=os.path.join(result_dir, "mean_median_max"),
        verbose=1
    )
    callback = CallbackList([checkpoint_callback, uneven_cb])
    # ────────────────────────────────────────────────────────────────

    # ─── 8) SAC 모델 초기화 ───────────────────────────────────────
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[512, 256], qf=[512, 512, 256])
    )
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(0.001),
        buffer_size=int(3e6),
        learning_starts=10000,
        batch_size=256,
        tau=0.02,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=1,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(result_dir, "tensorboard_log"),
        verbose=1,
        device="cuda",
        use_sde=True,
        sde_sample_freq=1,
    )
    # ────────────────────────────────────────────────────────────────

    # ─── 9) 학습 수행 및 저장 ────────────────────────────────────
    model.learn(
        total_timesteps=75_000_000,
        callback=callback,
        log_interval=500,
        progress_bar=True
    )
    model.save(os.path.join(result_dir, "final_sac_model_syn_cl1"))
    model.save_replay_buffer(os.path.join(result_dir, "final_sac_model_replay_buffer_syn_cl1.pkl"))

    # VecNormalize 파라미터 저장
    train_env.save(os.path.join(result_dir,
                           "final_sac_model_vecnormalize_syn_cl1.pkl"))
    train_env.close()
    eval_env.close()

import os
import sys
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
from copy import deepcopy
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrapper.H2190Wrapper import H2190Wrapper
from callback.UnevenCallback import UnevenCallback


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
        env = gym.make('sconewalk_h2190-v1')
        env = H2190Wrapper(
            env,
            syn_matrix=np.array([
                [ 3.74099584e-02, 0.00000000e+00, 0.00000000e+00, 4.42216255e-02, 3.36171127e-02, 1.40870218e-02, 0.00000000e+00, 1.39093748e-01, 4.83707092e-02, 2.28104004e-01, 2.25385014e-01, 2.63395105e-01, 9.82596742e-02, 1.19688689e-01, 9.77016607e-02, 3.91719676e-02, 1.18595703e-01, 8.41267689e-02, 1.18270061e-01, 0.00000000e+00, 2.61404131e-02, 1.15268975e-01, 3.82132319e-01, 2.58458749e-01, 1.15578987e-01, 2.38533877e-02, 1.99277969e-02, 2.24852042e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.12814725e-02, 0.00000000e+00, 2.71834355e-03, 1.89127994e-03, 1.52693129e-02, 4.42418970e-02, 7.42203545e-02, 3.71149331e-03, 0.00000000e+00 ],
                [ 4.63385194e-01, 2.56124286e-01, 2.17259390e-01, 1.67314708e-01, 1.59923485e-01, 1.26128982e-01, 2.29499584e-02, 4.25092163e-03, 7.21496727e-03, 9.92993834e-02, 1.04030827e-01, 0.00000000e+00, 6.52798645e-03, 5.33045426e-03, 5.13060453e-03, 6.23182378e-03, 2.93958231e-01, 0.00000000e+00, 5.71437763e-03, 5.50420288e-02, 3.60006321e-03, 0.00000000e+00, 2.36762812e-01, 4.93257195e-02, 0.00000000e+00, 9.23488844e-03, 7.71109965e-02, 3.45144602e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.57827989e-02, 3.00785340e-01, 1.22512013e-01, 7.59393273e-01, 1.31108743e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00 ],
                [ 9.13264314e-01, 6.17135350e-01, 6.62181137e-01, 2.63994920e-01, 2.89302999e-01, 3.01285815e-01, 0.00000000e+00, 0.00000000e+00, 1.06859874e-03, 0.00000000e+00, 1.11416380e-01, 0.00000000e+00, 1.45016220e-03, 2.33227423e-03, 0.00000000e+00, 5.08963735e-03, 2.25787218e-01, 5.53027548e-03, 0.00000000e+00, 4.56344159e-01, 4.31685349e-01, 2.63346580e-03, 3.99393798e-03, 0.00000000e+00, 0.00000000e+00, 3.24764565e-04, 2.99916518e-01, 3.03609911e-01, 4.16493470e-01, 3.12714026e-01, 2.68971559e-01, 0.00000000e+00, 0.00000000e+00, 2.71228744e-02, 2.08335589e-01, 2.61414688e-01, 2.15326005e-02, 1.10586994e-02, 6.73497314e-02, 1.07464689e-01 ],
                [ 4.56952086e-03, 2.36199907e-02, 1.89430037e-01, 6.57283278e-03, 1.30963487e-02, 2.73316113e-02, 3.20206614e-01, 1.88932672e-03, 7.55548300e-02, 4.83136857e-01, 4.00555046e-01, 1.11383231e-02, 1.84553119e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.50787090e-01, 0.00000000e+00, 4.64041433e-02, 1.13215984e-02, 0.00000000e+00, 0.00000000e+00, 4.96512055e-01, 3.60497574e-01, 3.84766393e-01, 2.31560388e-01, 3.38028273e-01, 0.00000000e+00, 2.77269548e-01, 0.00000000e+00, 3.23331719e-01, 7.08080048e-02, 2.18986069e-01, 0.00000000e+00, 0.00000000e+00, 5.36730532e-01, 1.23035681e-01, 2.43862139e-01, 1.90545506e-01, 4.75981656e-01 ],
                [ 2.81003898e-01, 7.02947311e-02, 1.75177813e-02, 2.45691057e-01, 2.19564031e-01, 1.71230389e-01, 7.25101424e-03, 0.00000000e+00, 1.44123232e-02, 1.60845325e-01, 3.98392574e-01, 1.04681851e-01, 0.00000000e+00, 0.00000000e+00, 2.55291283e-03, 6.04627099e-03, 4.98386713e-01, 7.00704082e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.56240338e-02, 9.47050974e-01, 7.21131134e-01, 0.00000000e+00, 0.00000000e+00, 9.70269229e-02, 9.24639897e-01, 0.00000000e+00, 4.64535963e-03, 0.00000000e+00, 2.89369804e-01, 6.48150292e-01, 1.02433411e+00, 2.22435564e-01, 0.00000000e+00, 2.76083864e-02, 8.61404208e-02, 0.00000000e+00, 0.00000000e+00 ],
                [ 6.61029484e-02, 1.39599164e-01, 3.47076956e-01, 0.00000000e+00, 0.00000000e+00, 3.46364544e-02, 6.34788653e-01, 4.84490570e-01, 6.44429771e-01, 6.34410287e-01, 0.00000000e+00, 1.91108369e-01, 2.04808404e-01, 2.83668938e-01, 3.09795178e-01, 3.26824150e-01, 0.00000000e+00, 7.30314781e-02, 1.92321670e-01, 2.73110729e-01, 4.47863999e-01, 3.93482849e-01, 0.00000000e+00, 0.00000000e+00, 1.77699636e-01, 0.00000000e+00, 1.60998476e-01, 2.90741661e-02, 1.72073802e-02, 6.33727255e-02, 0.00000000e+00, 7.29365975e-02, 1.29691261e-01, 0.00000000e+00, 0.00000000e+00, 7.22474687e-01, 1.93278717e-01, 3.62613745e-01, 2.79832332e-01, 6.59238075e-01 ],
                [ 0.00000000e+00, 0.00000000e+00, 7.12598132e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.84221952e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.39547299e-02, 0.00000000e+00, 2.53078747e-03, 7.81731076e-03, 1.06847119e-02, 2.10479338e-02, 9.59385472e-01, 1.32506189e-01, 0.00000000e+00, 9.52521073e-03, 2.12860678e-02, 1.13308633e-02, 7.14965712e-01, 7.72404162e-01, 6.52804750e-01, 5.89318931e-01, 4.66655054e-01, 8.55168980e-01, 1.09279515e+00, 1.04184337e+00, 1.07961625e+00, 1.20714502e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.20890129e-01, 1.01237997e-01, 1.23535879e-01, 2.34235453e-01, 3.75788318e-01 ],
            ]),
            n_syn=7,
            # terrain_dir=r"C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\random_terrain",    # ← 추가
            # n_terrains=100,      # ← 추가
            use_synergy=True,
            use_symmetry=False,
            init_activations_mean=0.01,
            init_activations_std=0.0,
            fall_penalty=250.0,
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

    result_dir = r"C:\Users\ok\Documents\GitHub\sconegym\result_data\Level\syn7"


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

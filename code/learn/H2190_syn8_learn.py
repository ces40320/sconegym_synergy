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
                [ 4.57272585e-01, 2.53225373e-01, 2.14345299e-01, 1.65433898e-01, 1.58645095e-01, 1.25152858e-01, 2.40008403e-02, 2.64847070e-03, 1.00424897e-02, 8.42521949e-02, 1.02397236e-01, 0.00000000e+00, 5.48604623e-03, 5.62554533e-03, 6.98207637e-03, 8.33807175e-03, 2.91799073e-01, 0.00000000e+00, 3.83875498e-03, 5.40856983e-02, 3.18483440e-03, 1.95856777e-03, 2.21905555e-01, 4.62071730e-02, 0.00000000e+00, 7.11655970e-03, 7.82019949e-02, 3.35704877e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.43810287e-02, 2.90101659e-01, 1.18870809e-01, 7.50937004e-01, 1.28707445e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00 ],
                [ 2.85011491e-01, 7.39019688e-02, 2.28275744e-02, 2.44637650e-01, 2.17896071e-01, 1.70076260e-01, 0.00000000e+00, 0.00000000e+00, 3.51721993e-03, 1.72064785e-01, 3.84549018e-01, 1.09757754e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.86733176e-01, 7.13700316e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.24425801e-02, 9.39669422e-01, 7.02413541e-01, 0.00000000e+00, 0.00000000e+00, 8.79283853e-02, 9.41652632e-01, 0.00000000e+00, 1.07790828e-03, 0.00000000e+00, 2.86069292e-01, 6.48042204e-01, 1.01731424e+00, 2.26454666e-01, 0.00000000e+00, 3.00318664e-02, 8.90624371e-02, 0.00000000e+00, 0.00000000e+00 ],
                [ 9.07999302e-01, 6.15999929e-01, 6.61833443e-01, 2.59697975e-01, 2.84921107e-01, 2.98334733e-01, 2.76904745e-03, 1.79086398e-03, 6.14022386e-03, 0.00000000e+00, 9.99784364e-02, 0.00000000e+00, 3.12528475e-03, 2.93720708e-03, 0.00000000e+00, 8.68037742e-03, 2.18738194e-01, 3.69579315e-03, 0.00000000e+00, 4.61364657e-01, 4.40462539e-01, 2.30319446e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.39353250e-03, 3.00190843e-01, 2.81082765e-01, 4.27634691e-01, 3.07288910e-01, 2.86757137e-01, 0.00000000e+00, 0.00000000e+00, 2.41186179e-02, 1.92417069e-01, 2.52652975e-01, 1.90664463e-02, 6.79950752e-03, 6.47945113e-02, 1.03025045e-01 ],
                [ 0.00000000e+00, 0.00000000e+00, 4.04808760e-02, 7.28541835e-03, 1.06394760e-02, 5.81807412e-04, 2.37285524e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.26228983e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.37042804e-02, 4.95508977e-01, 0.00000000e+00, 2.83739657e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.02972696e-01, 6.77199557e-01, 4.01780562e-01, 4.63072670e-01, 4.19475249e-01, 0.00000000e+00, 8.19366446e-01, 2.42681261e-01, 9.26211686e-01, 4.70741297e-02, 6.64759439e-02, 0.00000000e+00, 9.37237398e-03, 1.87173275e-02, 9.67691818e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00 ],
                [ 4.67294432e-02, 0.00000000e+00, 0.00000000e+00, 4.63655408e-02, 3.78557746e-02, 2.03120603e-02, 0.00000000e+00, 9.25654020e-02, 7.64383960e-03, 2.28920548e-01, 2.56499927e-01, 1.98409341e-01, 6.66573427e-02, 6.80403079e-02, 4.61356515e-02, 7.31715694e-03, 1.15209504e-01, 6.15269801e-02, 9.66663304e-02, 0.00000000e+00, 0.00000000e+00, 4.64466780e-02, 3.85383526e-01, 2.69826113e-01, 1.17108794e-01, 4.11403250e-02, 4.43918722e-02, 1.33578145e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.01153337e-02, 0.00000000e+00, 0.00000000e+00, 2.59358501e-03, 1.30766728e-02, 3.26755122e-02, 5.70175880e-02, 0.00000000e+00, 0.00000000e+00 ],
                [ 6.97206579e-03, 4.51147601e-02, 8.27000490e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.92504515e-01, 5.36320363e-01, 5.65478514e-01, 2.01601408e-01, 0.00000000e+00, 2.79593399e-01, 2.41588097e-01, 3.62430400e-01, 3.97394043e-01, 3.89775403e-01, 0.00000000e+00, 8.48805367e-02, 2.03646851e-01, 1.96419013e-01, 4.17626620e-01, 5.17308579e-01, 0.00000000e+00, 4.93535442e-02, 2.01180535e-02, 0.00000000e+00, 4.75282468e-03, 0.00000000e+00, 2.94497417e-02, 0.00000000e+00, 2.23048523e-03, 4.06408522e-02, 0.00000000e+00, 6.73355416e-03, 0.00000000e+00, 0.00000000e+00, 6.06885964e-02, 1.11256665e-01, 0.00000000e+00, 7.83286756e-03 ],
                [ 0.00000000e+00, 0.00000000e+00, 2.77693814e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.11029537e-02, 3.59066230e-02, 5.17909754e-02, 4.41273833e-02, 2.28235971e-02, 6.17483620e-01, 1.64564781e-01, 2.61880209e-02, 0.00000000e+00, 1.81427703e-02, 5.03416072e-02, 4.73959038e-01, 4.02644809e-01, 4.13563862e-01, 2.53156805e-01, 1.69008055e-01, 1.03511263e+00, 4.47666014e-01, 8.38238970e-01, 3.49869517e-01, 0.00000000e+00, 0.00000000e+00, 2.18240024e-02, 0.00000000e+00, 5.00582949e-01, 1.18721873e-01, 1.58729067e-01, 2.33250872e-01, 3.70885585e-01 ],
                [ 9.62053824e-03, 4.92585282e-02, 2.13322937e-01, 0.00000000e+00, 0.00000000e+00, 2.42472803e-02, 2.79530205e-01, 3.44768377e-02, 1.41085846e-01, 4.64909759e-01, 9.84930543e-02, 0.00000000e+00, 6.18758268e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.43269756e-02, 5.03072957e-02, 4.22505371e-02, 0.00000000e+00, 1.06300056e-01, 6.37270342e-03, 1.91212315e-01, 3.31592676e-02, 1.61066814e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.65048766e-02, 1.70175985e-01, 0.00000000e+00, 0.00000000e+00, 6.21627772e-01, 1.31942804e-01, 2.54471339e-01, 2.33689894e-01, 5.59883056e-01 ],
            ]),
            n_syn=8,
            # terrain_dir=r"C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\random_terrain",    # ← 추가
            # n_terrains=100,      # ← 추가
            use_synergy=True,
            use_symmetry=True,
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

    result_dir = r"C:\Users\ok\Documents\GitHub\sconegym\result_data\Level\syn8_plateau_1"


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

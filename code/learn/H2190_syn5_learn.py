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
                [ 0.00000000e+00, 0.00000000e+00, 1.44001643e-01, 0.00000000e+00, 0.00000000e+00, 4.53194324e-04, 2.78468394e-01, 0.00000000e+00, 0.00000000e+00, 1.52037120e-01, 3.32442492e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.28583095e-01, 1.03195440e-01, 3.78143978e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.51241211e-01, 9.23524972e-01, 7.92172011e-01, 6.61712066e-01, 6.05755570e-01, 7.08661998e-01, 1.12849666e+00, 8.58699736e-01, 1.15845040e+00, 4.57447236e-02, 1.16515383e-01, 0.00000000e+00, 0.00000000e+00, 6.83538369e-01, 1.39058356e-01, 2.10926700e-01, 2.81675158e-01, 5.29362657e-01 ],
                [ 5.00639139e-01, 2.74884559e-01, 2.34145286e-01, 1.88432483e-01, 1.80673032e-01, 1.45157041e-01, 3.18583728e-02, 0.00000000e+00, 8.17222440e-03, 1.11817691e-01, 1.27018975e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.27099980e-04, 3.25535169e-01, 0.00000000e+00, 0.00000000e+00, 5.44127764e-02, 3.90665559e-03, 0.00000000e+00, 2.88377770e-01, 8.27321988e-02, 0.00000000e+00, 7.55371228e-03, 9.45877740e-02, 4.03037155e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.18867283e-02, 3.84794203e-01, 2.16910651e-01, 8.09372306e-01, 1.40779579e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00 ],
                [ 9.10980069e-01, 6.19606718e-01, 6.67017548e-01, 2.59996881e-01, 2.84546216e-01, 2.97374694e-01, 1.73320416e-02, 2.07760272e-02, 3.00362170e-02, 0.00000000e+00, 8.37994376e-02, 2.34835213e-03, 1.31167016e-02, 2.05210752e-02, 2.01446457e-02, 2.80453930e-02, 2.43220691e-01, 1.25395305e-02, 3.45153381e-04, 4.67886652e-01, 4.54346396e-01, 2.81854791e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.83491615e-03, 2.96617203e-01, 3.27798101e-01, 4.31148838e-01, 3.34094057e-01, 2.82875893e-01, 0.00000000e+00, 0.00000000e+00, 3.37228520e-03, 2.12372279e-01, 2.79255149e-01, 2.61953674e-02, 1.73150044e-02, 7.62940444e-02, 1.21830044e-01 ],
                [ 0.00000000e+00, 6.57945180e-02, 2.71290964e-01, 0.00000000e+00, 0.00000000e+00, 1.99409219e-02, 5.60763481e-01, 4.13783632e-01, 5.11297809e-01, 8.16156326e-01, 2.50055814e-01, 2.83836031e-01, 1.96290430e-01, 2.50011853e-01, 2.50266067e-01, 2.31779366e-01, 0.00000000e+00, 8.36288729e-02, 2.29915502e-01, 1.64320131e-01, 2.86224051e-01, 3.09704020e-01, 3.00260013e-01, 1.60634647e-01, 3.03314078e-01, 5.49891013e-02, 2.00693944e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.91559854e-02, 1.42497803e-01, 0.00000000e+00, 0.00000000e+00, 7.00216168e-01, 2.05690018e-01, 3.92514550e-01, 2.61944394e-01, 6.41827701e-01 ],
                [ 2.51509644e-01, 4.57845635e-02, 0.00000000e+00, 2.41470269e-01, 2.11640351e-01, 1.58228993e-01, 0.00000000e+00, 3.73940898e-02, 1.75295031e-02, 2.21769350e-01, 4.64660394e-01, 2.09236800e-01, 3.83285512e-02, 4.56068515e-02, 3.97127368e-02, 1.90924627e-02, 5.19204007e-01, 1.04121247e-01, 4.27174561e-02, 0.00000000e+00, 0.00000000e+00, 5.58282915e-02, 1.04724224e+00, 8.01530540e-01, 4.02537248e-02, 1.60942063e-02, 9.08699688e-02, 9.58636804e-01, 1.13939807e-02, 1.60930740e-02, 5.36016044e-03, 2.68771549e-01, 5.49127863e-01, 9.35260849e-01, 1.43014288e-01, 0.00000000e+00, 4.21745107e-02, 1.09392622e-01, 0.00000000e+00, 0.00000000e+00 ],
            ]),
            n_syn=5,
            # terrain_dir=r"C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\random_terrain",    # ← 추가
            # n_terrains=100,      # ← 추가
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

    result_dir = r"C:\Users\ok\Documents\GitHub\sconegym\result_data\Level\syn5"


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

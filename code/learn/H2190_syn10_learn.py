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
                [ 1.56561168e-02, 0.00000000e+00, 4.26913886e-02, 2.46841484e-01, 2.27937310e-01, 1.75395638e-01, 5.78351435e-02, 0.00000000e+00, 4.05230310e-02, 0.00000000e+00, 7.54003559e-01, 2.57360767e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.41227063e-01, 1.50132763e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.93366629e-01, 6.89313702e-01, 4.93152884e-02, 3.18612930e-02, 2.52184189e-01, 2.74770811e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.35528309e-03, 0.00000000e+00, 1.94940398e-01, 1.51926219e-01, 0.00000000e+00, 1.78266992e-02, 6.26586534e-02, 0.00000000e+00, 0.00000000e+00 ],
                [ 2.61672870e-01, 6.97019208e-02, 4.82499561e-03, 1.50188848e-01, 1.30945927e-01, 1.02880532e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.37611995e-01, 1.07571183e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.54201535e-01, 1.10155523e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.53145217e-03, 7.71182213e-01, 4.46042611e-01, 0.00000000e+00, 0.00000000e+00, 5.80929323e-03, 8.29595188e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.79369195e-01, 6.64896169e-01, 9.58344674e-01, 1.79736350e-01, 0.00000000e+00, 1.88000364e-02, 5.51259297e-02, 0.00000000e+00, 0.00000000e+00 ],
                [ 9.05363034e-01, 6.16702166e-01, 6.59519521e-01, 2.55002159e-01, 2.80738559e-01, 2.95426244e-01, 1.48733433e-02, 1.17065485e-02, 1.92746047e-02, 0.00000000e+00, 7.88029955e-02, 0.00000000e+00, 5.80284118e-03, 8.50344196e-03, 6.49405364e-03, 1.61249717e-02, 2.11973245e-01, 4.40358527e-03, 0.00000000e+00, 4.68718127e-01, 4.55490405e-01, 1.12002809e-02, 4.89641483e-04, 0.00000000e+00, 0.00000000e+00, 5.35832993e-04, 2.89529295e-01, 2.94627130e-01, 4.37814342e-01, 3.20797594e-01, 2.93361641e-01, 1.94757320e-03, 0.00000000e+00, 3.32726861e-02, 1.68277049e-01, 2.43851223e-01, 1.91344942e-02, 4.58696677e-03, 6.43601993e-02, 9.92460739e-02 ],
                [ 0.00000000e+00, 0.00000000e+00, 5.18953757e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.97497847e-01, 0.00000000e+00, 0.00000000e+00, 4.11141025e-02, 2.27831449e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.05369814e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.42408777e-01, 5.83433813e-01, 3.61749640e-01, 4.25339989e-01, 3.73859419e-01, 0.00000000e+00, 8.47287951e-01, 2.18703707e-01, 9.48701220e-01, 7.30000246e-02, 1.09191173e-01, 0.00000000e+00, 4.47008681e-03, 3.01949190e-02, 9.98193701e-03, 1.49356653e-02, 0.00000000e+00, 0.00000000e+00 ],
                [ 0.00000000e+00, 3.20797589e-02, 1.90945539e-01, 0.00000000e+00, 0.00000000e+00, 1.59714815e-02, 1.11681969e-01, 0.00000000e+00, 4.33346919e-02, 3.02385197e-01, 1.65411623e-01, 0.00000000e+00, 2.65734792e-02, 1.43439973e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.76559568e-03, 1.75253133e-02, 3.51206052e-02, 1.92840820e-02, 0.00000000e+00, 9.21623184e-02, 2.45345858e-02, 2.44552712e-01, 6.21870375e-02, 1.90328116e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.02689112e-03, 1.05551653e-01, 0.00000000e+00, 0.00000000e+00, 6.21940248e-01, 1.30187506e-01, 2.48738479e-01, 2.37149392e-01, 5.48554811e-01 ],
                [ 0.00000000e+00, 0.00000000e+00, 4.14236563e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.76847297e-01, 2.16603803e-01, 0.00000000e+00, 0.00000000e+00, 3.97755221e-01, 2.52057439e-01, 3.83276449e-01, 3.69984117e-01, 2.34589950e-01, 0.00000000e+00, 1.18526310e-01, 1.45340559e-01, 1.02082191e-01, 2.55740647e-01, 4.57979204e-01, 2.16043177e-03, 4.72992084e-02, 1.11341575e-01, 0.00000000e+00, 1.67984246e-02, 1.13939257e-01, 4.68194334e-02, 0.00000000e+00, 5.15442305e-03, 0.00000000e+00, 0.00000000e+00, 1.75338393e-02, 5.51890263e-03, 6.93122797e-02, 8.46568877e-02, 1.56269276e-01, 3.38434327e-02, 4.71675808e-02 ],
                [ 0.00000000e+00, 0.00000000e+00, 6.14313887e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.39293346e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.80409462e-04, 7.17701656e-01, 1.41630961e-01, 1.39609110e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.43342478e-01, 4.75434733e-01, 4.13024627e-01, 2.91439278e-01, 1.89551084e-01, 1.02469844e+00, 4.46648846e-01, 8.90486578e-01, 3.64507325e-01, 0.00000000e+00, 0.00000000e+00, 1.02920773e-02, 0.00000000e+00, 4.38583310e-01, 8.91336091e-02, 1.05300199e-01, 2.08216738e-01, 3.25404543e-01 ],
                [ 5.04370614e-02, 1.00006072e-01, 2.01088998e-01, 0.00000000e+00, 0.00000000e+00, 2.54152892e-02, 8.35022720e-01, 4.26689020e-01, 6.49978502e-01, 6.54697314e-01, 0.00000000e+00, 0.00000000e+00, 3.20091615e-02, 4.69329722e-02, 1.11421544e-01, 2.26409933e-01, 0.00000000e+00, 0.00000000e+00, 1.49680383e-01, 1.67002493e-01, 2.71279145e-01, 1.61178628e-01, 8.69409004e-02, 3.27359155e-02, 6.64480139e-03, 0.00000000e+00, 6.62124443e-02, 0.00000000e+00, 0.00000000e+00, 3.08257230e-02, 0.00000000e+00, 1.20329895e-01, 1.69615989e-01, 0.00000000e+00, 0.00000000e+00, 2.97537391e-01, 7.44035124e-02, 1.44937288e-01, 9.89528063e-02, 2.97986133e-01 ],
                [ 4.32209826e-01, 2.46257875e-01, 2.17663739e-01, 1.54089671e-01, 1.49296617e-01, 1.18347178e-01, 9.13515630e-03, 0.00000000e+00, 0.00000000e+00, 5.63449254e-02, 1.02829185e-01, 0.00000000e+00, 6.83065112e-03, 8.35753119e-03, 8.70772536e-03, 5.41393796e-03, 2.67041449e-01, 0.00000000e+00, 0.00000000e+00, 5.31797735e-02, 0.00000000e+00, 5.75829650e-03, 1.35935093e-01, 1.89417125e-03, 0.00000000e+00, 5.50215767e-03, 8.51715731e-02, 2.45331328e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.14995869e-01, 2.06304986e-02, 7.30828077e-01, 1.34918560e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00 ],
                [ 7.33342044e-02, 1.34108845e-02, 0.00000000e+00, 3.38744292e-02, 2.63102103e-02, 1.23025431e-02, 0.00000000e+00, 7.25724894e-02, 7.32190442e-03, 3.10928029e-01, 1.74798431e-01, 1.22121104e-01, 2.96171170e-02, 1.64403060e-02, 3.01806559e-03, 0.00000000e+00, 8.27097898e-02, 3.28890023e-02, 8.33772886e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.91853802e-01, 2.14840364e-01, 6.22444857e-02, 2.32199631e-02, 3.25403693e-03, 1.23121788e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.83347965e-02, 1.00282206e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.07900680e-02, 3.26042751e-02, 0.00000000e+00, 0.00000000e+00 ],
            ]),
            n_syn=10,
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

    result_dir = r"C:\Users\ok\Documents\GitHub\sconegym\result_data\Level\syn10_plateau"


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

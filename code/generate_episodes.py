import gym
import sconegym
import random
import numpy as np
from stable_baselines3 import SAC
from gym.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# from wrapper.H2190Wrapper import H2190Wrapper
from wrapper.H2190TorsoVarWrapper import H2190Wrapper
from utils.synergy_results_arr  import get_NMF_matrix


# Domain Randomization을 위한 환경 ID 리스트
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
    
    
def make_env():
    def _init():
        # env = gym.make('sconewalk_h2190-v1')
        env = gym.make('sconewalk_h2190_uneven6-v1')
        n_syn = 8  # NMF로 생성된 시너지 차원 수
        env = H2190Wrapper(
            env,
            n_syn = n_syn,
            syn_matrix = get_NMF_matrix(n_syn),
            # terrain_dir=r"C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\random_terrain",    # ← 추가
            # n_terrains=100,      # ← 추가
            use_synergy=True,
            use_symmetry=True,
            init_activations_mean=0.01,
            init_activations_std=0.0,
            fall_penalty=100.0,
            step_size=0.025,
        )
        env = GymCompatibilityWrapper(env)
        env.render_mode = getattr(env, "render_mode", "rgb_array")
        return env
    return _init
# ───────────────
if __name__ == '__main__':
    eval_env = DummyVecEnv([make_env()])


    vecnormalize_path = r"C:\Users\ok\Documents\GitHub\sconegym\result_data\Level\syn8_plateau_1\mean_median_max\best_max_vecnormalize.pkl"

    eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    # 저장된 SAC 모델 policy 불러오기 (파일 경로에 맞게 수정)
    model_path = r"C:\Users\ok\Documents\GitHub\sconegym\result_data\Level\syn8_plateau_1\mean_median_max\best_max.zip"
    model = SAC.load(model_path, env=eval_env)


    num_episodes = 10
    max_steps = 1000

    # Scone 환경은 에피소드 저장을 위해 사전에 호출해줘야 합니다.
    eval_env.envs[0].store_next_episode()
    eval_env.reset()

    for ep in range(num_episodes):
        obs = eval_env.reset()  # 에피소드 시작
        ep_reward = 0.0
        ep_steps = 0
        done = [False]
        while not done[0]:
            # 결정적 정책을 사용하여 액션 선택
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward[0]
            ep_steps += 1

        # 에피소드가 자연 종료되지 않더라도 반드시 에피소드를 저장합니다.
        eval_env.envs[0].write_now()
        eval_env.envs[0].store_next_episode()

        # Scone 환경에 따라 Center of Mass 값 등 추가 정보를 호출할 수 있습니다.
        com_pos = None
        if hasattr(eval_env.envs[0], 'model') and hasattr(eval_env.envs[0].model, 'com_pos'):
            com_pos = eval_env.envs[0].model.com_pos()

        print(f"Episode {ep} ended; steps = {ep_steps}; total reward = {ep_reward:.3f}; com = {com_pos}")

    eval_env.close()

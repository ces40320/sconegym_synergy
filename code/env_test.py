# test_uneven_wrapper.py
import os
import gym
import numpy as np
import sys

# SCONE native 라이브러리 로딩 (필요 시 경로 수정)
if sys.platform.startswith("win"):
    sys.path.append("C:/Program Files/SCONE/bin")
elif sys.platform.startswith("linux"):
    sys.path.append("/opt/scone/lib")
else:
    sys.path.append("/Applications/SCONE.app/Contents/MacOS/lib")

import sconepy
import sconegym
from wrapper.H2190UnevenWrapper import H2190UnevenWrapper

def main():
    # ─── 0) 기본 Env 등록은 이미 끝났다고 가정 ────────────────────────────────
    base_env_id = "sconewalk_h2190-v1"
    env = gym.make(base_env_id)

    # ─── 1) 래퍼 생성: terrain_dir, n_terrains 추가 ───────────────────────────
    syn_matrix  = np.zeros((6, 40))
    n_syn       = 6

    # 여기에 100개의 random_terrain_*.scone 파일이 모여 있는 폴더 경로
    terrain_dir = r"C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\random_terrain"
    n_terrains  = 100

    env = H2190UnevenWrapper(
        env,
        syn_matrix=syn_matrix,
        n_syn=n_syn,
        terrain_dir=terrain_dir,    # ← 추가
        n_terrains=n_terrains,      # ← 추가
        use_synergy=True,
        use_symmetry=True,
        init_activations_mean=0.01,
        init_activations_std=0.0,
        fall_penalty=100.0,
        step_size=0.025,
    )

    # ─── 2) 테스트 루프: 1 에피소드만 ───────────────────────────────────────
    for ep in range(10):
        env.store_next_episode()

        obs = env.reset()
        print(f"\n=== Episode {ep} Start ===")
        print("Initial obs shape:", obs.shape)

        total_reward = 0.0
        steps = 0
        while True:
            action, reward, done, info = *env.step(env.action_space.sample()), 
            total_reward += reward
            steps += 1
            if done:
                print(f"Episode {ep} done after {steps} steps, "
                      f"total_reward = {total_reward:.3f}, info = {info}")
                break

    env.close()

if __name__ == "__main__":
    main()

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class UnevenCallback(BaseCallback):
    """
    Simple callback: every eval_freq steps, run evaluate_policy on the provided train_env
    for eval_episodes episodes, compute mean/median/max rewards and lengths, log to TensorBoard,
    and save model when mean, median, or max reward improves.
    """
    def __init__(
        self,
        train_env,
        eval_episodes: int = 10,
        eval_freq: int = 10_000,
        save_path: str = ".",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.save_path = save_path

        # best metrics for saving
        self.best_mean = -np.inf
        self.best_median = -np.inf
        self.best_max = -np.inf

    def _on_step(self) -> bool:
        # only run evaluation every eval_freq steps
        if self.n_calls % self.eval_freq != 0:
            return True

        # disable training mode for VecNormalize if present
        if hasattr(self.train_env, 'training'):
            self.train_env.training = False
        if hasattr(self.train_env, 'norm_reward'):
            self.train_env.norm_reward = False

        # run evaluation
        rewards, lengths = evaluate_policy(
            self.model,
            self.train_env,
            n_eval_episodes=self.eval_episodes,
            return_episode_rewards=True,
            deterministic=True,
        )

        # compute statistics
        mean_r = float(np.mean(rewards))
        median_r = float(np.median(rewards))
        max_r = float(np.max(rewards))
        mean_l = float(np.mean(lengths))
        median_l = float(np.median(lengths))
        max_l = float(np.max(lengths))

        # log to TensorBoard
        self.logger.record('eval/mean_reward', mean_r)
        self.logger.record('eval/median_reward', median_r)
        self.logger.record('eval/max_reward', max_r)
        self.logger.record('eval/mean_length', mean_l)
        self.logger.record('eval/median_length', median_l)
        self.logger.record('eval/max_length', max_l)

        if self.verbose:
            print(
                f"Eval @ step={self.n_calls}: "
                f"mean_r={mean_r:.2f}, median_r={median_r:.2f}, max_r={max_r:.2f}, "
                f"mean_l={mean_l:.1f}, median_l={median_l:.1f}, max_l={max_l:.1f}"
            )

        # save when improved
        if mean_r > self.best_mean:
            self.best_mean = mean_r
            path = os.path.join(self.save_path, 'best_mean')
            self.model.save(path)
            if hasattr(self.train_env, 'save'):
                self.train_env.save(path + '_vecnormalize.pkl')
            if self.verbose:
                print(f"Saved new best mean model to {path}")

        if median_r > self.best_median:
            self.best_median = median_r
            path = os.path.join(self.save_path, 'best_median')
            self.model.save(path)
            if hasattr(self.train_env, 'save'):
                self.train_env.save(path + '_vecnormalize.pkl')
            if self.verbose:
                print(f"Saved new best median model to {path}")

        if max_r > self.best_max:
            self.best_max = max_r
            path = os.path.join(self.save_path, 'best_max')
            self.model.save(path)
            if hasattr(self.train_env, 'save'):
                self.train_env.save(path + '_vecnormalize.pkl')
            if self.verbose:
                print(f"Saved new best max model to {path}")

        return True

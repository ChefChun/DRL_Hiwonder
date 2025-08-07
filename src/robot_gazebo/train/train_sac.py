# train_sac.py
import rclpy
import os
import argparse

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

from jetauto_env import JetAutoEnv  # 你的环境定义文件

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--configs", 
        help="Path to configs.json",
        default="configs.json"
    )
    p.add_argument(
        "--total-timesteps", 
        type=int, 
        default=3_000_000,
        help="Total training timesteps"
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=100_000,
        help="Evaluate every N timesteps"
    )
    p.add_argument(
        "--save-dir",
        default="./models/",
        help="Directory to save models and logs"
    )
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    rclpy.init()

    # 1) 创建环境，并用 VecMonitor 跟踪 episode 信息
    def make_env():
        return JetAutoEnv(config_path=args.configs)
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    # 2) 配置 logger，将 TensorBoard 日志存到 save_dir/log
    new_logger = configure(args.save_dir, ["stdout", "tensorboard"])

    # 3) 创建 SAC 模型
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tensorboard"),
        device="auto",
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
    )
    model.set_logger(new_logger)

    # 4) 回调：定期存 checkpoint；定期做 Eval
    #   EvalCallback 会在 eval_env 上评估并把最佳模型复制到 best_model.zip
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, "best_model"),
        log_path=os.path.join(args.save_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path=args.save_dir,
        name_prefix="checkpoint"
    )

    # 5) 开始训练
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="SAC_JetAuto"
    )

    # 6) 最后保存
    model.save(os.path.join(args.save_dir, "final_model"))
    print("Training completed. Models saved to", args.save_dir)

if __name__ == "__main__":
    main()

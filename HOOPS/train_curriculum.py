import os
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from env_curriculum import HoopsCurriculumEnv

# ============================================================
# Helper: create envs
# ============================================================
def make_env(rank, log_dir):
    def _init():
        env = HoopsCurriculumEnv()
        monitor_file = os.path.join(log_dir, f"monitor_{rank}.csv")
        return Monitor(env, filename=monitor_file)
    return _init

# ============================================================
# Curriculum Callback
# ============================================================
class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.current_level = 1
        self.success_buffer = []
        self.success_threshold = 0.8 # 80% success rate to advance
        self.min_episodes = 50 # Minimum episodes to calculate rate

    def _on_step(self) -> bool:
        # Check for episode completions
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for i, done in enumerate(dones):
            if done:
                # Check if it was a success
                is_success = infos[i].get("is_success", False)
                self.success_buffer.append(1 if is_success else 0)
                
        # Keep buffer size manageable (last 100 episodes)
        if len(self.success_buffer) > 100:
            self.success_buffer = self.success_buffer[-100:]

        # Periodically check if we should advance level
        if self.n_calls % self.check_freq == 0:
            if len(self.success_buffer) >= self.min_episodes:
                success_rate = np.mean(self.success_buffer)
                
                if self.verbose > 0:
                    print(f"Step {self.num_timesteps}: Level {self.current_level} Success Rate: {success_rate:.2f}")
                
                # Log to WandB
                wandb.log({"curriculum/success_rate": success_rate, "curriculum/level": self.current_level})

                if success_rate >= self.success_threshold:
                    if self.current_level < 7:
                        self.current_level += 1
                        print(f"*** PROMOTING TO LEVEL {self.current_level} ***")
                        # Update all environments
                        self.training_env.env_method("set_difficulty", self.current_level)
                        # Clear buffer to prove worth in new level
                        self.success_buffer = []
                    else:
                        print("Max level reached!")

        return True

if __name__ == "__main__":

    # ============================================================
    # CONFIGURATION
    # ============================================================
    TRAIN_FROM_SCRATCH = True 
    OLD_MODEL_PATH = "" 
    
    TRAINING_STEPS = 5_000_000
    NUM_ENVS = 8

    LOG_DIR = "HOOPS/tmp/logs_curriculum/"
    os.makedirs(LOG_DIR, exist_ok=True)

    SAVE_FREQ = 100_000 // NUM_ENVS 

    # ============================================================
    # TRAINING SETUP
    # ============================================================
    print("Creating environment...")
    env = SubprocVecEnv([make_env(i, LOG_DIR) for i in range(NUM_ENVS)])

    print("Building PPO model...")
    new_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        gamma=0.998,
        n_steps=3072,
        batch_size=256,
        ent_coef=0.001,
        clip_range=0.2,
    )

    # --------------------------
    # W&B init
    # --------------------------
    run = wandb.init(
        project="PPO-hoops-curriculum",
        config={
            "learning_rate": 3e-4,
            "gamma": 0.998,
            "n_steps": 3072,
            "batch_size": 256,
            "ent_coef": 0.001,
            "clip_range": 0.2,
            "total_timesteps": TRAINING_STEPS,
            "train_from_scratch": TRAIN_FROM_SCRATCH,
            "curriculum": True
        },
        sync_tensorboard=True,
        monitor_gym=True,
    )

    print("Starting training...")
    
    curriculum_callback = CurriculumCallback(check_freq=1000)
    
    wandb_callback = WandbCallback(
        gradient_save_freq=SAVE_FREQ,
        model_save_path=f"HOOPS/models_curriculum/{run.id}",
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=f"HOOPS/models_curriculum/{run.id}",
        name_prefix="ppo_drone_curriculum",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    callback = CallbackList([wandb_callback, checkpoint_callback, curriculum_callback])
    new_model.learn(total_timesteps=TRAINING_STEPS, callback=callback)

    env.close()
    run.finish()

    print("Training completed successfully.")

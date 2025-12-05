import os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from env_hoops import HoopsDroneEnv

# ============================================================
# Helper: create envs
# ============================================================
def make_env(rank, log_dir):
    def _init():
        env = HoopsDroneEnv()
        monitor_file = os.path.join(log_dir, f"monitor_{rank}.csv")
        return Monitor(env, filename=monitor_file)
    return _init


if __name__ == "__main__":

    # ============================================================
    # CONFIGURATION
    # ============================================================
    TRAIN_FROM_SCRATCH = False  # Start fresh for the new task
    
    OLD_MODEL_PATH = "HOOPS\models\ickwdsds\ppo_drone_hoops_5000000_steps.zip" # Not used if scratch
    
    TRAINING_STEPS = 5_000_000
    NUM_ENVS = 8

    LOG_DIR = "HOOPS/tmp/logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    SAVE_FREQ = 100_000 // NUM_ENVS  # ensures ~100k global steps

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

    if not TRAIN_FROM_SCRATCH:
        print(f"Loading OLD model weights from {OLD_MODEL_PATH}...")
        if not os.path.exists(OLD_MODEL_PATH):
            raise FileNotFoundError(f"Old model not found at {OLD_MODEL_PATH}")
        
        # Load the old model
        old_model = PPO.load(OLD_MODEL_PATH, device="cpu")
        
        # Transfer weights
        print("Copying old weights into new model...")
        new_model.policy.load_state_dict(old_model.policy.state_dict())
        print("Weights loaded successfully.")

    # --------------------------
    # W&B init
    # --------------------------
    run = wandb.init(
        project="PPO-hoops",
        config={
            "learning_rate": 3e-4,
            "gamma": 0.998,
            "n_steps": 3072,
            "batch_size": 256,
            "ent_coef": 0.001,
            "clip_range": 0.2,
            "total_timesteps": TRAINING_STEPS,
            "train_from_scratch": TRAIN_FROM_SCRATCH,
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,       # auto-upload gym videos
    )

    print("Starting training...")
    wandb_callback = WandbCallback(
        gradient_save_freq=SAVE_FREQ,
        model_save_path=f"HOOPS/models/{run.id}",
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=f"HOOPS/models/{run.id}",
        name_prefix="ppo_drone_hoops",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    callback = CallbackList([wandb_callback, checkpoint_callback])
    new_model.learn(total_timesteps=TRAINING_STEPS, callback=callback)

    env.close()
    run.finish()

    print("Training completed successfully.")

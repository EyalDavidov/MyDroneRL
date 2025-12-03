import os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from env import DroneEnv


# ============================================================
# Helper: create envs
# ============================================================
def make_env(rank, log_dir):
    def _init():
        env = DroneEnv()
        monitor_file = os.path.join(log_dir, f"monitor_{rank}.csv")
        return Monitor(env, filename=monitor_file)
    return _init


if __name__ == "__main__":

    # ============================================================
    # CONFIGURATION
    # ============================================================
    TRAIN_FROM_SCRATCH = False  # Set to True to train a new model, False to fine-tune
    
    OLD_MODEL_PATH = "models\/1at8if5m\model.zip"
    NEW_MODEL_PATH = "models/ppo_drone_testing"
    
    TRAINING_STEPS = 2_000_000
    NUM_ENVS = 16

    LOG_DIR = "tmp/logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    SAVE_FREQ = 100_000 // NUM_ENVS  # ensures ~100k global steps

    # ============================================================
    # TRAINING SETUP
    # ============================================================
    if not TRAIN_FROM_SCRATCH:
        print(f"Loading OLD model weights from {OLD_MODEL_PATH}...")
        if not os.path.exists(OLD_MODEL_PATH):
            raise FileNotFoundError(f"Old model not found at {OLD_MODEL_PATH}")
        old_model = PPO.load(OLD_MODEL_PATH, device="cpu")
    else:
        print("Training from SCRATCH (no pre-trained weights loaded).")


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
        print("Copying old weights into new model...")
        new_model.policy.load_state_dict(old_model.policy.state_dict())

    # --------------------------
    # W&B init
    # --------------------------
    run = wandb.init(
        project="drone-rl-ppo",
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
        model_save_path=f"models/{run.id}",
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=f"models/{run.id}",
        name_prefix="ppo_drone",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    callback = CallbackList([wandb_callback, checkpoint_callback])
    new_model.learn(total_timesteps=TRAINING_STEPS, callback=callback)

    env.close()
    run.finish()

    print("Training completed successfully.")

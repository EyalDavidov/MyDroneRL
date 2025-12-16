import os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from env_figure8 import DroneEnvFigure8

# ============================================================
# Helper: create envs
# ============================================================
def make_env(rank, log_dir):
    def _init():
        env = DroneEnvFigure8()
        monitor_file = os.path.join(log_dir, f"monitor_{rank}.csv")
        return Monitor(env, filename=monitor_file)
    return _init


if __name__ == "__main__":

    # ============================================================
    # CONFIGURATION
    # ============================================================
    TRAIN_FROM_SCRATCH = True  
    
    OLD_MODEL_PATH = "models/last_run/model.zip" # Placeholder
    NEW_MODEL_PATH = "models/ppo_figure8"
    
    TRAINING_STEPS = 5_000_000
    NUM_ENVS = 16

    LOG_DIR = "tmp/logs/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    SAVE_FREQ = 100_000 // NUM_ENVS  # ensures ~100k global steps

    # ============================================================
    # TRAINING SETUP
    # ============================================================
    if not TRAIN_FROM_SCRATCH and os.path.exists(OLD_MODEL_PATH):
        print(f"Loading OLD model weights from {OLD_MODEL_PATH}...")
        old_model = PPO.load(OLD_MODEL_PATH, device="cpu")
    else:
        print("Training from SCRATCH.")


    print("Creating environment...")
    # Use SubprocVecEnv for parallel execution
    env = SubprocVecEnv([make_env(i, LOG_DIR) for i in range(NUM_ENVS)])


    print("Building PPO model...")
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    
    new_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu", # Change to "cuda" if GPU is available
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=512, # Larger batch for stable gradients
        n_epochs=10,    # More epochs per update to squeeze data
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
    )

    if not TRAIN_FROM_SCRATCH and 'old_model' in locals():
        print("Copying old weights into new model...")
        new_model.policy.load_state_dict(old_model.policy.state_dict())

    # --------------------------
    # W&B init
    # --------------------------
    # Initialize wandb only if api key is present or just skip it if user prefers.
    # Assuming user wants it since it was in original code.
    try:
        run = wandb.init(
            project="drone-rl-figure8",
            config={
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "n_steps": 2048,
                "batch_size": 512,
                "n_epochs": 10,
                "net_arch": "256x256",
                "ent_coef": 0.01,
                "total_timesteps": TRAINING_STEPS,
                "train_from_scratch": TRAIN_FROM_SCRATCH,
            },
            sync_tensorboard=True,
            monitor_gym=True,
        )
        use_wandb = True
    except Exception as e:
        print(f"WandB init failed: {e}. continuing without wandb.")
        use_wandb = False

    print("Starting training...")
    
    callbacks = []
    
    if use_wandb:
        wandb_callback = WandbCallback(
            gradient_save_freq=SAVE_FREQ,
            model_save_path=f"models/{run.id}",
            verbose=1,
        )
        callbacks.append(wandb_callback)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=f"models/checkpoints",
        name_prefix="ppo_figure8",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    new_model.learn(total_timesteps=TRAINING_STEPS, callback=CallbackList(callbacks))

    # Save final model
    new_model.save(NEW_MODEL_PATH)
    env.close()
    
    if use_wandb:
        run.finish()

    print("Training completed successfully.")

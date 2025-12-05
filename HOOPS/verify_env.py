import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'HOOPS'))

try:
    from env_hoops import HoopsDroneEnv
    print("Import successful")
    env = HoopsDroneEnv()
    print("Env instantiated")
    obs, _ = env.reset()
    print("Env reset")
    print(f"Obs shape: {obs.shape}")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Step successful. Reward: {reward}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from env_figure8 import DroneEnvFigure8

env = DroneEnvFigure8()
obs, _ = env.reset()

print("Observation space:", env.observation_space.shape)
print("Initial obs:", obs)

# Visualize stations
x_coords = [st['pos'][0] for st in env.stations]
y_coords = [st['pos'][1] for st in env.stations]

plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, c='blue', label='Stations')

# Draw station lines
for st in env.stations:
    dx = np.cos(st['angle'])
    dy = np.sin(st['angle'])
    
    # Green (Left side if we face st['angle']?? Wait, code says P_left = center - width/2)
    # Actually my logic was:
    # Green: [-w/2, 0] along st_dir.
    # Red: [0, w/2] along st_dir.
    
    # Let's verify direction.
    # st_dir is vector along the gate width.
    # Normal is perpendicular.
    
    # Plot Green part
    p_center = st['pos']
    p_green_start = p_center - np.array([dx, dy]) * (st['width'] / 2)
    p_green_end = p_center
    
    p_red_start = p_center
    p_red_end = p_center + np.array([dx, dy]) * (st['width'] / 2)
    
    plt.plot([p_green_start[0], p_green_end[0]], [p_green_start[1], p_green_end[1]], c='green', linewidth=3)
    plt.plot([p_red_start[0], p_red_end[0]], [p_red_start[1], p_red_end[1]], c='red', linewidth=3)


drone_path_x = []
drone_path_y = []

# Run loop
for _ in range(200):
    action = env.action_space.sample()
    # Random walk: just some random actions
    obs, reward, terminated, truncated, _ = env.step(action)
    
    state = env.state
    drone_path_x.append(state[0])
    drone_path_y.append(state[1])
    
    if terminated or truncated:
        print("Done. Reward:", reward)
        break

plt.plot(drone_path_x, drone_path_y, c='black', label='Drone Path', alpha=0.5)
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.legend()
plt.title("Drone Figure-8 Env Test")
plt.savefig('test_figure8.png')
print("Saved test_figure8.png")

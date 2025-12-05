import numpy as np
from math import sin, cos, atan2, pi, sqrt
from random import randrange, random, uniform
import gymnasium as gym
from gymnasium import spaces

class HoopsCurriculumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # ------- PHYSICS CONSTANTS -------
        self.gravity = 0.08
        self.thruster_mean = 0.04
        self.thruster_amplitude = 0.08
        self.diff_amplitude = 0.005
        self.arm = 25
        self.mass = 1

        # -------- WIND SETTINGS --------
        self.wind_probability = 0.00
        self.wind_strength_min = 0.01
        self.wind_strength_max = 0.08
        self.wind_duration_min = 0.5
        self.wind_duration_max = 2.0

        self.current_wind = np.array([0.0, 0.0], dtype=np.float32)
        self.wind_timer = 0.0

        # -------- CURRICULUM SETTINGS --------
        self.level = 1
        self.max_level = 7
        self.hoops = []
        self.num_hoops = 0
        self.current_hoop_index = 0
        
        # Initial position
        self.start_pos = np.array([500, 500], dtype=np.float32)

        # -------- ACTION SPACE --------
        self.action_space = spaces.Discrete(5)

        # Observations: Same as before
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))

        # -------- INTERNAL STATE --------
        self.state = None
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean

        self.time = 0
        self.time_limit = 30 # Base time

    def set_difficulty(self, level):
        self.level = min(max(1, level), self.max_level)
        # print(f"Difficulty set to Level {self.level}")

    def generate_hoops(self):
        self.hoops = []
        
        # Level 1: 1 Hoop, random position but not too hard
        # Level 2+: More hoops, forming a path
        
        current_x, current_y = self.start_pos
        
        for i in range(self.level):
            # Generate next hoop relative to current position
            # Distance 150-300 pixels
            dist = uniform(150, 300)
            angle = uniform(0, 2*pi)
            
            # Keep within bounds (0-800) roughly
            next_x = current_x + dist * cos(angle)
            next_y = current_y + dist * sin(angle)
            
            # Clamp to screen area with margin
            next_x = np.clip(next_x, 100, 700)
            next_y = np.clip(next_y, 100, 700)
            
            # Orientation: Random
            orient = uniform(-180, 180)
            
            self.hoops.append((next_x, next_y, 30, orient))
            
            current_x, current_y = next_x, next_y

        self.num_hoops = len(self.hoops)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean
        self.current_hoop_index = 0

        # Generate hoops for this episode based on current level
        self.generate_hoops()
        
        # Adjust time limit based on level
        self.time_limit = 20 + (self.level * 5)

        # Start at center
        self.state = np.array([self.start_pos[0], self.start_pos[1], 0, 0, 0, 0], dtype=np.float32)

        self.current_wind[:] = 0
        self.wind_timer = 0
        
        # For potential-based reward
        self.prev_dist_to_target = self._get_dist_to_target()

        return self.get_obs(), {}

    def _get_dist_to_target(self):
        x, y = self.state[0], self.state[1]
        if self.current_hoop_index < self.num_hoops:
            tx, ty, _, _ = self.hoops[self.current_hoop_index]
        else:
            tx, ty = self.start_pos
        return sqrt((x - tx)**2 + (y - ty)**2)

    def maybe_generate_wind(self):
        if self.wind_timer <= 0:
            if random() < self.wind_probability:
                wind_angle = random() * 2 * pi
                wind_strength = np.random.uniform(self.wind_strength_min, self.wind_strength_max)
                self.current_wind = np.array([
                    wind_strength * cos(wind_angle),
                    wind_strength * sin(wind_angle)
                ], dtype=np.float32)
                self.wind_timer = np.random.uniform(self.wind_duration_min, self.wind_duration_max)
        else:
            self.wind_timer -= 1/60.0
        
        if self.wind_timer <= 0:
            self.current_wind[:] = 0

    def apply_action(self, action):
        if action == 1:    # up
            self.Tl += self.thruster_amplitude
            self.Tr += self.thruster_amplitude
        elif action == 2:  # down
            self.Tl -= self.thruster_amplitude
            self.Tr -= self.thruster_amplitude
        elif action == 3:  # left
            self.Tl -= self.diff_amplitude
            self.Tr += self.diff_amplitude
        elif action == 4:  # right
            self.Tr -= self.diff_amplitude
            self.Tl += self.diff_amplitude

        self.Tl = np.clip(self.Tl, -0.05, 0.1)
        self.Tr = np.clip(self.Tr, -0.05, 0.1)

    def step(self, action: int):
        reward = 0
        self.time += 1 / 60.0
        
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean
        self.apply_action(int(action))

        x, y, vx, vy, angle_deg, ang_speed = self.state
        Tl, Tr = self.Tl, self.Tr
        rad = angle_deg * pi / 180.0

        # Wind
        self.maybe_generate_wind()
        wind_ax, wind_ay = self.current_wind

        # Physics
        ax = -(Tl + Tr) * sin(rad) + wind_ax
        ay = self.gravity - (Tl + Tr) * cos(rad) + wind_ay
        ang_acc = self.arm * (Tr - Tl)

        vx += ax
        vy += ay
        ang_speed += ang_acc
        x += vx
        y += vy
        angle_deg += ang_speed

        self.state = np.array([x, y, vx, vy, angle_deg, ang_speed], dtype=np.float32)

        # Logic
        if self.current_hoop_index < self.num_hoops:
            tx, ty, tr, t_orient = self.hoops[self.current_hoop_index]
        else:
            tx, ty = self.start_pos
            tr = 20
            t_orient = 0

        dist = sqrt((x - tx)**2 + (y - ty)**2)
        
        # --- REWARD SHAPING ---
        
        # 1. Progress Reward (Potential-based)
        # Reward for getting closer, penalty for getting further
        # Scale factor 0.1 means 100 pixels closer = +10 reward
        progress = (self.prev_dist_to_target - dist)
        reward += progress * 0.1
        self.prev_dist_to_target = dist

        # 2. Alignment Reward (Continuous)
        # Only if we are somewhat close (e.g. < 200 units) and it's a hoop
        if self.current_hoop_index < self.num_hoops and dist < 200:
            vel_mag = sqrt(vx*vx + vy*vy)
            if vel_mag > 0.1:
                # Normalized velocity vector
                vx_u, vy_u = vx/vel_mag, vy/vel_mag
                # Target orientation vector
                t_rad = t_orient * pi / 180.0
                tx_u, ty_u = cos(t_rad), sin(t_rad)
                
                # Dot product = cosine of angle difference
                # 1.0 = perfect alignment, -1.0 = opposite
                alignment = (vx_u * tx_u + vy_u * ty_u)
                
                # Give small reward for good alignment
                reward += alignment * 0.05

        # Check if passed hoop
        if dist < tr:
            passed = True
            if self.current_hoop_index < self.num_hoops:
                vel_angle = atan2(vy, vx) * 180 / pi
                diff = (vel_angle - t_orient + 180) % 360 - 180
                if abs(diff) > 90:
                    passed = False
            
            if passed:
                reward += 100 # Big reward
                self.current_hoop_index += 1
                self.time_limit += 5
                # Update potential for new target
                self.prev_dist_to_target = self._get_dist_to_target()
            else:
                # Loitering penalty
                reward -= 0.5

        crashed = dist > 1000
        
        # Survival reward (small)
        reward += 0.01 
        
        # Time penalty (small, to encourage speed)
        reward -= 0.01

        terminated = False
        truncated = self.time >= self.time_limit

        if crashed:
            reward -= 100
            terminated = True
        
        if self.current_hoop_index > self.num_hoops:
            reward += 1000
            terminated = True
            return self.get_obs(), float(reward), terminated, truncated, {"is_success": True}

        return self.get_obs(), float(reward), terminated, truncated, {"is_success": False}

    def get_obs(self):
        x, y, vx, vy, angle_deg, ang_speed = self.state
        
        if self.current_hoop_index < self.num_hoops:
            tx, ty, _, t_orient = self.hoops[self.current_hoop_index]
            
            if self.current_hoop_index + 1 < self.num_hoops:
                nx, ny, _, _ = self.hoops[self.current_hoop_index + 1]
            else:
                nx, ny = self.start_pos
        else:
            tx, ty = self.start_pos
            t_orient = 0
            nx, ny = self.start_pos

        angle_to_up = angle_deg * pi / 180.0
        velocity_mag = sqrt(vx*vx + vy*vy)
        vel_angle = ang_speed
        
        dist_to_target = sqrt((tx - x)**2 + (ty - y)**2) / 500
        angle_to_target = np.arctan2(ty - y, tx - x)
        
        angle_target_and_velocity = np.arctan2(ty - y, tx - x) - np.arctan2(vy, vx)

        current_vel_angle = np.arctan2(vy, vx)
        target_orient_rad = t_orient * pi / 180.0
        alignment = current_vel_angle - target_orient_rad

        dist_to_next = sqrt((nx - x)**2 + (ny - y)**2) / 500
        angle_to_next = np.arctan2(ny - y, nx - x)

        obs = np.array([
            angle_to_up,
            velocity_mag,
            vel_angle,
            dist_to_target,
            angle_to_target,
            angle_target_and_velocity,
            alignment,
            dist_to_next,
            angle_to_next
        ], dtype=np.float32)

        return obs

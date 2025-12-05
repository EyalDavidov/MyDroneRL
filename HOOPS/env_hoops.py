import numpy as np
from math import sin, cos, atan2, pi, sqrt
from random import randrange, random
import gymnasium as gym
from gymnasium import spaces

class HoopsDroneEnv(gym.Env):
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
        self.wind_probability = 0.00      # 5% chance per step
        self.wind_strength_min = 0.01
        self.wind_strength_max = 0.08
        self.wind_duration_min = 0.5
        self.wind_duration_max = 2.0

        self.current_wind = np.array([0.0, 0.0], dtype=np.float32)
        self.wind_timer = 0.0

        # -------- HOOPS SETTINGS --------
        # Define 7 hoops: (x, y, radius, orientation_deg)
        # Orientation is the angle the drone must be travelling (roughly) to pass.
        # e.g. 0 = flying East, 90 = flying South (screen coords), etc.
        self.hoops = [
            (200, 400, 30, 0),    # Fly East
            (300, 500, 30, 45),   # Fly South-East
            (500, 500, 30, 0),    # Fly East
            (600, 400, 30, -45),  # Fly North-East
            (500, 300, 30, 180),  # Fly West
            (300, 300, 30, 180),  # Fly West
            (400, 400, 30, 90)    # Fly South (to finish)
        ]
        self.num_hoops = len(self.hoops)
        self.current_hoop_index = 0
        
        # Initial position
        self.start_pos = np.array([400, 400], dtype=np.float32)

        # -------- ACTION SPACE --------
        self.action_space = spaces.Discrete(5)

        # Observations: 
        # 0: angle_to_up
        # 1: velocity_mag
        # 2: ang_velocity
        # 3: dist_to_target
        # 4: angle_to_target
        # 5: angle_target_and_velocity
        # 6: target_hoop_orientation_relative_to_velocity (alignment)
        # 7: dist_to_next_hoop
        # 8: angle_to_next_hoop
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))

        # -------- INTERNAL STATE --------
        self.state = None
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean

        self.time = 0
        self.time_limit = 60 # Give more time for 7 hoops

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean
        self.current_hoop_index = 0

        # x, y, vx, vy, angle_deg, ang_speed
        # Start at 400, 400 (center)
        self.state = np.array([self.start_pos[0], self.start_pos[1], 0, 0, 0, 0], dtype=np.float32)

        # reset wind
        self.current_wind[:] = 0
        self.wind_timer = 0

        return self.get_obs(), {}

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

        self.Tl = np.clip(self.Tl, -0.1, 0.2)
        self.Tr = np.clip(self.Tr, -0.1, 0.2)

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
            # Finished all hoops, target is start pos to finish
            tx, ty = self.start_pos
            tr = 20 # Landing radius
            t_orient = 0 # No specific orientation for landing

        dist = sqrt((x - tx)**2 + (y - ty)**2)
        
        # Check if passed hoop
        if dist < tr:
            # Check orientation alignment if it's a hoop
            passed = True
            if self.current_hoop_index < self.num_hoops:
                # Calculate velocity angle
                vel_angle = atan2(vy, vx) * 180 / pi
                # Normalize difference to [-180, 180]
                diff = (vel_angle - t_orient + 180) % 360 - 180
                if abs(diff) > 90: # Must be within 90 degrees of target direction
                    passed = False
            
            if passed:
                reward += 100 # Big reward for passing hoop
                self.current_hoop_index += 1
                # Bonus time
                self.time_limit += 2
            else:
                # Penalty for being in the hoop 'zone' but with wrong direction/velocity
                # This prevents the agent from loitering inside the hoop to minimize distance penalty
                reward -= 0.5

        crashed = dist > 1000 # Too far away
        
        # Reward
        reward += 1/60 # Survival
        reward -= dist / (100 * 60) # Distance penalty
        reward -= 0.05 # Time penalty

        terminated = False
        truncated = self.time >= self.time_limit

        if crashed:
            reward -= 1000
            terminated = True
        
        if self.current_hoop_index > self.num_hoops:
            # Completed course
            reward += 1000
            terminated = True

        return self.get_obs(), float(reward), terminated, truncated, {}

    def get_obs(self):
        x, y, vx, vy, angle_deg, ang_speed = self.state
        
        if self.current_hoop_index < self.num_hoops:
            tx, ty, _, t_orient = self.hoops[self.current_hoop_index]
            
            # Next next hoop for anticipation
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

        # Alignment with hoop orientation
        # We want the drone's velocity vector to align with t_orient
        # But we also want the drone's position relative to hoop to be aligned? 
        # No, just velocity direction matters for passing through.
        # Let's provide the difference between velocity angle and hoop orientation.
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

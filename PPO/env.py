import numpy as np
from math import sin, cos, atan2, pi, sqrt
from random import randrange, random

import gymnasium as gym
from gymnasium import spaces


class DroneEnv(gym.Env):
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

        # -------- ACTION SPACE --------
        self.action_space = spaces.Discrete(5)

        # 6 observations: angle_to_up, velocity, angle_velocity,
        # distance_to_target, angle_to_target, angle_target_and_velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

        # -------- INTERNAL STATE --------
        self.state = None
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean

        self.time = 0
        self.time_limit = 20

        self.new_target()

    # ----------------------------------------------------
    def new_target(self):
        self.target = np.array([
            randrange(200, 600),
            randrange(200, 600)
        ], dtype=np.float32)

    # ----------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean

        # x, y, vx, vy, angle_deg, ang_speed
        self.state = np.array([400, 400, 0, 0, 0, 0], dtype=np.float32)

        # reset wind
        self.current_wind[:] = 0
        self.wind_timer = 0

        self.new_target()

        return self.get_obs(), {}

    # ----------------------------------------------------
    def maybe_generate_wind(self):
        """Randomly triggers a wind gust."""
        if self.wind_timer <= 0:
            # Try creating a wind event
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

        # When timer finishes → wind stops
        if self.wind_timer <= 0:
            self.current_wind[:] = 0


    # ----------------------------------------------------
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

    # ----------------------------------------------------
    def step(self, action: int):
        reward = 0

        self.time += 1 / 60.0

        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean

        self.apply_action(int(action))

        x, y, vx, vy, angle_deg, ang_speed = self.state
        Tl, Tr = self.Tl, self.Tr

        rad = angle_deg * pi / 180.0

        # ------ WIND CHECK ------
        self.maybe_generate_wind()
        wind_ax, wind_ay = self.current_wind

        # ------- PHYSICS -------
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

        # ------- CONDITIONS -------
        dist = sqrt((x - self.target[0])**2 + (y - self.target[1])**2)
        crashed = dist > 1000
        collected = dist < 50

        # ------- REWARD -------
        reward += 1/60      # survival reward
        reward -= dist / (100 * 60)

        if collected:
            reward += 100
            self.new_target()

        if crashed:
            reward -= 1000

        terminated = crashed
        truncated = self.time >= self.time_limit

        return self.get_obs(), float(reward), terminated, truncated, {}

    # ----------------------------------------------------
    def get_obs(self):
        x, y, vx, vy, angle_deg, ang_speed = self.state
        tx, ty = self.target

        angle_to_up = angle_deg * pi / 180.0
        velocity_mag = sqrt(vx*vx + vy*vy)
        vel_angle = ang_speed
        distance = sqrt((tx - x)**2 + (ty - y)**2) / 500
        angle_to_target = np.arctan2(ty - y, tx - x)

        angle_target_and_velocity = np.arctan2(
            ty - y, tx - x
        ) - np.arctan2(vy, vx)

        obs = np.array([
            angle_to_up,
            velocity_mag,
            vel_angle,
            distance,
            angle_to_target,
            angle_target_and_velocity
        ], dtype=np.float32)

        return obs

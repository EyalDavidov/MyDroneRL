import numpy as np
from math import sin, cos, pi, sqrt
from random import randrange

class HumanDroneEnv:
    def __init__(self):
        # ORIGINAL CONSTANTS
        self.gravity = 0.08
        self.thruster_mean = 0.04
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.003
        self.arm = 25
        self.mass = 1

        # initial thrust
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean

        # state = x, y, vx, vy, angle(deg), angular_speed
        self.state = np.array([400, 400, 0, 0, 0, 0], dtype=np.float32)

        # target
        self.new_target()

    def new_target(self):
        self.target = np.array([
            randrange(200, 600),
            randrange(200, 600)
        ], dtype=np.float32)

    def reset(self):
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean
        self.state = np.array([400, 400, 0, 0, 0, 0], dtype=np.float32)
        self.new_target()
        return self.get_obs()

    def apply_human_control(self, up, down, left, right):
        if up:
            self.Tl += self.thruster_amplitude
            self.Tr += self.thruster_amplitude

        if down:
            self.Tl -= self.thruster_amplitude
            self.Tr -= self.thruster_amplitude

        if left:
            self.Tl -= self.diff_amplitude
        if right:
            self.Tr -= self.diff_amplitude

        # clamp
        self.Tl = np.clip(self.Tl, -0.003, 0.083)
        self.Tr = np.clip(self.Tr, -0.003, 0.083)

    def step(self):
        x, y, vx, vy, angle, ang_speed = self.state
        Tl, Tr = self.Tl, self.Tr

        print(f"Thrusters: Tl={Tl:.3f}, Tr={Tr:.3f}")

        rad = angle * pi / 180

        # EXACT ORIGINAL ACCELERATIONS
        ax = -(Tl + Tr) * sin(rad) 
        ay = self.gravity -(Tl + Tr) * cos(rad) 
        ang_acc = self.arm * (Tr - Tl)

        # update velocities (NO dt)
        vx += ax
        vy += ay
        ang_speed += ang_acc

        # update positions
        x += vx
        y += vy
        angle += ang_speed

        self.state = np.array([x, y, vx, vy, angle, ang_speed], dtype=np.float32)

        dist = sqrt((x - self.target[0])**2 + (y - self.target[1])**2)

        collected = dist < 50
        crashed = dist > 1300 or y < -200 or y > 1200

        if collected:
            self.new_target()

        return self.get_obs(), collected, crashed

    def get_obs(self):
        return np.concatenate([self.state, self.target])

import pygame
import numpy as np
from human_env import HumanDroneEnv
import os

WIDTH = 800
HEIGHT = 800

class ManualHumanController:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Human Drone Test")

        self.env = HumanDroneEnv()
        self.obs = self.env.reset()
        self.clock = pygame.time.Clock()

        # LOAD sprites
        self.drone_frames = []
        for i in range(1, 5):
            img = pygame.image.load(os.path.join(
            "assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-"
            + str(i)
            + ".png"
        ))
            img = pygame.transform.scale(img, (80, 24))
            self.drone_frames.append(img)

        self.balloon_frames = []
        for i in range(1, 8):
            img = pygame.image.load(os.path.join(
            "assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/red-plain-"
            + str(i)
            + ".png"
        ))
            img = pygame.transform.scale(img, (30, 52))
            self.balloon_frames.append(img)

        self.drone_anim_speed = 0.3
        self.balloon_anim_speed = 0.1
        self.step_count = 0

    def draw_drone(self, x, y, angle_deg):
        frame = int(self.step_count * self.drone_anim_speed) % len(self.drone_frames)
        img = self.drone_frames[frame]

        rotated = pygame.transform.rotate(img, angle_deg)
        rect = rotated.get_rect(center=(x, y))
        self.screen.blit(rotated, rect)

    def draw_balloon(self, x, y):
        frame = int(self.step_count * self.balloon_anim_speed) % len(self.balloon_frames)
        img = self.balloon_frames[frame]
        rect = img.get_rect(center=(x, y))
        self.screen.blit(img, rect)

    def run(self):
        running = True

        while running:
            self.step_count += 1

            # INPUT
            up = down = left = right = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.env.Tl = self.env.thruster_mean
            self.env.Tr = self.env.thruster_mean

            keys = pygame.key.get_pressed()
            up = keys[pygame.K_UP]
            down = keys[pygame.K_DOWN]
            left = keys[pygame.K_LEFT]
            right = keys[pygame.K_RIGHT]

            # physics update ONCE per frame
            self.env.apply_human_control(up, down, left, right)
            obs, collected, crashed = self.env.step()

            # RENDER
            self.screen.fill((140, 180, 200))

            drone_x = int(obs[0])
            drone_y = int(obs[1])
            drone_angle = obs[4]
            target_x = int(obs[6])
            target_y = int(obs[7])

            self.draw_drone(drone_x, drone_y, drone_angle)
            self.draw_balloon(target_x, target_y)

            pygame.display.update()
            self.clock.tick(60)

            if crashed:
                print("Crashed → reset!")
                self.obs = self.env.reset()

            if collected:
                print("Collected balloon!")

        pygame.quit()

if __name__ == "__main__":
    ManualHumanController().run()

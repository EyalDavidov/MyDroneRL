import pygame
import numpy as np
import os
from math import atan2, pi, cos, sin
from stable_baselines3 import PPO

from env_figure8 import DroneEnvFigure8

WIDTH = 1000
HEIGHT = 1000
FPS = 60

# Model path
MODEL_PATH = "models\checkpoints\ppo_figure8_5000000_steps.zip" 

class DroneSimulationViewer:
    def __init__(self, model_path=MODEL_PATH):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Trained PPO Drone Simulation - Figure 8")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 20)

        self.env = DroneEnvFigure8()
        self.obs, _ = self.env.reset()

        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
        else:
            print(f"Model not found at {model_path}, using random agent.")
            self.model = None

        self.drone_frames = []
        asset_base = "../PPO/assets/balloon-flat-asset-pack/png/objects/drone-sprites/"
        if not os.path.exists(asset_base):
            self.use_sprites = False
        else:
            self.use_sprites = True
            for i in range(1, 5):
                img = pygame.image.load(asset_base + "drone-" + str(i) + ".png")
                img = pygame.transform.scale(img, (80, 24))
                self.drone_frames.append(img)

        self.frame_count = 0
        self.drone_anim_speed = 0.3

        self.display_reward = 0.0
        self.episode_timer = 0.0
        self.episode_reward = 0.0
        self.display_episode_reward = 0.0
        
        self.target_index = 0

    def draw_station(self, station, is_target):
        # Station now has 'green_seg' and 'red_seg' tuples of points
        
        g1, g2 = station['green_seg']
        r1, r2 = station['red_seg']
        
        width_line = 4
        
        color_g = (0, 255, 0) if is_target else (0, 100, 0)
        color_r = (255, 0, 0) if is_target else (100, 0, 0)
        
        # Cast to int for Pygame
        pygame.draw.line(self.screen, color_g, (int(g1[0]), int(g1[1])), (int(g2[0]), int(g2[1])), width_line)
        pygame.draw.line(self.screen, color_r, (int(r1[0]), int(r1[1])), (int(r2[0]), int(r2[1])), width_line)
        
        # Draw connector for visual clarity (transparent?)
        # pygame.draw.line(self.screen, (200, 200, 200), (int(g1[0]), int(g1[1])), (int(r1[0]), int(r1[1])), 1)

    def draw_drone(self, x, y, angle):
        if self.use_sprites:
            frame = int(self.frame_count * self.drone_anim_speed) % len(self.drone_frames)
            img = self.drone_frames[frame]
            rotated = pygame.transform.rotate(img, -angle)
            rect = rotated.get_rect(center=(x, y))
            self.screen.blit(rotated, rect)
        else:
            pygame.draw.circle(self.screen, (0, 0, 255), (int(x), int(y)), 10)
            dx = cos(angle * pi / 180) * 20
            dy = sin(angle * pi / 180) * 20
            pygame.draw.line(self.screen, (255, 0, 0), (int(x), int(y)), (int(x+dx), int(y+dy)), 2)
            
        # Draw sensor rays for debug
        # We can re-calc them here or pass from env
        for ray_a in self.env.ray_angles:
             tot = angle * pi / 180 + ray_a
             rx = x + cos(tot) * 100
             ry = y + sin(tot) * 100
             pygame.draw.line(self.screen, (200, 200, 0), (int(x), int(y)), (int(rx), int(ry)), 1)


    def draw_hud(self):
        ep_reward_text = f"Episode Reward: {self.display_episode_reward:.2f}"
        self.screen.blit(self.font.render(ep_reward_text, True, (20, 20, 20)), (10, 40))

        target_text = f"Target Station: {self.target_index + 1} / {len(self.env.stations)}"
        self.screen.blit(self.font.render(target_text, True, (20, 20, 20)), (10, 70))
        
        if self.target_index < len(self.env.stations):
             passed = self.env.stations[self.target_index]['passed_green']
             status = "Entered Green" if passed else "Approaching"
             st_text = f"Status: {status}"
             self.screen.blit(self.font.render(st_text, True, (20, 20, 20)), (10, 130))


    def run(self):
        running = True
        while running:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.frame_count += 1
            self.episode_timer += 1 / FPS

            if self.model:
                action, _ = self.model.predict(self.obs, deterministic=True)
            else:
                action = self.env.action_space.sample()

            self.obs, reward, terminated, truncated, _ = self.env.step(action)
            self.episode_reward += reward
            self.target_index = self.env.current_target_index
            
            if terminated or truncated:
                 self.display_episode_reward = self.episode_reward
            else:
                 self.display_episode_reward = self.episode_reward

            self.screen.fill((200, 230, 255)) 

            for i, st in enumerate(self.env.stations):
                self.draw_station(st, is_target=(i == self.env.current_target_index))

            x, y, vx, vy, angle_deg, ang_speed = self.env.state
            self.draw_drone(x, y, angle_deg)
            self.draw_hud()

            pygame.display.update()
            self.clock.tick(FPS)

            if terminated or truncated:
                print(f"Episode finished. Reward: {self.episode_reward}")
                self.obs, _ = self.env.reset()
                self.episode_timer = 0.0
                self.episode_reward = 0.0
                self.target_index = 0

        pygame.quit()

if __name__ == "__main__":
    DroneSimulationViewer().run()

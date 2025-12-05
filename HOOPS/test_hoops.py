import pygame
import numpy as np
import os
import glob
from math import atan2, pi, cos, sin
from stable_baselines3 import PPO

from env_hoops import HoopsDroneEnv

WIDTH = 1000
HEIGHT = 1000
FPS = 60

# Try to find the latest model
def get_latest_model():
    list_of_files = glob.glob('HOOPS/models/*/*.zip') 
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

MODEL_PATH = get_latest_model()

class DroneHoopsViewer:
    def __init__(self, model_path=MODEL_PATH):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Drone Hoops Challenge")
        self.clock = pygame.time.Clock()

        # ---- Font for HUD ----
        self.font = pygame.font.SysFont("consolas", 20)

        # ---- Environment ----
        self.env = HoopsDroneEnv()
        self.obs, _ = self.env.reset()

        # ---- Load PPO Model ----
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = PPO.load(model_path)
        else:
            print("No model found or specified. Running with random actions.")
            self.model = None

        # ---- Load Drone Sprites ----
        self.drone_frames = []
        # Assuming assets are in the root assets folder
        asset_path = "assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-"
        for i in range(1, 5):
            if os.path.exists(asset_path + str(i) + ".png"):
                img = pygame.image.load(asset_path + str(i) + ".png")
                img = pygame.transform.scale(img, (80, 24))
                self.drone_frames.append(img)
            else:
                # Fallback if assets not found
                surface = pygame.Surface((80, 24))
                surface.fill((0, 0, 0))
                self.drone_frames.append(surface)

        self.frame_count = 0
        self.drone_anim_speed = 0.3

        # ---------------- HUD VARIABLES ----------------
        self.display_reward = 0.0
        self.reward_update_timer = 0.0
        self.episode_timer = 0.0
        self.episode_reward = 0.0
        self.display_episode_reward = 0.0
        self.episode_reward_display_timer = 0.0

    # ---------------------------------------------------------
    def draw_drone(self, x, y, angle):
        if self.drone_frames:
            frame = int(self.frame_count * self.drone_anim_speed) % len(self.drone_frames)
            img = self.drone_frames[frame]
            rotated = pygame.transform.rotate(img, angle)
            rect = rotated.get_rect(center=(x, y))
            self.screen.blit(rotated, rect)
        else:
            pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), 10)

    def draw_hoops(self):
        for i, (hx, hy, hr, h_orient) in enumerate(self.env.hoops):
            color = (200, 200, 200) # Gray for inactive
            width = 2
            
            if i == self.env.current_hoop_index:
                color = (0, 255, 0) # Green for current target
                width = 4
            elif i < self.env.current_hoop_index:
                color = (100, 100, 100) # Darker gray for passed
            
            # Draw hoop circle
            pygame.draw.circle(self.screen, color, (int(hx), int(hy)), int(hr), width)
            
            # Draw orientation arrow
            # Arrow pointing in the direction of h_orient
            rad = h_orient * pi / 180.0
            end_x = hx + hr * cos(rad)
            end_y = hy + hr * sin(rad)
            # Start slightly before center to show direction clearly
            start_x = hx - (hr/2) * cos(rad)
            start_y = hy - (hr/2) * sin(rad)
            
            pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 2)
            
            # Draw arrow head
            arrow_len = 10
            arrow_angle = 0.5 # radians
            
            head_x1 = end_x - arrow_len * cos(rad - arrow_angle)
            head_y1 = end_y - arrow_len * sin(rad - arrow_angle)
            
            head_x2 = end_x - arrow_len * cos(rad + arrow_angle)
            head_y2 = end_y - arrow_len * sin(rad + arrow_angle)
            
            pygame.draw.line(self.screen, color, (end_x, end_y), (head_x1, head_y1), 2)
            pygame.draw.line(self.screen, color, (end_x, end_y), (head_x2, head_y2), 2)
            
            # Draw number
            text = self.font.render(str(i+1), True, color)
            self.screen.blit(text, (hx - 5, hy - hr - 20))

        # Draw finish landing zone if all hoops passed
        if self.env.current_hoop_index >= self.env.num_hoops:
            fx, fy = self.env.start_pos
            pygame.draw.circle(self.screen, (0, 0, 255), (int(fx), int(fy)), 20, 2)
            text = self.font.render("FINISH", True, (0, 0, 255))
            self.screen.blit(text, (fx - 20, fy - 40))

    # ---------------------------------------------------------
    def draw_wind_info(self):
        """Displays numerical wind force."""
        wind_x, wind_y = self.env.current_wind
        text = f"Wind X: {wind_x:+.3f}   Wind Y: {wind_y:+.3f}"
        surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(surface, (10, 10))

    # ---------------------------------------------------------
    def draw_hud(self):
        """HUD: rewards, timers, counters."""

        # Episode reward (updated once per sec)
        ep_reward_text = f"Episode Reward: {self.display_episode_reward:.2f}"
        self.screen.blit(self.font.render(ep_reward_text, True, (20, 20, 20)), (10, 40))

        # Hoops collected
        hoops_text = f"Hoops: {min(self.env.current_hoop_index, self.env.num_hoops)} / {self.env.num_hoops}"
        self.screen.blit(self.font.render(hoops_text, True, (20, 20, 20)), (10, 70))

        # Episode timer
        timer_text = f"Time: {self.episode_timer:.1f}s"
        self.screen.blit(self.font.render(timer_text, True, (20, 20, 20)), (10, 100))

        Tl_text = f"Thruster L: {self.env.Tl:.3f}"
        self.screen.blit(self.font.render(Tl_text, True, (20, 20, 20)), (400, 10))

        Tr_text = f"Thruster R: {self.env.Tr:.3f}"
        self.screen.blit(self.font.render(Tr_text, True, (20, 20, 20)), (600, 10))


    # ---------------------------------------------------------
    def run(self):
        running = True

        while running:
            pygame.event.pump()
            self.frame_count += 1

            # Update timers
            self.reward_update_timer += 1 / FPS
            self.episode_reward_display_timer += 1 / FPS
            self.episode_timer += 1 / FPS

            # Get model action
            if self.model:
                action, _ = self.model.predict(self.obs, deterministic=True)
            else:
                action = self.env.action_space.sample()

            # Step environment
            self.obs, reward, terminated, truncated, _ = self.env.step(action)

            # Update per-step reward display every 0.1s
            if self.reward_update_timer >= 0.1:
                self.display_reward = reward
                self.reward_update_timer = 0.0

            # Accumulate full episode reward
            self.episode_reward += reward

            # Update displayed episode reward every 1s
            if self.episode_reward_display_timer >= 1.0:
                self.display_episode_reward = self.episode_reward
                self.episode_reward_display_timer = 0.0

            # Extract drone state
            x, y, vx, vy, angle_deg, ang_speed = self.env.state

            # ---------- Rendering ----------
            self.screen.fill((240, 240, 255)) # Lighter background

            # Objects
            self.draw_hoops()
            self.draw_drone(int(x), int(y), angle_deg)
            self.draw_wind_info()
            self.draw_hud()

            pygame.display.update()
            self.clock.tick(FPS)

            # Reset episode
            if terminated or truncated:
                self.obs, _ = self.env.reset()
                self.episode_timer = 0.0
                self.episode_reward = 0.0
                self.display_episode_reward = 0.0
                self.episode_reward_display_timer = 0.0

        pygame.quit()


if __name__ == "__main__":
    DroneHoopsViewer().run()

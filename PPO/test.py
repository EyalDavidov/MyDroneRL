import pygame
import numpy as np
import os
from math import atan2, pi
from stable_baselines3 import PPO

from env import DroneEnv

WIDTH = 1000
HEIGHT = 1000
FPS = 60

MODEL_PATH = "HOOPS\models\ogrjmjem\ppo_drone_hoops_5000000_steps.zip"

class DroneSimulationViewer:
    def __init__(self, model_path=MODEL_PATH):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Trained PPO Drone Simulation")
        self.clock = pygame.time.Clock()

        # ---- Font for HUD ----
        self.font = pygame.font.SysFont("consolas", 20)

        # ---- Load PPO Model ----
        self.model = PPO.load(model_path)

        # ---- Environment ----
        self.env = DroneEnv()
        self.obs, _ = self.env.reset()

        # ---- Load Drone Sprites ----
        self.drone_frames = []
        for i in range(1, 5):
            img = pygame.image.load(
                os.path.join(
                    "assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-" + str(i) + ".png"
                )
            )
            img = pygame.transform.scale(img, (80, 24))
            self.drone_frames.append(img)

        # ---- Load Balloon Sprites ----
        self.balloon_frames = []
        for i in range(1, 8):
            img = pygame.image.load(
                os.path.join(
                    "assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/red-plain-" + str(i) + ".png"
                )
            )
            img = pygame.transform.scale(img, (30, 52))
            self.balloon_frames.append(img)


        self.frame_count = 0
        self.drone_anim_speed = 0.3
        self.balloon_anim_speed = 0.1

        # ---------------- HUD VARIABLES ----------------
        self.display_reward = 0.0
        self.reward_update_timer = 0.0

        self.balloons_collected = 0
        self.max_balloons_collected = 0

        self.episode_timer = 0.0

        # Episode reward accumulation
        self.episode_reward = 0.0
        self.display_episode_reward = 0.0
        self.episode_reward_display_timer = 0.0

    # ---------------------------------------------------------
    def draw_drone(self, x, y, angle):
        frame = int(self.frame_count * self.drone_anim_speed) % len(self.drone_frames)
        img = self.drone_frames[frame]
        rotated = pygame.transform.rotate(img, angle)
        rect = rotated.get_rect(center=(x, y))
        self.screen.blit(rotated, rect)

    def draw_balloon(self, x, y):
        frame = int(self.frame_count * self.balloon_anim_speed) % len(self.balloon_frames)
        img = self.balloon_frames[frame]
        rect = img.get_rect(center=(x, y))
        self.screen.blit(img, rect)

    # ---------------------------------------------------------
    def draw_wind_info(self):
        """Displays numerical wind force."""
        wind_x, wind_y = self.env.current_wind
        text = f"Wind X: {wind_x:+.3f}   Wind Y: {wind_y:+.3f}"
        surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(surface, (10, 10))

    # ---------------------------------------------------------
    def draw_wind_arrow(self):
        """Draw a large white arrow showing wind direction, with length proportional to wind magnitude."""
        wind_x, wind_y = self.env.current_wind

        if wind_x == 0 and wind_y == 0:
            return

        # Compute angle of wind vector
        angle_rad = atan2(wind_y, wind_x)
        angle_deg = -angle_rad * 180 / pi  # pygame rotates clockwise

        # ======== Scale arrow length by wind magnitude ========
        magnitude = (wind_x**2 + wind_y**2) ** 0.5
        base_length = 50
        scale_factor = 2000
        arrow_length = max(1, int(base_length + magnitude * scale_factor))
        # =======================================================

        base_x = WIDTH // 2
        base_y = HEIGHT // 2

        arrow = pygame.Surface((arrow_length, 30), pygame.SRCALPHA)

        pygame.draw.polygon(
            arrow,
            (255, 255, 255, 150),
            [
                (0, 15),
                (arrow_length - 30, 20),
                (arrow_length - 30, 5),
                (arrow_length, 15),
                (arrow_length - 30, 25),
                (arrow_length - 30, 10),
                (0, 15)
            ]
        )

        rotated_arrow = pygame.transform.rotate(arrow, angle_deg)
        rect = rotated_arrow.get_rect(center=(base_x, base_y))
        self.screen.blit(rotated_arrow, rect)


    # ---------------------------------------------------------
    def draw_hud(self):
        """HUD: rewards, timers, counters."""

        # Episode reward (updated once per sec)
        ep_reward_text = f"Episode Reward: {self.display_episode_reward:.2f}"
        self.screen.blit(self.font.render(ep_reward_text, True, (20, 20, 20)), (10, 40))

        # Balloons collected
        bc_text = f"Balloons: {self.balloons_collected}"
        self.screen.blit(self.font.render(bc_text, True, (20, 20, 20)), (10, 70))

        # Max balloons collected
        max_text = f"Max Balloons: {self.max_balloons_collected}"
        self.screen.blit(self.font.render(max_text, True, (20, 20, 20)), (10, 100))

        # Episode timer
        timer_text = f"Time: {self.episode_timer:.1f}s"
        self.screen.blit(self.font.render(timer_text, True, (20, 20, 20)), (10, 130))

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
            action, _ = self.model.predict(self.obs, deterministic=True)

            # Step environment
            prev_target = self.env.target.copy()
            self.obs, reward, terminated, truncated, _ = self.env.step(action)

            # Detect balloon collection
            if not np.array_equal(prev_target, self.env.target):
                self.balloons_collected += 1
                self.max_balloons_collected = max(self.max_balloons_collected, self.balloons_collected)

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
            tx, ty = self.env.target

            # ---------- Rendering ----------
            self.screen.fill((150, 200, 220))

            # BACKGROUND: Wind direction arrow
            self.draw_wind_arrow()

            # Foreground objects
            self.draw_drone(int(x), int(y), angle_deg)
            self.draw_balloon(int(tx), int(ty))
            self.draw_wind_info()
            self.draw_hud()

            pygame.display.update()
            self.clock.tick(FPS)

            # Reset episode
            if terminated or truncated:
                self.obs, _ = self.env.reset()
                self.balloons_collected = 0
                self.episode_timer = 0.0

                self.episode_reward = 0.0
                self.display_episode_reward = 0.0
                self.episode_reward_display_timer = 0.0

        pygame.quit()


if __name__ == "__main__":
    DroneSimulationViewer().run()

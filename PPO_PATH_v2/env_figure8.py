import numpy as np
from math import sin, cos, atan2, pi, sqrt
import gymnasium as gym
from gymnasium import spaces

class DroneEnvFigure8(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # ------- PHYSICS CONSTANTS -------
        self.gravity = 0.08
        self.thruster_mean = 0.04
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.005
        self.arm = 25
        self.mass = 1

        # ------- TRACK SETTINGS -------
        self.n_stations = 7
        self.stations = [] 
        self.current_target_index = 0
        self.station_width = 100 
        self.station_gap = 3 # Distance between Green and Red lines

        # ------- SENSOR SETTINGS -------
        self.num_rays = 18
        self.fov = 2 * pi
        self.sensor_range = 300
        self.ray_angles = np.linspace(-pi, pi, self.num_rays, endpoint=False)

        # -------- ACTION SPACE --------
        self.action_space = spaces.Discrete(5)

        # -------- OBSERVATION SPACE --------
        # 4 (Inertial) + 2 (Next Target) + 2 (Next-Next Target) + 2*num_rays (Sensor)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + 2 + 2 + 2 * self.num_rays,), dtype=np.float32)

        # -------- INTERNAL STATE --------
        self.state = None
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean

        self.time = 0
        self.time_limit = 60 
        
        # Station Timer tracking
        self.station_timer = 0.0
        self.station_timeout = 2.0 # Seconds to pass Red after Green

        self.reset()

    def generate_stations(self):
        self.stations = []
        cx, cy = 500, 500
        scale = 300
        
        ts = np.linspace(0, 2*pi, self.n_stations + 1)[:-1] 
        
        for t in ts:
            # Position on curve (Original Horizontal)
            # Rotate +90 degrees: x_new = -y_raw, y_new = x_raw
            x_rel = -1 * (scale * cos(t) * sin(t) / (1 + sin(t)**2))
            y_rel = scale * cos(t) / (1 + sin(t)**2)
            
            # Widen the path (Scale X (which was Y originally? No, X is X))
            # Width factor increased to 2.0 for wider loops
            width_factor = 2.0
            x_rel *= width_factor
            
            x = cx + x_rel
            y = cy + y_rel
            
            # Derivative for orientation (Tangent)
            dt = 0.01
            tp = t + dt
            
            xp_raw = -1 * (scale * cos(tp) * sin(tp) / (1 + sin(tp)**2))
            yp_raw = scale * cos(tp) / (1 + sin(tp)**2)
            
            xp = cx + xp_raw * width_factor
            yp = cy + yp_raw
            
            tangent_angle = atan2(yp - y, xp - x)
            
            dir_vec = np.array([cos(tangent_angle), sin(tangent_angle)])
            width_vec = np.array([-sin(tangent_angle), cos(tangent_angle)])
            
            green_center = np.array([x, y]) - dir_vec * (self.station_gap / 2)
            red_center   = np.array([x, y]) + dir_vec * (self.station_gap / 2)
            
            w = self.station_width
            
            g_p1 = green_center - width_vec * (w/2)
            g_p2 = green_center + width_vec * (w/2)
            
            r_p1 = red_center - width_vec * (w/2)
            r_p2 = red_center + width_vec * (w/2)
            
            self.stations.append({
                'center': np.array([x, y]),
                'angle': tangent_angle, 
                'green_seg': (g_p1, g_p2),
                'red_seg': (r_p1, r_p2),
                'passed_green': False, 
                'width': w
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.station_timer = 0.0
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean
        # Start outside the figure-8 (First station near 500, 800)
        self.state = np.array([700, 800, 0, 0, 0, 0], dtype=np.float32)
        
        self.generate_stations()
        self.current_target_index = 0
        
        return self.get_obs(), {}
    
    # ... (ray intersection helper remains same) ...

    def get_ray_intersection(self, ray_start, ray_dir, segment_start, segment_end):
        p = ray_start
        r = ray_dir
        q = segment_start
        s = segment_end - segment_start
        
        def cross(v1, v2):
            return v1[0]*v2[1] - v1[1]*v2[0]
            
        r_cross_s = cross(r, s)
        if r_cross_s == 0:
            return None
            
        q_minus_p = q - p
        t = cross(q_minus_p, s) / r_cross_s
        u = cross(q_minus_p, r) / r_cross_s
        
        if 0 <= u <= 1 and t > 0:
            return t
        return None

    def get_sensor_readings(self):
        x, y, _, _, angle_deg, _ = self.state
        drone_angle = angle_deg * pi / 180.0
        drone_pos = np.array([x, y])
        readings = []
        
        for ray_a in self.ray_angles:
            total_angle = drone_angle + ray_a
            ray_dir = np.array([cos(total_angle), sin(total_angle)])
            
            closest_dist = self.sensor_range
            detected_type = 0
            
            for st in self.stations:
                g1, g2 = st['green_seg']
                d_g = self.get_ray_intersection(drone_pos, ray_dir, g1, g2)
                if d_g is not None and d_g < closest_dist:
                    closest_dist = d_g
                    detected_type = 1 
                
                r1, r2 = st['red_seg']
                d_r = self.get_ray_intersection(drone_pos, ray_dir, r1, r2)
                if d_r is not None and d_r < closest_dist:
                    closest_dist = d_r
                    detected_type = -1 
            
            dist_norm = closest_dist / self.sensor_range
            readings.extend([dist_norm, detected_type])
            
        return np.array(readings, dtype=np.float32)

    def get_obs(self):
        x, y, vx, vy, angle_deg, ang_speed = self.state
        angle_to_up = angle_deg * pi / 180.0
        
        if self.current_target_index < len(self.stations):
            target = self.stations[self.current_target_index]
            tx, ty = target['center']
            dist_to_next = sqrt((tx - x)**2 + (ty - y)**2) / 500
            angle_to_next = atan2(ty - y, tx - x)
            
            # Next-Next Target
            if self.current_target_index + 1 < len(self.stations):
                target2 = self.stations[self.current_target_index + 1]
                tx2, ty2 = target2['center']
                dist_to_next2 = sqrt((tx2 - x)**2 + (ty2 - y)**2) / 500
                angle_to_next2 = atan2(ty2 - y, tx2 - x)
            else:
                dist_to_next2 = 0
                angle_to_next2 = 0
        else:
            dist_to_next = 0
            angle_to_next = 0
            dist_to_next2 = 0
            angle_to_next2 = 0
            
        inertial = np.array([angle_to_up, vx, vy, ang_speed], dtype=np.float32)
        gps = np.array([dist_to_next, angle_to_next], dtype=np.float32)
        gps2 = np.array([dist_to_next2, angle_to_next2], dtype=np.float32)
        sensor = self.get_sensor_readings()
        
        return np.concatenate([inertial, gps, gps2, sensor])

    def check_line_crossing(self, p_prev, p_curr, seg_p1, seg_p2):
        # Returns True if segment [p_prev, p_curr] intersects [seg_p1, seg_p2]
        # and direction of crossing?
        
        # We can implement full intersection check.
        # Line 1: P + t*R, Line 2: Q + u*S
        
        p = p_prev
        r = p_curr - p_prev
        q = seg_p1
        s = seg_p2 - seg_p1
        
        def cross(v1, v2):
             return v1[0]*v2[1] - v1[1]*v2[0]
             
        r_cross_s = cross(r, s)
        if r_cross_s == 0: return False # Parallel
        
        q_minus_p = q - p
        t = cross(q_minus_p, s) / r_cross_s
        u = cross(q_minus_p, r) / r_cross_s
        
        # t in [0, 1] means crossing happened between prev and curr
        # u in [0, 1] means it hit the segment length
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True
        return False
        
    def step(self, action):
        reward = 0
        self.time += 1/60.0
        terminated = False
        truncated = False
        
        # Action application (Physics)
        self.Tl = self.thruster_mean
        self.Tr = self.thruster_mean
        if action == 1: self.Tl += self.thruster_amplitude; self.Tr += self.thruster_amplitude
        elif action == 2: self.Tl -= self.thruster_amplitude; self.Tr -= self.thruster_amplitude
        elif action == 3: self.Tl -= self.diff_amplitude; self.Tr += self.diff_amplitude
        elif action == 4: self.Tr -= self.diff_amplitude; self.Tl += self.diff_amplitude
        
        self.Tl = np.clip(self.Tl, -0.05, 0.083)
        self.Tr = np.clip(self.Tr, -0.05, 0.083)
        
        x, y, vx, vy, angle_deg, ang_speed = self.state
        rad = angle_deg * pi / 180.0
        
        ax = -(self.Tl + self.Tr) * sin(rad)
        ay = self.gravity - (self.Tl + self.Tr) * cos(rad)
        ang_acc = self.arm * (self.Tr - self.Tl)
        
        vx += ax
        vy += ay
        ang_speed += ang_acc
        
        prev_x, prev_y = x, y
        x += vx
        y += vy
        angle_deg += ang_speed
        
        self.state = np.array([x, y, vx, vy, angle_deg, ang_speed], dtype=np.float32)
        
        # Check Crossings
        if self.current_target_index < len(self.stations):
            st = self.stations[self.current_target_index]
            
            # --- Timeout Logic ---
            if st['passed_green']:
                self.station_timer += 1/60.0
                if self.station_timer > self.station_timeout:
                    terminated = True
            
            p_prev = np.array([prev_x, prev_y])
            p_curr = np.array([x, y])
            
            # Check Green Crossing
            if self.check_line_crossing(p_prev, p_curr, st['green_seg'][0], st['green_seg'][1]):
                path_angle = st['angle']
                path_dir = np.array([cos(path_angle), sin(path_angle)])
                vel_dir = np.array([vx, vy])
                
                if np.dot(vel_dir, path_dir) > 0:
                    # Correct way through Green
                    if not st['passed_green']:
                        # Entered Green - Start Sequence
                        st['passed_green'] = True
                        self.station_timer = 0.0
                        # No immediate reward
                else:
                    # Wrong way through Green
                    reward -= 10
            
            # Check Red Crossing
            if self.check_line_crossing(p_prev, p_curr, st['red_seg'][0], st['red_seg'][1]):
                path_angle = st['angle']
                path_dir = np.array([cos(path_angle), sin(path_angle)])
                vel_dir = np.array([vx, vy])
                
                if np.dot(vel_dir, path_dir) > 0:
                    if st['passed_green']:
                        # SUCCESS! (Green then Red immediately)
                        
                        # Calculate proximity to center for reward
                        # Re-calculate 'u' parameter of intersection
                        p = p_prev
                        r = p_curr - p_prev
                        q = st['red_seg'][0]
                        s = st['red_seg'][1] - q
                        
                        # Helpers for cross product
                        def cross_p(v1, v2): return v1[0]*v2[1] - v1[1]*v2[0]
                        
                        r_cross_s = cross_p(r, s)
                        # We know it crossed, so r_cross_s != 0
                        
                        q_minus_p = q - p
                        u = cross_p(q_minus_p, r) / r_cross_s
                        
                        # u is fraction along segment [0, 1]. Center is 0.5.
                        # deviation = abs(u - 0.5) varies 0 to 0.5
                        # factor = 1 - (deviation / 0.5) -> 1 at center, 0 at edge
                        
                        center_factor = 1.0 - (abs(u - 0.5) * 2.0)
                        center_factor = max(0.0, min(1.0, center_factor)) # Clamp just in case
                        
                        reward_amount = 150.0 * center_factor
                        reward += reward_amount
                        
                        self.current_target_index += 1
                        if self.current_target_index >= len(self.stations):
                            reward += 500
                            terminated = True
                    else:
                        # Skipped Green or entered wrong
                        terminated = True
                else:
                    # Crossed Red backwards
                    reward -= 1000
                    terminated = True

            # Distance Penalty to Current Target
            tx, ty = st['center']
            dist = sqrt((tx - x)**2 + (ty - y)**2)
            reward -= dist / 600.0

        reward -= 1/60.0 # Time penalty
        
        if sqrt((x-500)**2 + (y-500)**2) > 1000:
            terminated = True
            reward -= 1000
            
        if self.time >= self.time_limit:
            truncated = True

        return self.get_obs(), float(reward), terminated, truncated, {}

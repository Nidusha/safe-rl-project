"""Warehouse safety simulation with a simple Q-learning robot controller.

This simulation uses a discretized observation space, a small discrete action
set, and reward shaping to guide the robot toward a shared goal while
avoiding static and dynamic obstacles.
"""

import arcade
import math
import heapq
import random
import time
from dataclasses import dataclass, field
from datetime import datetime


SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 960
SCREEN_TITLE = "Warehouse Safety + RL PoC Simulation"
SCREEN_ASPECT = SCREEN_WIDTH / SCREEN_HEIGHT

FLOOR_COLOR = (229, 232, 226)
GRID_COLOR = (216, 220, 214)
AISLE_COLOR = (201, 205, 198)
AISLE_MARKING = (245, 211, 79)
WALL_STRIPE = (150, 158, 150)

TEXT_COLOR = arcade.color.BLACK
INFO_COLOR = arcade.color.DARK_BLUE
SUCCESS_COLOR = arcade.color.DARK_GREEN
WARNING_COLOR = arcade.color.ORANGE
COLLISION_COLOR = arcade.color.RED

ROBOT_SAFE = (60, 180, 90)
ROBOT_WARNING = (245, 200, 60)
ROBOT_DANGER = (220, 70, 70)
ROBOT_TOP = (220, 230, 240)
ROBOT_DARK = (35, 55, 95)
ROBOT_WHEEL = (35, 35, 35)
ROBOT_SENSOR = (255, 220, 70)

AGV_SAFE = (70, 130, 255)
AGV_WARNING = (255, 185, 60)
AGV_DANGER = (235, 85, 85)
AGV_CHARGING = (120, 220, 255)

HUMAN_SKIN = (230, 190, 160)
HUMAN_HELMET = (245, 210, 60)
HUMAN_PANTS = (60, 70, 90)
HUMAN_COLORS = {
    "human": (231, 101, 72),
    "humanoid": (72, 162, 115),
    "quadruped": (120, 92, 190),
}

STATIC_COLORS = {
    "wall": (120, 118, 116),
    "rack": (108, 108, 110),
    "cupboard": (147, 113, 84),
    "table": (164, 136, 92),
    "chair": (110, 84, 62),
    "forklift": (224, 170, 44),
    "crane": (120, 140, 170),
}

GOAL_FILL = (219, 238, 213)
GOAL_BORDER = (67, 138, 73)
GOAL_TEXT = (44, 84, 48)

CHARGER_FILL = (208, 231, 241)
CHARGER_BORDER = (54, 113, 144)
CHARGER_TEXT = (28, 70, 92)

SAFE_ZONE_COLOR = (255, 207, 92, 10)
DANGER_ZONE_COLOR = (230, 94, 81, 14)
ZONE_SAFE_COLOR = (60, 180, 90, 24)
ZONE_WARNING_COLOR = (245, 200, 60, 34)
ZONE_DANGER_COLOR = (220, 70, 70, 42)
ZONE_CHARGING_COLOR = (120, 220, 255, 34)
PREDICTION_COLOR = (95, 101, 109, 70)
PATH_COLOR = (86, 124, 164, 55)
PANEL_COLOR = (245, 246, 241, 235)
PANEL_BORDER = (174, 180, 170)

ROBOT_RADIUS = 22
AGV_RADIUS = 20
HUMAN_RADIUS = 18
HUMANOID_RADIUS = 19
QUADRUPED_RADIUS = 16

ROBOT_BASE_SPEED = 4.6
ROBOT_SLOW_SPEED = 2.0
ROBOT_ESCAPE_SPEED = 1.2
ROBOT_DANGER_SPEED = ROBOT_ESCAPE_SPEED

AGV_SPEED_RANGE = (1.2, 3.2)
HUMAN_SPEED_RANGE = (0.7, 1.4)
HUMANOID_SPEED_RANGE = (0.8, 1.6)
QUADRUPED_SPEED_RANGE = (1.4, 2.2)

GOAL_REACHED_DISTANCE = 30
CHARGING_REACHED_DISTANCE = 28
WAYPOINT_REACHED_DISTANCE = 28

ANGLE_SMOOTHING = 0.22
RL_STEP_INTERVAL = 0.14
MESSAGE_HOLD_TIME = 0.40
NEAR_MISS_MARGIN = 22
RL_GAMMA = 0.92
RL_ALPHA = 0.14
RL_EPSILON = 0.12
PATH_SAMPLE_LIMIT = 240
PIXELS_PER_METER = 20.0
REPORT_PATH = "simulation_report.md"
WARNING_CLEARANCE_HYSTERESIS = 1.18
DANGER_CLEARANCE_HYSTERESIS = 1.22
OBSTACLE_CLEARANCE_MARGIN = 4.0
GOAL_GUIDANCE_WEIGHT = 1.35
DYNAMIC_SAFETY_MARGIN = 30.0
BODY_COLLISION_MARGIN = 3.0

ACTIONS = [
    (-1, 0),
    (1, 0),
    (0, 1),
    (0, -1),
    (-1, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (0, 0),
]

ACTION_LABELS = [
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "UP-LEFT",
    "UP-RIGHT",
    "DOWN-LEFT",
    "DOWN-RIGHT",
    "HOLD",
]


def initial_window_size():
    """Choose a startup size that fits the current laptop display."""
    try:
        display_width, display_height = arcade.get_display_size()
    except Exception:
        return 1280, 720

    max_width = max(900, display_width - 120)
    max_height = max(560, display_height - 180)
    width = min(SCREEN_WIDTH, max_width)
    height = int(width / SCREEN_ASPECT)

    if height > max_height:
        height = min(SCREEN_HEIGHT, max_height)
        width = int(height * SCREEN_ASPECT)

    return int(width), int(height)


def px_to_meters(value):
    return value / PIXELS_PER_METER


def clamp(value, low, high):
    return max(low, min(high, value))


def stable_safety_status(current_status, clearance, safe_distance, danger_distance):
    """Classify risk with hysteresis so circles do not flicker near thresholds."""
    if current_status == "HIGH RISK":
        if clearance <= danger_distance * DANGER_CLEARANCE_HYSTERESIS:
            return "HIGH RISK"
        if clearance <= safe_distance * WARNING_CLEARANCE_HYSTERESIS:
            return "WARNING"
        return "SAFE"

    if current_status == "WARNING":
        if clearance <= danger_distance:
            return "HIGH RISK"
        if clearance <= safe_distance * WARNING_CLEARANCE_HYSTERESIS:
            return "WARNING"
        return "SAFE"

    if clearance <= danger_distance:
        return "HIGH RISK"
    if clearance <= safe_distance:
        return "WARNING"
    return "SAFE"


def normalize(dx, dy):
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return 0.0, 0.0
    return dx / length, dy / length


def wrap_text_lines(text, max_chars):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def smooth_angle(current, target, factor=ANGLE_SMOOTHING):
    delta = normalize_angle(target - current)
    return current + delta * factor


def circle_rect_collision(circle_x, circle_y, radius, rect):
    rect_x, rect_y, rect_w, rect_h = rect
    left = rect_x - rect_w / 2
    right = rect_x + rect_w / 2
    bottom = rect_y - rect_h / 2
    top = rect_y + rect_h / 2
    closest_x = max(left, min(circle_x, right))
    closest_y = max(bottom, min(circle_y, top))
    dx = circle_x - closest_x
    dy = circle_y - closest_y
    return (dx * dx + dy * dy) < (radius * radius)


def distance_circle_to_rect_edge(circle_x, circle_y, rect):
    rect_x, rect_y, rect_w, rect_h = rect
    left = rect_x - rect_w / 2
    right = rect_x + rect_w / 2
    bottom = rect_y - rect_h / 2
    top = rect_y + rect_h / 2
    closest_x = max(left, min(circle_x, right))
    closest_y = max(bottom, min(circle_y, top))
    dx = circle_x - closest_x
    dy = circle_y - closest_y
    return math.sqrt(dx * dx + dy * dy)


@dataclass
class StaticObject:
    name: str
    category: str
    x: float
    y: float
    width: float
    height: float
    color: tuple[int, int, int]

    @property
    def rect(self):
        return (self.x, self.y, self.width, self.height)


@dataclass
class DynamicObject:
    name: str
    kind: str
    x: float
    y: float
    radius: float
    color: tuple[int, int, int]
    route: list[tuple[float, float]]
    speed_min: float
    speed_max: float
    speed: float
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 0.0
    route_index: int = 0
    status: str = "SAFE"
    safety_behavior: str = "CRUISE"
    last_clearance: float = 999.0
    adaptive_safe_distance: float = 150.0
    adaptive_danger_distance: float = 90.0
    message: str = ""

    def step_route(self):
        if not self.route:
            return 0.0, 0.0
        target_x, target_y = self.route[self.route_index]
        dx = target_x - self.x
        dy = target_y - self.y
        if math.sqrt(dx * dx + dy * dy) <= WAYPOINT_REACHED_DISTANCE:
            self.route_index = (self.route_index + 1) % len(self.route)
            target_x, target_y = self.route[self.route_index]
            dx = target_x - self.x
            dy = target_y - self.y
        return normalize(dx, dy)

    def scalar_speed(self):
        return math.sqrt(self.vx * self.vx + self.vy * self.vy)


@dataclass
class EvaluationSample:
    """One path-tracking sample for RMSE/MAE reporting."""

    step: int
    actual_x: float
    actual_y: float
    predicted_x: float
    predicted_y: float
    absolute_error: float


@dataclass
class SafetyMetrics:
    collisions: int = 0
    near_misses: int = 0
    unsafe_steps: int = 0
    total_reward: float = 0.0
    reward_samples: int = 0
    min_clearance: float = 999.0
    clearance_sum: float = 0.0
    clearance_samples: int = 0
    path_error_sum_sq: float = 0.0
    path_error_abs_sum: float = 0.0
    path_error_samples: int = 0
    tracking_errors: list[float] = field(default_factory=list)
    evaluation_samples: list[EvaluationSample] = field(default_factory=list)

    def record_clearance(self, clearance):
        self.min_clearance = min(self.min_clearance, clearance)
        self.clearance_sum += clearance
        self.clearance_samples += 1

    def record_path_error(self, actual_x, actual_y, predicted_x, predicted_y, error_value):
        self.path_error_sum_sq += error_value * error_value
        self.path_error_abs_sum += abs(error_value)
        self.path_error_samples += 1
        self.tracking_errors.append(error_value)
        self.evaluation_samples.append(
            EvaluationSample(
                self.path_error_samples,
                actual_x,
                actual_y,
                predicted_x,
                predicted_y,
                abs(error_value),
            )
        )
        if len(self.tracking_errors) > PATH_SAMPLE_LIMIT:
            self.tracking_errors.pop(0)
        if len(self.evaluation_samples) > PATH_SAMPLE_LIMIT:
            self.evaluation_samples.pop(0)

    @property
    def avg_clearance(self):
        if not self.clearance_samples:
            return 0.0
        return self.clearance_sum / self.clearance_samples

    @property
    def rmse(self):
        if not self.path_error_samples:
            return 0.0
        return math.sqrt(self.path_error_sum_sq / self.path_error_samples)

    @property
    def mae(self):
        if not self.path_error_samples:
            return 0.0
        return self.path_error_abs_sum / self.path_error_samples

    @property
    def rmse_m(self):
        return px_to_meters(self.rmse)

    @property
    def mae_m(self):
        return px_to_meters(self.mae)

    @property
    def avg_clearance_m(self):
        return px_to_meters(self.avg_clearance)

    @property
    def min_clearance_m(self):
        if self.min_clearance >= 999.0:
            return 0.0
        return px_to_meters(self.min_clearance)

    @property
    def average_reward(self):
        if not self.reward_samples:
            return 0.0
        return self.total_reward / self.reward_samples


class QLearningAgent:
    """Simple tabular Q-learning agent for the warehouse robot."""

    def __init__(self):
        self.q_table = {}
        self.last_state = None
        self.last_action = None

    def discretize(self, observation):
        """Convert raw observations into a compact discrete state tuple."""
        goal_sector = int(observation["goal_sector"])
        clearance_bucket = int(clamp(observation["clearance"] // 35, 0, 8))
        closing_bucket = int(clamp((observation["closing_speed"] + 2.5) // 1.0, 0, 6))
        static_bucket = int(clamp(observation["static_clearance"] // 35, 0, 8))
        danger_flag = 1 if observation["danger"] else 0
        return (goal_sector, clearance_bucket, closing_bucket, static_bucket, danger_flag)

    def select_action(self, state):
        """Choose an action with epsilon-greedy selection."""
        if random.random() < RL_EPSILON or state not in self.q_table:
            return random.randrange(len(ACTIONS))
        values = self.q_table[state]
        best_value = max(values)
        best_actions = [index for index, value in enumerate(values) if value == best_value]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """Update the Q-table using the Bellman equation."""
        self.q_table.setdefault(state, [0.0] * len(ACTIONS))
        self.q_table.setdefault(next_state, [0.0] * len(ACTIONS))
        old_value = self.q_table[state][action]
        next_best = max(self.q_table[next_state])
        self.q_table[state][action] = old_value + RL_ALPHA * (reward + RL_GAMMA * next_best - old_value)


class WarehouseSimulation(arcade.Window):
    def __init__(self):
        window_width, window_height = initial_window_size()
        super().__init__(window_width, window_height, SCREEN_TITLE, resizable=True, center_window=True)
        arcade.set_background_color(FLOOR_COLOR)
        self.view_camera = arcade.camera.Camera2D(
            position=(0, 0),
            projection=arcade.LRBT(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT),
            viewport=self.rect,
        )

        self.start_time = time.time()
        self.last_rl_step = 0.0
        self.rl_enabled = True
        self.rl_mode_name = "Q-Learning PoC"
        self.rl_agent = QLearningAgent()
        self.current_reward = 0.0
        self.last_observation_state = None
        self.last_action_index = None
        self.last_action_label = "HOLD"

        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        self.robot_start_x = 180.0
        self.robot_start_y = 790.0
        self.robot_x = self.robot_start_x
        self.robot_y = self.robot_start_y
        self.robot_angle = 0.0
        self.robot_color = ROBOT_SAFE
        self.robot_status = "SAFE"
        self.robot_behavior = "CRUISE"
        self.robot_speed = ROBOT_BASE_SPEED
        self.robot_message = ""
        self.robot_message_color = WARNING_COLOR
        self.robot_message_until = 0.0
        self.last_near_miss_time = 0.0

        self.goal_x = 1490.0
        self.goal_y = 170.0
        self.goal_reached = False
        self.completion_time = None
        self.robot_goal_route = [
            (500, 790),
            (960, 790),
            (960, 180),
            (1280, 180),
            (self.goal_x, self.goal_y),
        ]
        self.robot_goal_route_index = 0
        self.robot_goal_path = []

        self.charger_x = 1500.0
        self.charger_y = 820.0
        self.agv_battery = 100.0
        self.agv_low_battery_threshold = 20.0
        self.agv_resume_battery_threshold = 95.0
        self.agv_drain_rate = 0.05
        self.agv_charge_rate = 0.45
        self.agv_mode = "WORKING"
        self.agv_charge_pulse_time = 0.0
        self.agv_charge_route = []
        self.agv_charge_route_index = 0

        self.metrics = SafetyMetrics()

        self.static_objects = self.build_static_objects()
        self.safe_waypoints = self.build_safe_waypoints()
        self.dynamic_objects = self.build_dynamic_objects()
        self.agv_actor = next(actor for actor in self.dynamic_objects if actor.kind == "agv")

        self.robot_path_trace = [(self.robot_x, self.robot_y)]
        self.dashboard_lines = self.build_strategy_summary()
        self.update_view_camera()

    def update_view_camera(self):
        """Scale the virtual warehouse canvas to fit the actual window."""
        window_width = max(1, self.width)
        window_height = max(1, self.height)
        window_aspect = window_width / window_height

        if window_aspect > SCREEN_ASPECT:
            viewport_height = window_height
            viewport_width = int(viewport_height * SCREEN_ASPECT)
            viewport_left = int((window_width - viewport_width) / 2)
            viewport_bottom = 0
        else:
            viewport_width = window_width
            viewport_height = int(viewport_width / SCREEN_ASPECT)
            viewport_left = 0
            viewport_bottom = int((window_height - viewport_height) / 2)

        self.view_camera.viewport = arcade.LBWH(viewport_left, viewport_bottom, viewport_width, viewport_height)
        self.view_camera.position = (0, 0)
        self.view_camera.projection = arcade.LRBT(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT)

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.update_view_camera()

    def build_static_objects(self):
        objects = []
        rack_positions = [
            (520, 250), (520, 450), (520, 650),
            (980, 250), (980, 450), (980, 650),
        ]
        for index, (x, y) in enumerate(rack_positions, start=1):
            objects.append(StaticObject(f"Rack {index}", "rack", x, y, 120, 46, STATIC_COLORS["rack"]))

        objects.extend([
            StaticObject("Cupboard A", "cupboard", 170, 560, 100, 50, STATIC_COLORS["cupboard"]),
            StaticObject("Table A", "table", 1230, 520, 150, 68, STATIC_COLORS["table"]),
            StaticObject("Chair A", "chair", 1142, 520, 34, 34, STATIC_COLORS["chair"]),
            StaticObject("Chair B", "chair", 1318, 520, 34, 34, STATIC_COLORS["chair"]),
            StaticObject("Parked Forklift", "forklift", 1460, 530, 92, 54, STATIC_COLORS["forklift"]),
            StaticObject("Crane Control Base", "crane", 1490, 720, 92, 60, STATIC_COLORS["crane"]),
        ])
        return objects

    def build_safe_waypoints(self):
        """Known clear aisle points used by wandering agents."""
        points = [
            (180, 790), (300, 790), (420, 790), (640, 790), (780, 790),
            (1120, 790), (1280, 790), (1400, 790), (1500, 790),
            (500, 790), (500, 720), (500, 590), (500, 535), (500, 365),
            (500, 305), (500, 215), (500, 180),
            (960, 790), (960, 720), (960, 590), (960, 535), (960, 365),
            (960, 305), (960, 215), (960, 180),
            (180, 180), (300, 180), (420, 180), (640, 180), (780, 180),
            (1120, 180), (1280, 180), (1400, 180), (1500, 180),
            (1160, 620), (1320, 620), (1400, 620), (1400, 300), (1160, 300),
            (1160, 430), (1320, 430), (1400, 430), (1500, 300),
        ]
        return [
            point for point in points
            if not self.position_blocked(point[0], point[1], QUADRUPED_RADIUS, OBSTACLE_CLEARANCE_MARGIN)
        ]

    def build_dynamic_objects(self):
        return [
            DynamicObject(
                name="Transit AGV",
                kind="agv",
                x=280,
                y=790,
                radius=AGV_RADIUS,
                color=AGV_SAFE,
                route=[
                    (640, 790),
                    (1120, 790),
                    (1400, 790),
                    (1400, 650),
                    (1400, 300),
                    (1160, 300),
                    (1160, 790),
                    (780, 790),
                ],
                speed_min=AGV_SPEED_RANGE[0],
                speed_max=AGV_SPEED_RANGE[1],
                speed=2.5,
            ),
            DynamicObject(
                name="Worker 1",
                kind="human",
                x=300,
                y=560,
                radius=HUMAN_RADIUS,
                color=HUMAN_COLORS["human"],
                route=[],
                speed_min=HUMAN_SPEED_RANGE[0],
                speed_max=HUMAN_SPEED_RANGE[1],
                speed=1.1,
            ),
            DynamicObject(
                name="Worker 2",
                kind="humanoid",
                x=1110,
                y=575,
                radius=HUMANOID_RADIUS,
                color=HUMAN_COLORS["humanoid"],
                route=[],
                speed_min=HUMANOID_SPEED_RANGE[0],
                speed_max=HUMANOID_SPEED_RANGE[1],
                speed=1.2,
            ),
            DynamicObject(
                name="Inspection Quadruped",
                kind="quadruped",
                x=1180,
                y=180,
                radius=QUADRUPED_RADIUS,
                color=HUMAN_COLORS["quadruped"],
                route=[],
                speed_min=QUADRUPED_SPEED_RANGE[0],
                speed_max=QUADRUPED_SPEED_RANGE[1],
                speed=1.7,
            ),
        ]

    def build_strategy_summary(self):
        return [
            "RL agent: yellow warehouse robot.",
            "State: distance, speed, position, heading.",
            "Actions: move, slow, stop, hold.",
            "Reward: +10 safe, -10 risk/collision.",
            self.path_tracking_si_summary(),
            "Tools: Blender + NVIDIA Omniverse for 3D.",
        ]

    def path_tracking_si_summary(self):
        return f"RMSE: {self.metrics.rmse_m:.3f} m | MAE: {self.metrics.mae_m:.3f} m"

    def distance_between(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def position_blocked(self, x, y, radius, margin=0.0):
        effective_radius = radius + margin
        if not (effective_radius <= x <= SCREEN_WIDTH - effective_radius):
            return "wall"
        if not (effective_radius <= y <= SCREEN_HEIGHT - effective_radius):
            return "wall"

        for obj in self.static_objects:
            if circle_rect_collision(x, y, effective_radius, obj.rect):
                return obj.category
        return None

    def movement_path_blocked(self, start_x, start_y, end_x, end_y, radius, margin=OBSTACLE_CLEARANCE_MARGIN):
        distance = self.distance_between(start_x, start_y, end_x, end_y)
        samples = max(2, int(distance / max(1.0, radius * 0.35)))
        for step in range(1, samples + 1):
            t = step / samples
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t
            if self.position_blocked(x, y, radius, margin):
                return True
        return False

    def random_free_position(self, radius):
        """Pick a reachable-looking point that is not inside a wall or obstacle."""
        if hasattr(self, "safe_waypoints") and self.safe_waypoints:
            waypoints = self.safe_waypoints[:]
            random.shuffle(waypoints)
            for waypoint_x, waypoint_y in waypoints:
                x = waypoint_x + random.uniform(-16, 16)
                y = waypoint_y + random.uniform(-16, 16)
                if not self.position_blocked(x, y, radius, OBSTACLE_CLEARANCE_MARGIN):
                    return x, y

        movement_areas = [
            (130, 760, 1510, 820),
            (470, 150, 530, 800),
            (930, 150, 990, 800),
            (130, 150, 1510, 215),
        ]
        for _ in range(80):
            left, bottom, right, top = random.choice(movement_areas)
            x = random.uniform(left, right)
            y = random.uniform(bottom, top)
            if not self.position_blocked(x, y, radius, OBSTACLE_CLEARANCE_MARGIN):
                return x, y
        return radius + 80, radius + 80

    def choose_safe_wander_target(self, actor):
        """Choose a waypoint the actor can reach without crossing an obstacle."""
        if not self.safe_waypoints:
            return self.random_free_position(actor.radius)

        candidates = []
        for waypoint in self.safe_waypoints:
            distance = self.distance_between(actor.x, actor.y, waypoint[0], waypoint[1])
            if distance < 80:
                continue
            if self.movement_path_blocked(actor.x, actor.y, waypoint[0], waypoint[1], actor.radius):
                continue
            candidates.append((distance, waypoint))

        if candidates:
            nearby = [item for item in candidates if item[0] < 520]
            pool = nearby or candidates
            return random.choice(pool)[1]

        safe_x, safe_y = self.random_free_position(actor.radius)
        return safe_x, safe_y

    def realistic_task_target(self, actor):
        task_points = {
            "human": [
                (300, 560), (300, 340), (300, 790), (500, 590),
                (500, 305), (640, 180), (1160, 430), (1320, 430),
                (1320, 620), (1160, 620),
            ],
            "humanoid": [
                (1110, 575), (1160, 620), (1320, 620), (1400, 620),
                (1400, 790), (960, 790), (960, 590), (960, 305),
            ],
            "quadruped": [
                (1180, 180), (1280, 180), (1400, 180), (1500, 180),
                (960, 180), (640, 180), (500, 215), (300, 180),
            ],
        }.get(actor.kind, self.safe_waypoints)

        valid_targets = [
            point for point in task_points
            if not self.position_blocked(point[0], point[1], actor.radius, OBSTACLE_CLEARANCE_MARGIN)
        ]
        if not valid_targets:
            return self.choose_safe_wander_target(actor)

        distant_targets = [
            point for point in valid_targets
            if self.distance_between(actor.x, actor.y, point[0], point[1]) > 120
        ]
        return random.choice(distant_targets or valid_targets)

    def find_safe_path(self, start, goal, radius):
        """Find a path through clear warehouse waypoints."""
        start = (float(start[0]), float(start[1]))
        goal = (float(goal[0]), float(goal[1]))
        if not self.position_blocked(goal[0], goal[1], radius, OBSTACLE_CLEARANCE_MARGIN):
            if not self.movement_path_blocked(start[0], start[1], goal[0], goal[1], radius):
                return [goal]

        nodes = [start]
        nodes.extend(self.safe_waypoints)
        if not self.position_blocked(goal[0], goal[1], radius, OBSTACLE_CLEARANCE_MARGIN):
            nodes.append(goal)

        node_count = len(nodes)
        graph = [[] for _ in range(node_count)]
        max_edge_distance = 360
        for i in range(node_count):
            for j in range(i + 1, node_count):
                distance = self.distance_between(nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1])
                if distance > max_edge_distance:
                    continue
                if self.movement_path_blocked(nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1], radius):
                    continue
                graph[i].append((j, distance))
                graph[j].append((i, distance))

        goal_index = node_count - 1
        queue = [(0.0, 0)]
        costs = {0: 0.0}
        previous = {}
        while queue:
            cost, index = heapq.heappop(queue)
            if index == goal_index:
                break
            if cost > costs.get(index, float("inf")):
                continue
            for neighbor, edge_cost in graph[index]:
                next_cost = cost + edge_cost
                if next_cost >= costs.get(neighbor, float("inf")):
                    continue
                costs[neighbor] = next_cost
                previous[neighbor] = index
                heapq.heappush(queue, (next_cost, neighbor))

        if goal_index not in costs:
            return [self.choose_safe_wander_target(type("PathActor", (), {"x": start[0], "y": start[1], "radius": radius})())]

        path_indexes = []
        cursor = goal_index
        while cursor != 0:
            path_indexes.append(cursor)
            cursor = previous[cursor]
        path_indexes.reverse()
        return [nodes[index] for index in path_indexes]

    def nearest_static_clearance(self, x, y, radius):
        min_dist = float("inf")
        for obj in self.static_objects:
            edge_dist = distance_circle_to_rect_edge(x, y, obj.rect)
            clearance = max(0.0, edge_dist - radius)
            min_dist = min(min_dist, clearance)
        wall_clearance = min(x - radius, y - radius, SCREEN_WIDTH - x - radius, SCREEN_HEIGHT - y - radius)
        min_dist = min(min_dist, wall_clearance)
        return max(0.0, min_dist)

    def move_actor_to_safe_position_if_blocked(self, actor):
        if not self.position_blocked(actor.x, actor.y, actor.radius, OBSTACLE_CLEARANCE_MARGIN):
            return
        actor.x, actor.y = self.random_free_position(actor.radius)
        actor.vx = 0.0
        actor.vy = 0.0
        actor.route = [self.random_free_position(actor.radius)]
        actor.route_index = 0

    def nearest_dynamic_threat(self, x, y, radius, own_vx=0.0, own_vy=0.0, ignore=None):
        best = None
        best_score = -float("inf")

        for actor in self.dynamic_objects:
            if actor is ignore:
                continue
            dx = actor.x - x
            dy = actor.y - y
            distance = math.sqrt(dx * dx + dy * dy)
            clearance = distance - (radius + actor.radius)
            rel_vx = actor.vx - own_vx
            rel_vy = actor.vy - own_vy
            toward_dx, toward_dy = normalize(dx, dy)
            closing_speed = -(rel_vx * toward_dx + rel_vy * toward_dy)
            score = (180 - clearance) + max(0.0, closing_speed) * 40

            if score > best_score:
                best_score = score
                best = {
                    "actor": actor,
                    "distance": distance,
                    "clearance": clearance,
                    "closing_speed": closing_speed,
                    "relative_dx": dx,
                    "relative_dy": dy,
                }

        return best

    def closest_dynamic_pair_metrics(self):
        """Measure relative distance, speed, and direction between moving objects."""
        best = None
        for index, first in enumerate(self.dynamic_objects):
            for second in self.dynamic_objects[index + 1:]:
                dx = second.x - first.x
                dy = second.y - first.y
                distance = math.sqrt(dx * dx + dy * dy)
                clearance = distance - (first.radius + second.radius)
                rel_vx = second.vx - first.vx
                rel_vy = second.vy - first.vy
                relative_speed = math.sqrt(rel_vx * rel_vx + rel_vy * rel_vy)
                direction = math.degrees(math.atan2(dy, dx))
                toward_dx, toward_dy = normalize(dx, dy)
                closing_speed = -(rel_vx * toward_dx + rel_vy * toward_dy)

                if best is None or clearance < best["clearance"]:
                    best = {
                        "first": first,
                        "second": second,
                        "distance": distance,
                        "clearance": clearance,
                        "relative_speed": relative_speed,
                        "direction": direction,
                        "closing_speed": closing_speed,
                    }
        return best

    def adaptive_safety_distances(self, own_speed, other_speed, closing_speed):
        safe_distance = 70 + own_speed * 10 + other_speed * 8 + max(0.0, closing_speed) * 18
        danger_distance = safe_distance * 0.52
        return safe_distance, danger_distance

    def current_robot_route_target(self):
        if not self.robot_goal_path:
            self.robot_goal_path = self.find_safe_path(
                (self.robot_x, self.robot_y),
                (self.goal_x, self.goal_y),
                ROBOT_RADIUS,
            )

        while len(self.robot_goal_path) > 1:
            target_x, target_y = self.robot_goal_path[0]
            if self.distance_between(self.robot_x, self.robot_y, target_x, target_y) > WAYPOINT_REACHED_DISTANCE:
                break
            self.robot_goal_path.pop(0)

        if self.robot_goal_path:
            target_x, target_y = self.robot_goal_path[0]
            if self.distance_between(self.robot_x, self.robot_y, target_x, target_y) <= WAYPOINT_REACHED_DISTANCE:
                self.robot_goal_path = self.find_safe_path(
                    (self.robot_x, self.robot_y),
                    (self.goal_x, self.goal_y),
                    ROBOT_RADIUS,
                )
        return self.robot_goal_path[0] if self.robot_goal_path else (self.goal_x, self.goal_y)

    def reference_path_projection(self, x, y):
        x1, y1 = self.robot_start_x, self.robot_start_y
        x2, y2 = self.goal_x, self.goal_y
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return x1, y1, 0.0
        t = ((x - x1) * dx + (y - y1) * dy) / length_sq
        t = clamp(t, 0.0, 1.0)
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        error = self.distance_between(x, y, proj_x, proj_y)
        return proj_x, proj_y, error

    def build_robot_observation(self):
        """Build the current observation used by the RL agent."""
        threat = self.nearest_dynamic_threat(self.robot_x, self.robot_y, ROBOT_RADIUS)
        goal_dx = self.goal_x - self.robot_x
        goal_dy = self.goal_y - self.robot_y
        goal_angle = math.degrees(math.atan2(goal_dy, goal_dx))
        sector = int(((normalize_angle(goal_angle - self.robot_angle) + 180) // 45) % 8)

        clearance = 999.0 if threat is None else threat["clearance"]
        closing_speed = 0.0 if threat is None else threat["closing_speed"]
        static_clearance = self.nearest_static_clearance(self.robot_x, self.robot_y, ROBOT_RADIUS)
        danger = False
        if threat is not None:
            other = threat["actor"]
            safe_dist, danger_dist = self.adaptive_safety_distances(
                self.robot_speed, other.scalar_speed(), threat["closing_speed"]
            )
            danger = threat["clearance"] < danger_dist

        return {
            "goal_sector": sector,
            "goal_dx": goal_dx,
            "goal_dy": goal_dy,
            "clearance": clearance,
            "closing_speed": closing_speed,
            "static_clearance": static_clearance,
            "danger": danger,
        }

    def robot_clearance_snapshot(self):
        threat = self.nearest_dynamic_threat(self.robot_x, self.robot_y, ROBOT_RADIUS)
        static_clearance = self.nearest_static_clearance(self.robot_x, self.robot_y, ROBOT_RADIUS)
        if threat is None:
            return static_clearance
        return min(static_clearance, threat["clearance"])

    def compute_rl_reward(self, previous_goal_dist, new_goal_dist, moved, previous_clearance, new_clearance):
        """Compute reward for the RL agent based on progress and safety."""
        reward = (previous_goal_dist - new_goal_dist) * 0.7
        clearance_gain = new_clearance - previous_clearance
        if clearance_gain > 0:
            reward += min(6.0, clearance_gain * 0.25)
        reward += 10.0 if self.robot_status == "SAFE" and moved else 0.0
        reward -= 10.0 if self.robot_status == "WARNING" else 0.0
        reward -= 10.0 if self.robot_status == "HIGH RISK" else 0.0
        reward -= 10.0 if not moved else 0.0
        reward -= 0.01
        reward += 20.0 if self.goal_reached else 0.0
        return reward

    def set_robot_message(self, message, color=WARNING_COLOR):
        self.robot_message = message
        self.robot_message_color = color
        self.robot_message_until = time.time() + MESSAGE_HOLD_TIME

    def update_robot_risk_state(self):
        threat = self.nearest_dynamic_threat(self.robot_x, self.robot_y, ROBOT_RADIUS)
        static_clearance = self.nearest_static_clearance(self.robot_x, self.robot_y, ROBOT_RADIUS)

        if threat is None:
            clearance = static_clearance
            safe_distance = 110
            danger_distance = 65
            threat_label = "static"
            closing_speed = 0.0
        else:
            other = threat["actor"]
            clearance = min(threat["clearance"], static_clearance)
            safe_distance, danger_distance = self.adaptive_safety_distances(
                ROBOT_BASE_SPEED, other.scalar_speed(), threat["closing_speed"]
            )
            threat_label = other.kind
            closing_speed = threat["closing_speed"]

        self.metrics.record_clearance(clearance)

        status = stable_safety_status(self.robot_status, clearance, safe_distance, danger_distance)

        if status == "HIGH RISK":
            self.robot_status = "HIGH RISK"
            self.robot_behavior = "AVOID"
            self.robot_color = ROBOT_DANGER
            self.robot_speed = ROBOT_DANGER_SPEED
            self.metrics.unsafe_steps += 1
            self.set_robot_message(f"Avoid: {threat_label} conflict")
        elif status == "WARNING":
            self.robot_status = "WARNING"
            self.robot_behavior = "SLOW"
            self.robot_color = ROBOT_WARNING
            self.robot_speed = ROBOT_SLOW_SPEED
            if closing_speed > 0.4:
                self.set_robot_message(f"Slow down: closing on {threat_label}")
            else:
                self.set_robot_message(f"Slow zone near {threat_label}")
        else:
            self.robot_status = "SAFE"
            self.robot_behavior = "CRUISE"
            self.robot_color = ROBOT_SAFE
            self.robot_speed = ROBOT_BASE_SPEED

        if clearance <= NEAR_MISS_MARGIN and (time.time() - self.last_near_miss_time) > 0.75:
            self.metrics.near_misses += 1
            self.last_near_miss_time = time.time()

        predicted_x, predicted_y, tracking_error = self.reference_path_projection(self.robot_x, self.robot_y)
        self.metrics.record_path_error(self.robot_x, self.robot_y, predicted_x, predicted_y, tracking_error)

    def build_safety_vector_for_robot(self, intended_dx, intended_dy):
        final_dx = intended_dx
        final_dy = intended_dy
        threat = self.nearest_dynamic_threat(
            self.robot_x,
            self.robot_y,
            ROBOT_RADIUS,
            intended_dx * self.robot_speed,
            intended_dy * self.robot_speed,
        )

        if threat is not None:
            actor = threat["actor"]
            safe_distance, danger_distance = self.adaptive_safety_distances(
                self.robot_speed, actor.scalar_speed(), threat["closing_speed"]
            )
            if threat["clearance"] < safe_distance:
                away_dx, away_dy = normalize(-threat["relative_dx"], -threat["relative_dy"])
                side_dx = -away_dy
                side_dy = away_dx
                if side_dx * intended_dx + side_dy * intended_dy < 0:
                    side_dx *= -1
                    side_dy *= -1
                if threat["clearance"] < danger_distance:
                    return normalize(away_dx * 3.0 + side_dx * 0.35, away_dy * 3.0 + side_dy * 0.35)
                weight = 2.2 if threat["clearance"] < danger_distance else 1.1
                final_dx += away_dx * weight + side_dx * 0.8
                final_dy += away_dy * weight + side_dy * 0.8

        future_x = self.robot_x + intended_dx * 26
        future_y = self.robot_y + intended_dy * 26
        for obj in self.static_objects:
            future_clearance = max(0.0, distance_circle_to_rect_edge(future_x, future_y, obj.rect) - ROBOT_RADIUS)
            if future_clearance < 14:
                away_dx, away_dy = normalize(self.robot_x - obj.x, self.robot_y - obj.y)
                final_dx += away_dx * 0.7
                final_dy += away_dy * 0.7

        return normalize(final_dx, final_dy)

    def try_move_robot(self, move_dx, move_dy, step_override=None):
        if move_dx == 0 and move_dy == 0:
            return False

        step = self.robot_speed if step_override is None else step_override
        if step == 0:
            return False

        candidates = [
            (self.robot_x + move_dx * step, self.robot_y + move_dy * step),
            (self.robot_x + move_dx * step, self.robot_y),
            (self.robot_x, self.robot_y + move_dy * step),
            (self.robot_x + move_dy * step * 0.8, self.robot_y - move_dx * step * 0.8),
            (self.robot_x - move_dy * step * 0.8, self.robot_y + move_dx * step * 0.8),
        ]

        for nx, ny in candidates:
            if self.position_blocked(nx, ny, ROBOT_RADIUS, OBSTACLE_CLEARANCE_MARGIN):
                continue
            if self.movement_path_blocked(self.robot_x, self.robot_y, nx, ny, ROBOT_RADIUS):
                continue

            collision = False
            for actor in self.dynamic_objects:
                candidate_distance = self.distance_between(nx, ny, actor.x, actor.y)
                current_distance = self.distance_between(self.robot_x, self.robot_y, actor.x, actor.y)
                body_limit = ROBOT_RADIUS + actor.radius + BODY_COLLISION_MARGIN
                safety_limit = ROBOT_RADIUS + actor.radius + DYNAMIC_SAFETY_MARGIN
                if candidate_distance < body_limit:
                    collision = True
                    self.metrics.collisions += 1
                    self.set_robot_message(f"Collision blocked by {actor.kind}", COLLISION_COLOR)
                    break
                if candidate_distance < safety_limit and candidate_distance <= current_distance:
                    collision = True
                    self.set_robot_message(f"Safety buffer blocked by {actor.kind}", WARNING_COLOR)
                    break

            if collision:
                continue

            move_angle = math.degrees(math.atan2(ny - self.robot_y, nx - self.robot_x))
            self.robot_angle = smooth_angle(self.robot_angle, move_angle)
            self.robot_x = nx
            self.robot_y = ny
            self.robot_path_trace.append((self.robot_x, self.robot_y))
            if len(self.robot_path_trace) > PATH_SAMPLE_LIMIT:
                self.robot_path_trace.pop(0)
            return True

        return False

    def score_robot_move(self, move_dx, move_dy, route_target):
        move_dx, move_dy = normalize(move_dx, move_dy)
        if move_dx == 0 and move_dy == 0:
            return -float("inf")

        step = self.robot_speed if self.robot_speed > 0 else ROBOT_ESCAPE_SPEED
        nx = self.robot_x + move_dx * step
        ny = self.robot_y + move_dy * step
        if self.position_blocked(nx, ny, ROBOT_RADIUS, OBSTACLE_CLEARANCE_MARGIN):
            return -float("inf")
        if self.movement_path_blocked(self.robot_x, self.robot_y, nx, ny, ROBOT_RADIUS):
            return -float("inf")

        min_clearance = self.nearest_static_clearance(nx, ny, ROBOT_RADIUS)
        for actor in self.dynamic_objects:
            candidate_distance = self.distance_between(nx, ny, actor.x, actor.y)
            current_distance = self.distance_between(self.robot_x, self.robot_y, actor.x, actor.y)
            body_limit = ROBOT_RADIUS + actor.radius + BODY_COLLISION_MARGIN
            safety_limit = ROBOT_RADIUS + actor.radius + DYNAMIC_SAFETY_MARGIN
            if candidate_distance < body_limit:
                return -float("inf")
            if candidate_distance < safety_limit and candidate_distance <= current_distance:
                return -float("inf")
            min_clearance = min(min_clearance, candidate_distance - (ROBOT_RADIUS + actor.radius))

        current_route_distance = self.distance_between(self.robot_x, self.robot_y, route_target[0], route_target[1])
        candidate_route_distance = self.distance_between(nx, ny, route_target[0], route_target[1])
        route_progress = current_route_distance - candidate_route_distance
        return min_clearance * 0.35 + route_progress * 2.0

    def safety_shielded_robot_vector(self, primary_dx, primary_dy, route_dx, route_dy, route_target):
        candidates = [(primary_dx, primary_dy), (route_dx, route_dy)]
        for action_dx, action_dy in ACTIONS:
            action_dx, action_dy = normalize(action_dx, action_dy)
            candidates.append(normalize(action_dx + route_dx * GOAL_GUIDANCE_WEIGHT, action_dy + route_dy * GOAL_GUIDANCE_WEIGHT))
            candidates.append((action_dx, action_dy))

        best_vector = (0.0, 0.0)
        best_score = -float("inf")
        for candidate_dx, candidate_dy in candidates:
            score = self.score_robot_move(candidate_dx, candidate_dy, route_target)
            if score > best_score:
                best_score = score
                best_vector = normalize(candidate_dx, candidate_dy)

        if best_score == -float("inf"):
            return 0.0, 0.0
        return best_vector

    def build_manual_input_vector(self):
        dx = 0
        dy = 0
        if self.left_pressed:
            dx -= 1
        if self.right_pressed:
            dx += 1
        if self.up_pressed:
            dy += 1
        if self.down_pressed:
            dy -= 1
        return normalize(dx, dy)

    def rl_control_step(self):
        if not self.rl_enabled or self.goal_reached:
            return

        now = time.time()
        if now - self.last_rl_step < RL_STEP_INTERVAL:
            return
        self.last_rl_step = now

        observation = self.build_robot_observation()
        state = self.rl_agent.discretize(observation)
        action_index = self.rl_agent.select_action(state)
        self.last_action_label = ACTION_LABELS[action_index]

        intended_dx, intended_dy = ACTIONS[action_index]
        intended_dx, intended_dy = normalize(intended_dx, intended_dy)
        route_target_x, route_target_y = self.current_robot_route_target()
        goal_dx, goal_dy = normalize(route_target_x - self.robot_x, route_target_y - self.robot_y)
        if self.robot_status != "HIGH RISK":
            intended_dx, intended_dy = normalize(
                intended_dx + goal_dx * GOAL_GUIDANCE_WEIGHT,
                intended_dy + goal_dy * GOAL_GUIDANCE_WEIGHT,
            )
            self.last_action_label = f"{self.last_action_label}+ROUTE"
        safe_dx, safe_dy = self.build_safety_vector_for_robot(intended_dx, intended_dy)
        safe_dx, safe_dy = self.safety_shielded_robot_vector(safe_dx, safe_dy, goal_dx, goal_dy, (route_target_x, route_target_y))
        if safe_dx == 0 and safe_dy == 0:
            self.last_action_label = "SHIELD-HOLD"

        previous_goal_dist = self.distance_between(self.robot_x, self.robot_y, self.goal_x, self.goal_y)
        previous_clearance = self.robot_clearance_snapshot()
        moved = self.try_move_robot(safe_dx, safe_dy)
        if not moved:
            self.robot_goal_path = self.find_safe_path((self.robot_x, self.robot_y), (self.goal_x, self.goal_y), ROBOT_RADIUS)
            fallback_dx, fallback_dy = self.build_safety_vector_for_robot(goal_dx, goal_dy)
            moved = self.try_move_robot(fallback_dx, fallback_dy)
        self.check_goal()
        new_goal_dist = self.distance_between(self.robot_x, self.robot_y, self.goal_x, self.goal_y)
        new_clearance = self.robot_clearance_snapshot()

        reward = self.compute_rl_reward(previous_goal_dist, new_goal_dist, moved, previous_clearance, new_clearance)
        next_state = self.rl_agent.discretize(self.build_robot_observation())
        self.rl_agent.update(state, action_index, reward, next_state)

        self.current_reward = reward
        self.metrics.total_reward += reward
        self.metrics.reward_samples += 1
        self.last_observation_state = state
        self.last_action_index = action_index

    def metric_trend_explanation(self):
        errors = self.metrics.tracking_errors
        if len(errors) < 8:
            return (
                "RMSE and MAE are still stabilising because only a small number "
                "of evaluation samples have been collected."
            )

        midpoint = len(errors) // 2
        early_mae = sum(abs(value) for value in errors[:midpoint]) / midpoint
        late_mae = sum(abs(value) for value in errors[midpoint:]) / (len(errors) - midpoint)
        difference_m = px_to_meters(late_mae - early_mae)

        if difference_m > 0.20:
            return (
                "RMSE and MAE are increasing because recent robot positions are "
                "further from the predicted reference path. This indicates weaker "
                "path-tracking performance, usually caused by avoidance maneuvers, "
                "slow/stop safety responses, or exploratory RL actions."
            )
        if difference_m < -0.20:
            return (
                "RMSE and MAE are decreasing because recent robot positions are "
                "closer to the predicted reference path. This indicates improving "
                "model performance as the policy selects safer and more goal-aligned "
                "actions."
            )
        return (
            "RMSE and MAE are approximately stable, which indicates that the "
            "robot is maintaining similar path-tracking accuracy over recent "
            "time steps."
        )

    def build_evaluation_table_lines(self):
        samples = self.metrics.evaluation_samples[-8:]
        if not samples:
            return ["No evaluation samples collected yet."]

        lines = [
            "| Step | Actual x,y (m) | Predicted x,y (m) | Absolute error (m) |",
            "| --- | --- | --- | --- |",
        ]
        for sample in samples:
            actual = f"{px_to_meters(sample.actual_x):.2f}, {px_to_meters(sample.actual_y):.2f}"
            predicted = f"{px_to_meters(sample.predicted_x):.2f}, {px_to_meters(sample.predicted_y):.2f}"
            error = f"{px_to_meters(sample.absolute_error):.2f}"
            lines.append(f"| {sample.step} | {actual} | {predicted} | {error} |")
        return lines

    def build_marking_report(self):
        current_time = self.completion_time if self.goal_reached else (time.time() - self.start_time)
        min_clearance = self.metrics.min_clearance_m
        dynamic_pair = self.closest_dynamic_pair_metrics()
        if dynamic_pair:
            pair_summary = (
                f"{dynamic_pair['first'].kind} to {dynamic_pair['second'].kind}: "
                f"distance {px_to_meters(dynamic_pair['distance']):.2f} m, "
                f"clearance {px_to_meters(dynamic_pair['clearance']):.2f} m, "
                f"relative speed {px_to_meters(dynamic_pair['relative_speed']):.2f} m/tick, "
                f"direction {dynamic_pair['direction']:.0f} degrees."
            )
        else:
            pair_summary = "No dynamic object pair was available."
        lines = [
            "# Warehouse Safety RL Simulation Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Elapsed simulation time: {current_time:.1f} s",
            "",
            "## Environment and Object Design",
            "",
            "Static objects are fixed warehouse items: walls, racks/shelves, cupboard, table, chairs, parked forklift, crane control base, charger, loading bay, and packing area.",
            "",
            "Dynamic objects are moving agents: the yellow RL robot, AGV, human worker, humanoid robot, and quadruped robot.",
            "",
            "Forklift and crane objects are included as realistic warehouse obstacles. Dynamic objects use obstacle checks before moving so they should not pass through static objects.",
            "",
            "## Motion and Safety Analysis",
            "",
            f"Closest dynamic-object pair: {pair_summary}",
            "",
            "Relative distance is calculated using Euclidean distance between object centres, adjusted by each object's radius to estimate clearance. Motion is represented by velocity components, scalar speed, heading direction, and closing speed.",
            "",
            "The safety mechanism adapts the safe and danger thresholds using object speed and closing motion. The robot continues in safe areas, slows in warning areas, and stops or avoids movement in high-risk areas. Safety circles show green for safe, yellow for warning, and red for danger.",
            "",
            "## 1. 2D Simulation Evaluation Metrics",
            "",
            f"Distance unit: metres. The 2D display uses {PIXELS_PER_METER:.0f} pixels = 1 metre.",
            f"RMSE: {self.metrics.rmse_m:.3f} m",
            f"MAE: {self.metrics.mae_m:.3f} m",
            f"Minimum clearance: {min_clearance:.3f} m",
            f"Average clearance: {self.metrics.avg_clearance_m:.3f} m",
            "",
            "Actual values are the robot's measured positions. Predicted values are the closest points on the planned reference path from start to goal. Absolute error is the Euclidean distance between those two positions.",
            "",
            *self.build_evaluation_table_lines(),
            "",
            self.metric_trend_explanation(),
            "",
            "## 2. Reinforcement Learning Core Explanation",
            "",
            "AI reinforcement learning is used so the robot can learn from warehouse interactions, make adaptive decisions, and improve safety over time.",
            "",
            "- Agent: the decision-maker, represented by the warehouse robot/AGV.",
            "- Environment: warehouse layout containing humans, obstacles, racks, shelves, charging station, and moving robots.",
            "- State: the current situation, including distance to humans/objects, robot speed, position, goal direction, and danger flag.",
            "- Action: movement choice such as move forward, move left/right, slow, stop, or hold.",
            "- Reward function: +10 for safe successful movement, positive reward for moving closer to the goal or increasing clearance, and -10 for unsafe behaviour, blocked movement, warning state, or collision risk.",
            "- Policy: the Q-table strategy used by the robot to select actions with epsilon-greedy exploration.",
            "",
            "Rewards and penalties are applied after each action. Positive reward reinforces safe progress toward the goal and movement away from danger; negative reward discourages unsafe proximity, blocked movement, and high-risk states. Over repeated state-action-reward-policy-update cycles, the Q-values change so safer actions become more likely.",
            "",
            "## 3. RL Implementation Details",
            "",
            f"Learning agent: {self.rl_mode_name}.",
            f"Latest discrete state: {self.last_observation_state}.",
            f"Latest action: {self.last_action_label}.",
            f"Latest reward: {self.current_reward:.3f}.",
            "",
            "Decision cycle implemented in the simulation: observe state -> select action -> apply safety vector -> move robot -> calculate reward -> update Q-table policy.",
            "",
            "## 4. Input Data for RL",
            "",
            "- Distance between objects is measured using Euclidean distance between object centres, then adjusted by object radii to calculate clearance.",
            "- Object detection is represented by simulated sensor geometry over known dynamic actors and static rectangles. In a real robot this would be supplied by LiDAR and perception sensors.",
            "- Direction of movement is stored as velocity components and heading angle.",
            "- Example real-world human walking speed: 2-3 km/h, approximately 0.56-0.83 m/s.",
            "",
            "## 5. Motion and Simulation Logic",
            "",
            "At every clock tick, dynamic objects move along routes, the environment updates positions and safety clearances, the agent observes the new state, the agent chooses an action, and the robot applies movement with collision and safety checks.",
            "",
            "## 6. Safety Mechanism",
            "",
            "Safety is based on adaptive distance thresholds, dynamic object motion, and robot behaviour. Safe means continue at normal speed, warning means slow down, and danger/high risk means stop or avoid the conflict.",
            "",
            "## 7. Why Use Reinforcement Learning",
            "",
            "Traditional fixed rules are useful but not adaptive. RL is used because it learns from experience, handles changing warehouse conditions, and improves decision-making over time.",
            "",
            "## AI / RL Strategy",
            "",
            "The implemented AI approach is tabular Q-learning. It is suitable for this proof-of-concept because the robot state is discretised into goal direction, dynamic clearance, closing speed, static clearance, and danger flag. The Q-table stores action values for each state.",
            "",
            "A future Deep Reinforcement Learning version could use DQN, where a neural network estimates Q-values from continuous sensor inputs such as LiDAR distances, object velocities, and relative positions. The environment loop would remain state -> action -> reward -> policy update.",
            "",
            "## 8. Real-World Justification",
            "",
            "Real warehouse robots commonly use LiDAR, which stands for Light Detection and Ranging, to measure distance, detect surroundings, and avoid collisions.",
            "",
            "## 9. 3D Simulation Extension",
            "",
            "The 3D simulation should focus on manipulation strategy and a realistic warehouse environment. Blender can be used for modelling and animation, while NVIDIA Omniverse libraries can be used for physics-based simulation and digital-twin workflows.",
            "",
            "## 10. Literature Reference",
            "",
            "Sutton, R. S., and Barto, A. G. (2018). Reinforcement Learning: An Introduction. This supports the agent-environment framework, state, action, reward, policy, and value-learning concepts used in this simulation.",
            "",
        ]
        return "\n".join(lines)

    def export_marking_report(self):
        with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
            report_file.write(self.build_marking_report())
        self.set_robot_message(f"Report saved: {REPORT_PATH}", SUCCESS_COLOR)

    def check_goal(self):
        distance = self.distance_between(self.robot_x, self.robot_y, self.goal_x, self.goal_y)
        if distance < GOAL_REACHED_DISTANCE and not self.goal_reached:
            self.goal_reached = True
            self.completion_time = time.time() - self.start_time
            self.set_robot_message("Goal reached", SUCCESS_COLOR)
            self.robot_goal_path = [(self.goal_x, self.goal_y)]

    def can_dynamic_occupy(self, actor, x, y):
        if self.position_blocked(x, y, actor.radius, OBSTACLE_CLEARANCE_MARGIN):
            return False
        if self.movement_path_blocked(actor.x, actor.y, x, y, actor.radius):
            return False
        candidate_robot_distance = self.distance_between(x, y, self.robot_x, self.robot_y)
        current_robot_distance = self.distance_between(actor.x, actor.y, self.robot_x, self.robot_y)
        robot_body_limit = actor.radius + ROBOT_RADIUS + BODY_COLLISION_MARGIN
        robot_safety_limit = actor.radius + ROBOT_RADIUS + DYNAMIC_SAFETY_MARGIN
        if candidate_robot_distance < robot_body_limit:
            return False
        if candidate_robot_distance < robot_safety_limit and candidate_robot_distance <= current_robot_distance:
            return False

        for other in self.dynamic_objects:
            if other is actor:
                continue
            candidate_other_distance = self.distance_between(x, y, other.x, other.y)
            current_other_distance = self.distance_between(actor.x, actor.y, other.x, other.y)
            other_body_limit = actor.radius + other.radius + BODY_COLLISION_MARGIN
            other_safety_limit = actor.radius + other.radius + DYNAMIC_SAFETY_MARGIN
            if candidate_other_distance < other_body_limit:
                return False
            if candidate_other_distance < other_safety_limit and candidate_other_distance <= current_other_distance:
                return False
        return True

    def resolve_robot_dynamic_conflicts(self):
        """Emergency separation layer if any moving object gets too close to the RL robot."""
        for actor in self.dynamic_objects:
            dx = self.robot_x - actor.x
            dy = self.robot_y - actor.y
            distance = math.sqrt(dx * dx + dy * dy)
            minimum_distance = ROBOT_RADIUS + actor.radius + BODY_COLLISION_MARGIN
            if distance >= minimum_distance:
                continue

            away_dx, away_dy = normalize(dx, dy)
            if away_dx == 0 and away_dy == 0:
                away_dx, away_dy = 1.0, 0.0

            overlap = minimum_distance - distance
            moved_robot = self.try_move_robot(away_dx, away_dy, max(ROBOT_ESCAPE_SPEED, overlap + 2.0))
            if not moved_robot:
                actor_target_x = actor.x - away_dx * (overlap + 2.0)
                actor_target_y = actor.y - away_dy * (overlap + 2.0)
                if self.can_dynamic_occupy(actor, actor_target_x, actor_target_y):
                    actor.x = actor_target_x
                    actor.y = actor_target_y
                    actor.vx = -away_dx * (overlap + 2.0)
                    actor.vy = -away_dy * (overlap + 2.0)

            self.metrics.collisions += 1
            self.set_robot_message(f"Emergency separation from {actor.kind}", COLLISION_COLOR)

    def update_agv_battery(self):
        if self.agv_mode == "CHARGING":
            self.agv_battery = min(100.0, self.agv_battery + self.agv_charge_rate)
            self.agv_charge_pulse_time += 0.08
            self.agv_actor.vx = 0.0
            self.agv_actor.vy = 0.0
            self.agv_actor.speed = 0.0
        else:
            self.agv_battery = max(0.0, self.agv_battery - self.agv_drain_rate)

    def build_agv_charge_route(self):
        """Route AGV to the charger through clear aisles instead of diagonally through obstacles."""
        x = self.agv_actor.x
        y = self.agv_actor.y
        route = []

        if y < 500:
            route.append((1400, 300))
            route.append((1400, 650))
        elif y < 760:
            route.append((1400, 650))

        route.extend([
            (1400, 790),
            (1500, 790),
            (self.charger_x, self.charger_y),
        ])

        safe_route = []
        for waypoint in route:
            if not self.position_blocked(waypoint[0], waypoint[1], self.agv_actor.radius, OBSTACLE_CLEARANCE_MARGIN):
                safe_route.append(waypoint)

        if not safe_route:
            safe_route.append((x, y))
        return safe_route

    def update_agv_mode(self):
        distance_to_charger = self.distance_between(self.agv_actor.x, self.agv_actor.y, self.charger_x, self.charger_y)
        if self.agv_mode in ["WORKING", "RETURNING"] and self.agv_battery <= self.agv_low_battery_threshold:
            self.agv_mode = "GOING_TO_CHARGE"
            self.agv_charge_route = self.build_agv_charge_route()
            self.agv_charge_route_index = 0
        if self.agv_mode == "GOING_TO_CHARGE" and distance_to_charger <= CHARGING_REACHED_DISTANCE:
            self.agv_mode = "CHARGING"
        if self.agv_mode == "CHARGING" and self.agv_battery >= self.agv_resume_battery_threshold:
            self.agv_mode = "RETURNING"
            self.agv_charge_route = []
            self.agv_charge_route_index = 0

    def get_agv_target(self):
        if self.agv_mode == "GOING_TO_CHARGE":
            if not self.agv_charge_route:
                self.agv_charge_route = self.build_agv_charge_route()
                self.agv_charge_route_index = 0
            target = self.agv_charge_route[self.agv_charge_route_index]
            if self.distance_between(self.agv_actor.x, self.agv_actor.y, target[0], target[1]) <= WAYPOINT_REACHED_DISTANCE:
                self.agv_charge_route_index = min(self.agv_charge_route_index + 1, len(self.agv_charge_route) - 1)
                target = self.agv_charge_route[self.agv_charge_route_index]
            return target
        if self.agv_mode == "CHARGING":
            return self.agv_actor.x, self.agv_actor.y
        target = self.agv_actor.route[self.agv_actor.route_index]
        dx = target[0] - self.agv_actor.x
        dy = target[1] - self.agv_actor.y
        if math.sqrt(dx * dx + dy * dy) <= WAYPOINT_REACHED_DISTANCE:
            self.agv_actor.route_index = (self.agv_actor.route_index + 1) % len(self.agv_actor.route)
            target = self.agv_actor.route[self.agv_actor.route_index]
        return target

    def update_dynamic_actor_safety(self, actor, intended_dx, intended_dy):
        threat = self.nearest_dynamic_threat(actor.x, actor.y, actor.radius, actor.vx, actor.vy, ignore=actor)
        static_clearance = self.nearest_static_clearance(actor.x, actor.y, actor.radius)
        clearance = static_clearance
        closing_speed = 0.0
        other_speed = 0.0
        nearest_label = "static"

        if threat is not None:
            clearance = min(clearance, threat["clearance"])
            closing_speed = threat["closing_speed"]
            other_speed = threat["actor"].scalar_speed()
            nearest_label = threat["actor"].kind

        robot_dx = self.robot_x - actor.x
        robot_dy = self.robot_y - actor.y
        robot_distance = math.sqrt(robot_dx * robot_dx + robot_dy * robot_dy)
        robot_clearance = robot_distance - (actor.radius + ROBOT_RADIUS)
        robot_toward_dx, robot_toward_dy = normalize(robot_dx, robot_dy)
        robot_closing_speed = actor.vx * robot_toward_dx + actor.vy * robot_toward_dy
        if robot_clearance < clearance:
            clearance = robot_clearance
            closing_speed = robot_closing_speed
            other_speed = self.robot_speed
            nearest_label = "RL robot"

        safe_distance, danger_distance = self.adaptive_safety_distances(actor.speed, other_speed, closing_speed)
        actor.adaptive_safe_distance = safe_distance
        actor.adaptive_danger_distance = danger_distance
        actor.last_clearance = clearance

        status = stable_safety_status(actor.status, clearance, safe_distance, danger_distance)

        if status == "HIGH RISK":
            actor.status = "HIGH RISK"
            actor.safety_behavior = "AVOID"
            actor.message = f"Avoid {nearest_label}"
            target_speed = max(actor.speed_min * 0.35, 0.6)
        elif status == "WARNING":
            actor.status = "WARNING"
            actor.safety_behavior = "SLOW"
            actor.message = f"Slow near {nearest_label}"
            target_speed = actor.speed_min
        else:
            actor.status = "SAFE"
            actor.safety_behavior = "CRUISE"
            actor.message = ""
            target_speed = actor.speed_max

        actor.speed = clamp(target_speed, 0.0, actor.speed_max)

        final_dx = intended_dx
        final_dy = intended_dy
        if threat is not None and threat["clearance"] < safe_distance:
            away_dx, away_dy = normalize(-threat["relative_dx"], -threat["relative_dy"])
            side_dx = -away_dy
            side_dy = away_dx
            if side_dx * intended_dx + side_dy * intended_dy < 0:
                side_dx *= -1
                side_dy *= -1
            weight = 2.0 if threat["clearance"] < danger_distance else 1.0
            final_dx += away_dx * weight + side_dx * 0.6
            final_dy += away_dy * weight + side_dy * 0.6

        future_x = actor.x + intended_dx * 24
        future_y = actor.y + intended_dy * 24
        for obj in self.static_objects:
            future_clearance = max(0.0, distance_circle_to_rect_edge(future_x, future_y, obj.rect) - actor.radius)
            if future_clearance < 12:
                away_dx, away_dy = normalize(actor.x - obj.x, actor.y - obj.y)
                final_dx += away_dx * 0.65
                final_dy += away_dy * 0.65

        return normalize(final_dx, final_dy)

    def try_move_dynamic_actor(self, actor, move_dx, move_dy):
        if actor.speed <= 0 or (move_dx == 0 and move_dy == 0):
            actor.vx = 0.0
            actor.vy = 0.0
            return False

        step = actor.speed
        candidates = [
            (actor.x + move_dx * step, actor.y + move_dy * step),
            (actor.x + move_dx * step * 0.7, actor.y + move_dy * step * 0.7),
            (actor.x + move_dx * step, actor.y),
            (actor.x, actor.y + move_dy * step),
        ]

        for nx, ny in candidates:
            if not self.can_dynamic_occupy(actor, nx, ny):
                continue
            actor.vx = nx - actor.x
            actor.vy = ny - actor.y
            actor.angle = smooth_angle(actor.angle, math.degrees(math.atan2(actor.vy, actor.vx)))
            actor.x = nx
            actor.y = ny
            return True

        actor.vx = 0.0
        actor.vy = 0.0
        return False

    def update_dynamic_actors(self):
        self.update_agv_battery()
        self.update_agv_mode()

        for actor in self.dynamic_objects:
            self.move_actor_to_safe_position_if_blocked(actor)

            if actor.kind == "agv" and self.agv_mode == "CHARGING":
                actor.status = "CHARGING"
                actor.color = AGV_CHARGING
                actor.message = "Charging"
                continue

            if actor.kind == "agv":
                target_x, target_y = self.get_agv_target()
            else:
                if actor.route:
                    target_x, target_y = actor.route[actor.route_index]
                    if self.distance_between(actor.x, actor.y, target_x, target_y) <= WAYPOINT_REACHED_DISTANCE:
                        if actor.route_index < len(actor.route) - 1:
                            actor.route_index += 1
                        else:
                            task_target = self.realistic_task_target(actor)
                            actor.route = self.find_safe_path((actor.x, actor.y), task_target, actor.radius)
                            actor.route_index = 0
                        target_x, target_y = actor.route[actor.route_index]
                else:
                    task_target = self.realistic_task_target(actor)
                    actor.route = self.find_safe_path((actor.x, actor.y), task_target, actor.radius)
                    actor.route_index = 0
                    target_x, target_y = actor.route[0]

            intended_dx, intended_dy = normalize(target_x - actor.x, target_y - actor.y)
            safe_dx, safe_dy = self.update_dynamic_actor_safety(actor, intended_dx, intended_dy)
            moved = self.try_move_dynamic_actor(actor, safe_dx, safe_dy)
            if not moved and actor.kind != "agv":
                task_target = self.realistic_task_target(actor)
                actor.route = self.find_safe_path((actor.x, actor.y), task_target, actor.radius)
                actor.route_index = 0

            if actor.kind == "agv":
                if actor.status == "HIGH RISK":
                    actor.color = AGV_DANGER
                elif actor.status == "WARNING":
                    actor.color = AGV_WARNING
                else:
                    actor.color = AGV_SAFE

    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH, 66):
            arcade.draw_line(x, 0, x, SCREEN_HEIGHT, GRID_COLOR, 1)
        for y in range(0, SCREEN_HEIGHT, 66):
            arcade.draw_line(0, y, SCREEN_WIDTH, y, GRID_COLOR, 1)

    def draw_floor_markings(self):
        arcade.draw_lbwh_rectangle_filled(90, 120, 1500, 120, AISLE_COLOR)
        arcade.draw_lbwh_rectangle_filled(90, 735, 1500, 110, AISLE_COLOR)
        arcade.draw_lbwh_rectangle_filled(440, 120, 120, 725, AISLE_COLOR)
        arcade.draw_lbwh_rectangle_filled(900, 120, 120, 725, AISLE_COLOR)

        arcade.draw_lbwh_rectangle_filled(1060, 400, 390, 190, (224, 226, 220))
        arcade.draw_lbwh_rectangle_outline(1060, 400, 390, 190, PANEL_BORDER, 2)
        arcade.draw_text("PACKING AREA", 1255, 560, (84, 92, 82), 12, bold=True, anchor_x="center")

        arcade.draw_lbwh_rectangle_filled(1340, 90, 220, 120, (220, 224, 216))
        arcade.draw_lbwh_rectangle_outline(1340, 90, 220, 120, PANEL_BORDER, 2)
        arcade.draw_text("LOADING BAY", 1450, 168, (84, 92, 82), 13, bold=True, anchor_x="center")

        for x in range(130, 1510, 94):
            arcade.draw_lbwh_rectangle_filled(x, 173, 42, 8, AISLE_MARKING)
            arcade.draw_lbwh_rectangle_filled(x, 785, 42, 8, AISLE_MARKING)

        for y in range(150, 800, 94):
            arcade.draw_lbwh_rectangle_filled(486, y, 8, 42, AISLE_MARKING)
            arcade.draw_lbwh_rectangle_filled(946, y, 8, 42, AISLE_MARKING)

        arcade.draw_line(70, 100, 1590, 100, WALL_STRIPE, 3)
        arcade.draw_line(70, 860, 1590, 860, WALL_STRIPE, 3)

    def draw_static_object(self, obj):
        left = obj.x - obj.width / 2
        bottom = obj.y - obj.height / 2
        arcade.draw_lbwh_rectangle_filled(left, bottom, obj.width, obj.height, obj.color)
        arcade.draw_lbwh_rectangle_outline(left, bottom, obj.width, obj.height, (55, 55, 55), 2)

        if obj.category == "rack":
            for row in range(2):
                box_y = bottom + 5 + row * 15
                for col in range(2):
                    box_x = left + 12 + col * 28
                    box_color = (170, 115, 65) if row == 0 else (205, 155, 95)
                    arcade.draw_lbwh_rectangle_filled(box_x, box_y, 22, 12, box_color)
                    arcade.draw_lbwh_rectangle_outline(box_x, box_y, 22, 12, (110, 80, 40), 1)
        elif obj.category == "forklift":
            arcade.draw_circle_filled(obj.x - 18, obj.y - 18, 7, (40, 40, 40))
            arcade.draw_circle_filled(obj.x + 18, obj.y - 18, 7, (40, 40, 40))
            arcade.draw_line(obj.x + 20, obj.y - 6, obj.x + 34, obj.y - 6, (70, 70, 70), 3)
            arcade.draw_line(obj.x + 20, obj.y + 6, obj.x + 34, obj.y + 6, (70, 70, 70), 3)
        elif obj.category == "crane":
            arcade.draw_line(obj.x, obj.y + 20, obj.x, obj.y + 72, (70, 80, 100), 4)
            arcade.draw_line(obj.x, obj.y + 72, obj.x + 48, obj.y + 72, (70, 80, 100), 4)
            arcade.draw_circle_filled(obj.x + 48, obj.y + 58, 6, (120, 100, 60))

        if obj.category != "rack":
            arcade.draw_text(obj.category.upper(), obj.x, bottom + obj.height + 8, (78, 82, 76), 8, anchor_x="center")

    def draw_goal(self):
        arcade.draw_lbwh_rectangle_filled(1360, 110, 200, 86, GOAL_FILL)
        arcade.draw_lbwh_rectangle_outline(1360, 110, 200, 86, GOAL_BORDER, 3)
        arcade.draw_circle_outline(self.goal_x, self.goal_y, 24, GOAL_BORDER, 4)
        arcade.draw_circle_filled(self.goal_x, self.goal_y, 8, GOAL_BORDER)
        arcade.draw_text("RL ROBOT GOAL", 1460, 164, GOAL_TEXT, 13, bold=True, anchor_x="center")
        arcade.draw_text("+20 reward target", 1460, 144, GOAL_TEXT, 9, anchor_x="center")

    def draw_charging_station(self):
        arcade.draw_lbwh_rectangle_filled(1380, 690, 180, 96, CHARGER_FILL)
        arcade.draw_lbwh_rectangle_outline(1380, 690, 180, 96, CHARGER_BORDER, 3)
        arcade.draw_circle_outline(self.charger_x, self.charger_y, 26, CHARGER_BORDER, 4)
        arcade.draw_circle_filled(self.charger_x, self.charger_y, 8, CHARGER_BORDER)
        if self.agv_mode == "CHARGING":
            pulse_radius = 32 + 8 * math.sin(self.agv_charge_pulse_time)
            arcade.draw_circle_outline(self.charger_x, self.charger_y, pulse_radius, AGV_CHARGING, 4)
        arcade.draw_text("CHARGER", 1470, 760, CHARGER_TEXT, 12, bold=True, anchor_x="center")

    def draw_dynamic_actor(self, actor):
        zone_color = ZONE_SAFE_COLOR
        if actor.status == "WARNING":
            zone_color = ZONE_WARNING_COLOR
        elif actor.status == "HIGH RISK":
            zone_color = ZONE_DANGER_COLOR
        elif actor.status == "CHARGING":
            zone_color = ZONE_CHARGING_COLOR

        arcade.draw_circle_filled(actor.x, actor.y, actor.adaptive_safe_distance, zone_color)
        arcade.draw_circle_outline(actor.x, actor.y, actor.adaptive_danger_distance, zone_color, 2)
        arcade.draw_line(actor.x, actor.y, actor.x + actor.vx * 8, actor.y + actor.vy * 8, PREDICTION_COLOR, 2)

        if actor.kind == "agv":
            self.draw_robot_body(actor.x, actor.y, actor.angle, actor.color, 0.92)
        elif actor.kind in ["human", "humanoid"]:
            self.draw_human(actor.x, actor.y, actor.color)
        else:
            self.draw_quadruped(actor.x, actor.y, actor.color)

        actor_labels = {
            "agv": "AGV\nbattery robot",
            "human": "HUMAN\nmoving person",
            "humanoid": "HUMANOID\nmoving robot",
            "quadruped": "QUADRUPED\nmoving robot",
        }
        arcade.draw_text(
            actor_labels.get(actor.kind, actor.kind.upper()),
            actor.x,
            actor.y + actor.radius + 16,
            TEXT_COLOR,
            8,
            bold=True,
            anchor_x="center",
            multiline=True,
            align="center",
            width=110,
        )

    def draw_human(self, x, y, shirt_color):
        arcade.draw_ellipse_filled(x, y - 34, 28, 10, (160, 160, 160, 100))
        arcade.draw_arc_filled(x, y + 23, 14, 10, HUMAN_HELMET, 0, 180)
        arcade.draw_circle_filled(x, y + 15, 11, HUMAN_SKIN)
        arcade.draw_lbwh_rectangle_filled(x - 11, y - 12, 22, 28, shirt_color)
        arcade.draw_lbwh_rectangle_outline(x - 11, y - 12, 22, 28, (40, 40, 40), 1)
        arcade.draw_line(x - 10, y + 6, x - 18, y - 8, HUMAN_SKIN, 4)
        arcade.draw_line(x + 10, y + 6, x + 18, y - 8, HUMAN_SKIN, 4)
        arcade.draw_line(x - 5, y - 12, x - 8, y - 34, HUMAN_PANTS, 4)
        arcade.draw_line(x + 5, y - 12, x + 8, y - 34, HUMAN_PANTS, 4)

    def draw_quadruped(self, x, y, color):
        arcade.draw_ellipse_filled(x, y, 36, 20, color)
        arcade.draw_circle_filled(x + 18, y + 6, 8, color)
        for leg_offset in [-12, -4, 6, 14]:
            arcade.draw_line(x + leg_offset, y - 8, x + leg_offset, y - 24, (50, 50, 50), 3)
        arcade.draw_line(x - 18, y + 4, x - 28, y + 10, (50, 50, 50), 2)

    def draw_robot_body(self, x, y, angle, color, radius_scale=1.0):
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        arcade.draw_ellipse_filled(x, y - 26 * radius_scale, 34 * radius_scale, 10 * radius_scale, (150, 150, 150, 90))
        arcade.draw_rect_filled(arcade.XYWH(x, y, 48 * radius_scale, 34 * radius_scale), color, angle)
        arcade.draw_rect_outline(arcade.XYWH(x, y, 48 * radius_scale, 34 * radius_scale), ROBOT_DARK, 2, angle)
        arcade.draw_rect_filled(arcade.XYWH(x, y + 2 * radius_scale, 26 * radius_scale, 16 * radius_scale), ROBOT_TOP, angle)
        for ox, oy in [(-17, -15), (17, -15), (-17, 15), (17, 15)]:
            wx = x + ox * radius_scale * cos_a - oy * radius_scale * sin_a
            wy = y + ox * radius_scale * sin_a + oy * radius_scale * cos_a
            arcade.draw_circle_filled(wx, wy, 4.5 * radius_scale, ROBOT_WHEEL)
        fx = x + 20 * radius_scale * cos_a
        fy = y + 20 * radius_scale * sin_a
        arcade.draw_circle_filled(fx, fy, 5 * radius_scale, ROBOT_SENSOR)

    def draw_learning_robot_body(self, x, y, angle, color):
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        def rotate_point(px, py):
            return x + px * cos_a - py * sin_a, y + px * sin_a + py * cos_a

        nose = rotate_point(28, 0)
        rear_left = rotate_point(-22, 18)
        rear_right = rotate_point(-22, -18)
        arcade.draw_ellipse_filled(x, y - 28, 42, 12, (150, 150, 150, 85))
        arcade.draw_polygon_filled([nose, rear_left, rear_right], color)
        arcade.draw_polygon_outline([nose, rear_left, rear_right], ROBOT_DARK, 3)

        core = rotate_point(-2, 0)
        arcade.draw_circle_filled(core[0], core[1], 12, ROBOT_TOP)
        arcade.draw_circle_outline(core[0], core[1], 12, ROBOT_DARK, 2)

        for ox, oy in [(-16, -18), (-16, 18), (10, -12), (10, 12)]:
            wx, wy = rotate_point(ox, oy)
            arcade.draw_circle_filled(wx, wy, 5, ROBOT_WHEEL)

        sensor = rotate_point(22, 0)
        arcade.draw_circle_filled(sensor[0], sensor[1], 6, ROBOT_SENSOR)
        arcade.draw_circle_outline(sensor[0], sensor[1], 9, ROBOT_SENSOR, 2)

        antenna_base = rotate_point(-8, 0)
        antenna_tip = rotate_point(-18, 0)
        arcade.draw_line(antenna_base[0], antenna_base[1], antenna_tip[0], antenna_tip[1], ROBOT_DARK, 2)
        arcade.draw_circle_filled(antenna_tip[0], antenna_tip[1], 3, ROBOT_DARK)

    def draw_robot_path(self):
        if len(self.robot_path_trace) < 2:
            return
        for index in range(1, len(self.robot_path_trace)):
            x1, y1 = self.robot_path_trace[index - 1]
            x2, y2 = self.robot_path_trace[index]
            arcade.draw_line(x1, y1, x2, y2, PATH_COLOR, 2)

    def draw_info_panel(self):
        panel_x = 18
        panel_y = SCREEN_HEIGHT - 116
        panel_w = SCREEN_WIDTH - 36
        panel_h = 98

        arcade.draw_lbwh_rectangle_filled(panel_x, panel_y, panel_w, panel_h, PANEL_COLOR)
        arcade.draw_lbwh_rectangle_outline(panel_x, panel_y, panel_w, panel_h, PANEL_BORDER, 2)

        current_time = self.completion_time if self.goal_reached else (time.time() - self.start_time)
        arcade.draw_text("Safety Dashboard", panel_x + 14, panel_y + 73, TEXT_COLOR, 13, bold=True)
        arcade.draw_text(f"Time: {current_time:.1f} s", panel_x + 190, panel_y + 75, TEXT_COLOR, 9)

        agv_mode_short = {
            "WORKING": "WORKING",
            "GOING_TO_CHARGE": "TO CHARGE",
            "CHARGING": "CHARGING",
            "RETURNING": "RETURNING",
        }.get(self.agv_mode, self.agv_mode)
        agv_explain = {
            "WORKING": "following route",
            "GOING_TO_CHARGE": "following safe aisle route",
            "CHARGING": "parked at safe charger dock",
            "RETURNING": "leaving charger",
        }.get(self.agv_mode, "active")

        message = self.robot_message if time.time() <= self.robot_message_until else ""
        latest = self.metrics.evaluation_samples[-1] if self.metrics.evaluation_samples else None
        if latest:
            error = f"abs error {px_to_meters(latest.absolute_error):.2f}m"
        else:
            error = "abs error pending"
        dynamic_pair = self.closest_dynamic_pair_metrics()
        if dynamic_pair:
            pair_text = (
                f"Nearest pair: {dynamic_pair['first'].kind}-{dynamic_pair['second'].kind} "
                f"{px_to_meters(dynamic_pair['clearance']):.2f} m clear, "
                f"rel speed {px_to_meters(dynamic_pair['relative_speed']):.2f} m/tick"
            )
        else:
            pair_text = "Nearest pair: waiting for dynamic objects"

        section_w = panel_w / 4
        section_xs = [panel_x + 14, panel_x + section_w + 14, panel_x + section_w * 2 + 14, panel_x + section_w * 3 + 14]
        for boundary in range(1, 4):
            line_x = panel_x + section_w * boundary
            arcade.draw_line(line_x, panel_y + 10, line_x, panel_y + panel_h - 12, PANEL_BORDER, 1)

        title_y = panel_y + 50
        row_1 = panel_y + 35
        row_2 = panel_y + 20
        row_3 = panel_y + 7

        arcade.draw_text("1. Learning robot decision", section_xs[0], title_y, INFO_COLOR, 9, bold=True)
        arcade.draw_text(f"Mode: {'Shielded safety RL' if self.rl_enabled else 'manual control'}", section_xs[0], row_1, TEXT_COLOR, 8)
        arcade.draw_text(f"Agent: yellow RL robot | Status: {self.robot_status}", section_xs[0], row_2, TEXT_COLOR, 8)
        route_target = self.current_robot_route_target()
        arcade.draw_text(
            (
                f"Action: {self.last_action_label} | "
                f"Speed: {px_to_meters(self.robot_speed):.2f} m/tick | "
                f"Route target: ({px_to_meters(route_target[0]):.1f},"
                f"{px_to_meters(route_target[1]):.1f}) m"
            ),
            section_xs[0],
            row_3,
            TEXT_COLOR,
            8,
        )

        arcade.draw_text("2. Reward and penalties", section_xs[1], title_y, INFO_COLOR, 9, bold=True)
        arcade.draw_text(
            f"Current reward: {self.current_reward:.2f} pts | average: {self.metrics.average_reward:.2f} pts",
            section_xs[1],
            row_1,
            TEXT_COLOR,
            8,
        )
        arcade.draw_text("+10 safe move | +reward for more clearance", section_xs[1], row_2, TEXT_COLOR, 8)
        arcade.draw_text("Shared goal = robot target | +20 when reached", section_xs[1], row_3, TEXT_COLOR, 8)

        arcade.draw_text("3. AGV and charging", section_xs[2], title_y, INFO_COLOR, 9, bold=True)
        arcade.draw_text("Agents follow realistic task routes via safe waypoints", section_xs[2], row_1, TEXT_COLOR, 8)
        arcade.draw_text(f"Battery: {self.agv_battery:.0f} % | AGV state: {self.agv_actor.status}", section_xs[2], row_2, TEXT_COLOR, 8)
        arcade.draw_text(f"Mode: {agv_mode_short} ({agv_explain})", section_xs[2], row_3, TEXT_COLOR, 8)

        arcade.draw_text("4. Safety and evaluation", section_xs[3], title_y, INFO_COLOR, 9, bold=True)
        arcade.draw_text(f"Alert: {message or 'None'}", section_xs[3], row_1, self.robot_message_color if message else TEXT_COLOR, 8)
        arcade.draw_text(self.path_tracking_si_summary(), section_xs[3], row_2, TEXT_COLOR, 8)
        arcade.draw_text(pair_text, section_xs[3], row_3, TEXT_COLOR, 8)

        arcade.draw_text("Controls: arrows manual | R toggle RL | E export report", panel_x + 1245, panel_y + 75, (70, 76, 88), 8)

    def draw_requirement_panel(self):
        panel_w = 300
        panel_h = 140
        panel_x = 18
        panel_y = 520
        arcade.draw_lbwh_rectangle_filled(panel_x, panel_y, panel_w, panel_h, PANEL_COLOR)
        arcade.draw_lbwh_rectangle_outline(panel_x, panel_y, panel_w, panel_h, PANEL_BORDER, 2)
        arcade.draw_text("Summary", panel_x + 14, panel_y + panel_h - 24, INFO_COLOR, 13, bold=True)

        y = panel_y + panel_h - 44
        for line in self.build_strategy_summary():
            wrapped_lines = wrap_text_lines(line, 38)
            for wrapped_line in wrapped_lines:
                arcade.draw_text(wrapped_line, panel_x + 14, y, TEXT_COLOR, 8)
                y -= 14
            y -= 2

    def on_draw(self):
        self.clear()
        with self.view_camera.activate():
            self.draw_grid()
            self.draw_floor_markings()

            for obj in self.static_objects:
                self.draw_static_object(obj)

            self.draw_goal()
            self.draw_charging_station()

            for actor in self.dynamic_objects:
                self.draw_dynamic_actor(actor)

            self.draw_learning_robot_body(self.robot_x, self.robot_y, self.robot_angle, self.robot_color)
            arcade.draw_text("RL ROBOT\nlearning agent", self.robot_x, self.robot_y + 36, arcade.color.BLACK, 8, bold=True, anchor_x="center", multiline=True, align="center", width=90)
            self.draw_info_panel()

    def on_update(self, delta_time):
        self.update_dynamic_actors()

        if self.rl_enabled:
            self.rl_control_step()
        else:
            intended_dx, intended_dy = self.build_manual_input_vector()
            if intended_dx != 0 or intended_dy != 0:
                # Manual override should allow the driver to move again after a red stop.
                self.robot_speed = ROBOT_BASE_SPEED
                safe_dx, safe_dy = self.build_safety_vector_for_robot(intended_dx, intended_dy)
                moved = self.try_move_robot(safe_dx, safe_dy)
                if not moved:
                    self.try_move_robot(intended_dx, intended_dy)

        self.resolve_robot_dynamic_conflicts()
        self.update_robot_risk_state()
        self.check_goal()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT:
            if self.rl_enabled:
                self.rl_enabled = False
                self.set_robot_message("Manual control enabled", INFO_COLOR)
            self.left_pressed = True
        elif key == arcade.key.RIGHT:
            if self.rl_enabled:
                self.rl_enabled = False
                self.set_robot_message("Manual control enabled", INFO_COLOR)
            self.right_pressed = True
        elif key == arcade.key.UP:
            if self.rl_enabled:
                self.rl_enabled = False
                self.set_robot_message("Manual control enabled", INFO_COLOR)
            self.up_pressed = True
        elif key == arcade.key.DOWN:
            if self.rl_enabled:
                self.rl_enabled = False
                self.set_robot_message("Manual control enabled", INFO_COLOR)
            self.down_pressed = True
        elif key == arcade.key.R:
            self.rl_enabled = not self.rl_enabled
            mode = "RL enabled" if self.rl_enabled else "Manual override"
            self.set_robot_message(mode, INFO_COLOR)
        elif key == arcade.key.E:
            self.export_marking_report()

    def on_key_release(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.left_pressed = False
        elif key == arcade.key.RIGHT:
            self.right_pressed = False
        elif key == arcade.key.UP:
            self.up_pressed = False
        elif key == arcade.key.DOWN:
            self.down_pressed = False


def main():
    WarehouseSimulation()
    arcade.run()


if __name__ == "__main__":
    main()

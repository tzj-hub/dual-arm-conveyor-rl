"""Simulation and task parameters, in metres and seconds."""

from dataclasses import asdict, dataclass
import math
from numbers import Integral


@dataclass(frozen=True)
class Config:
    schema_version: int = 4
    object_family: str = "flat_prisms"
    visual_quality: str = "balanced"
    total_objects: int | None = None
    pool_size: int = 64
    bin_capacity: int = 12
    auto_unload: bool = True
    deep_bins: bool = False
    motion_mode: str = "optimized"
    strategy: str = "baseline"
    use_dispatch_preferences: bool = True
    direct_retask: bool = False
    nearest_bin_slot: bool = False
    responsive_release: bool = False
    direct_bin_release: bool = False
    extended_pick_deadline: bool = False
    active_recovery: bool = False
    future_planning: bool = False
    safe_preposition: bool = False
    planned_return: bool = False
    guarded_descent: bool = False
    stable_completion_reward: bool = False
    cooperative_motion: bool = False
    cooperative_drop_cost: bool = False
    downstream_completion: bool = False
    num_arms: int = 2
    gripper_mode: str = "fingers"
    grasp_offset: float = -0.009
    belt_speed: float = 0.1
    belt_profile: str = "constant"
    belt_speed_amplitude: float = 0.1
    belt_speed_period: float = 12.0
    belt_max_acceleration: float = 0.08
    spacing: float = 0.45
    spacing_jitter: float = 0.12
    arrival_mode: str = "auto"
    supply_lanes: int = 1
    lane_offset: float = 0.065
    lane_jitter: float = 0.012
    min_object_gap: float = 0.0
    physics_dt: float = 0.005
    decision_dt: float = 0.1
    belt_top: float = 0.205
    belt_half_length: float = 1.15
    belt_half_width: float = 0.16
    cube_size: float = 0.05
    cube_mass: float = 0.035
    lateral_range: float = 0.065
    yaw_range: float = math.pi
    arrival_jitter: float = 1.2
    close_time: float = 0.40
    settle_time: float = 0.35
    max_time: float | None = None
    control_speed: float = 0.45
    seed: int = 0
    sensor_mode: str = "vision"
    camera_rig: str = "overhead"
    camera_width: int = 480
    camera_height: int = 360
    camera_hz: float = 10.0
    camera_latency: float = 0.08
    depth_noise: float = 0.001
    camera_dropout: float = 0.02
    wrist_camera_mode: str = "off"
    wrist_camera_width: int = 320
    wrist_camera_height: int = 240
    wrist_camera_hz: float = 6.0
    wrist_max_correction: float = 0.008
    track_max_age: float = 0.5
    grasp_coast_time: float = 1.5

    def __post_init__(self):
        for key in (
            "belt_speed",
            "belt_speed_period",
            "belt_max_acceleration",
            "spacing",
            "physics_dt",
            "decision_dt",
            "cube_size",
            "cube_mass",
            "control_speed",
            "belt_top",
            "belt_half_length",
            "belt_half_width",
            "close_time",
            "settle_time",
            "camera_hz",
            "wrist_camera_hz",
            "wrist_max_correction",
            "track_max_age",
            "grasp_coast_time",
            "lane_offset",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.total_objects is not None and (
            isinstance(self.total_objects, bool)
            or not isinstance(self.total_objects, Integral)
            or self.total_objects < 1
        ):
            raise ValueError(
                "total_objects must be None (continuous) or a positive integer"
            )
        if not isinstance(self.deep_bins, bool):
            raise ValueError("deep_bins must be bool")
        if not isinstance(self.use_dispatch_preferences, bool):
            raise ValueError("use_dispatch_preferences must be bool")
        if not isinstance(self.extended_pick_deadline, bool):
            raise ValueError("extended_pick_deadline must be bool")
        if self.extended_pick_deadline and (
            self.num_arms != 4
            or self.strategy != "adaptive"
            or self.gripper_mode != "fingers"
        ):
            raise ValueError("extended_pick_deadline requires four adaptive finger arms")
        if not isinstance(self.downstream_completion, bool):
            raise ValueError("downstream_completion must be bool")
        if self.downstream_completion and (
            self.num_arms != 4
            or self.strategy != "adaptive"
            or self.gripper_mode != "fingers"
        ):
            raise ValueError("downstream_completion requires four adaptive finger arms")
        for key in ("direct_retask", "nearest_bin_slot", "responsive_release", "direct_bin_release", "cooperative_motion", "cooperative_drop_cost"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be bool")
            if getattr(self, key) and (
                self.num_arms != 4
                or self.strategy != "adaptive"
                or self.motion_mode != "optimized"
                or self.gripper_mode != "fingers"
                or not self.deep_bins
            ):
                raise ValueError(
                    f"{key} requires four adaptive finger arms, optimized motion and deep bins"
                )
        for key in ("active_recovery", "future_planning", "safe_preposition", "planned_return", "stable_completion_reward"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be bool")
            if getattr(self, key) and (
                self.num_arms != 4 or self.strategy != "adaptive"
                or self.motion_mode != "optimized" or self.gripper_mode != "fingers"
                or not self.deep_bins
            ):
                raise ValueError(f"{key} requires four adaptive finger arms, optimized motion and deep bins")
        if not isinstance(self.guarded_descent, bool):
            raise ValueError("guarded_descent must be bool")
        if self.guarded_descent and (self.strategy != "adaptive" or self.gripper_mode != "fingers"):
            raise ValueError("guarded_descent requires adaptive finger control")
        if self.safe_preposition and not self.future_planning:
            raise ValueError("safe_preposition requires future_planning")
        if self.planned_return and not self.future_planning:
            raise ValueError("planned_return requires future_planning")
        if self.planned_return and not self.direct_retask:
            raise ValueError("planned_return requires direct_retask")
        for key in ("cooperative_motion", "cooperative_drop_cost"):
            if getattr(self, key) and not (self.nearest_bin_slot and self.direct_retask):
                raise ValueError(f"{key} requires nearest_bin_slot and direct_retask")
        if self.deep_bins and (self.num_arms != 4 or self.auto_unload):
            raise ValueError("deep_bins require four arms and auto_unload=False")
        maximum_bin_capacity = (
            512 if self.deep_bins else (24 if self.num_arms == 4 else 12)
        )
        for key, lo, hi in (
            ("pool_size", 8, 1024 if self.deep_bins else 256),
            ("bin_capacity", 1, maximum_bin_capacity),
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not lo <= value <= hi
            ):
                raise ValueError(f"{key} must be an integer in [{lo},{hi}]")
        if self.num_arms == 4 and self.bin_capacity % 2:
            raise ValueError(
                "four-arm bin_capacity must be even for two equal station zones"
            )
        required_pool = (
            self.total_objects
            if self.deep_bins and self.total_objects is not None
            else 2 * self.bin_capacity + 8
        )
        if self.pool_size < required_pool:
            raise ValueError(
                "deep-bin finite batches require pool_size >= total_objects; sorted objects remain physical"
                if self.deep_bins and self.total_objects is not None
                else "pool_size must cover both bins plus eight in-flight objects"
            )
        if self.motion_mode not in ("optimized", "legacy"):
            raise ValueError("motion_mode must be optimized or legacy")
        if self.strategy not in ("baseline", "adaptive"):
            raise ValueError("strategy must be baseline or adaptive")
        if self.gripper_mode not in ("fingers", "vacuum"):
            raise ValueError("gripper_mode must be fingers or vacuum")
        if self.gripper_mode == "vacuum" and (
            self.strategy != "adaptive"
            or self.motion_mode != "optimized"
            or self.wrist_camera_mode != "off"
        ):
            raise ValueError(
                "vacuum requires adaptive strategy, optimized motion and wrist cameras off"
            )
        if (
            isinstance(self.num_arms, bool)
            or not isinstance(self.num_arms, Integral)
            or self.num_arms not in (2, 4)
        ):
            raise ValueError("num_arms must be 2 or 4")
        if self.num_arms == 4 and (
            self.strategy != "adaptive"
            or self.motion_mode != "optimized"
            or self.wrist_camera_mode != "off"
        ):
            raise ValueError(
                "four arms require adaptive strategy, optimized motion and wrist cameras off"
            )
        if self.belt_profile not in ("constant", "sine", "random"):
            raise ValueError("belt_profile must be constant, sine or random")
        if self.arrival_mode not in ("auto", "time", "distance"):
            raise ValueError("arrival_mode must be auto, time or distance")
        if (
            isinstance(self.supply_lanes, bool)
            or not isinstance(self.supply_lanes, Integral)
            or self.supply_lanes not in (1, 2)
        ):
            raise ValueError("supply_lanes must be 1 or 2")
        if not math.isfinite(self.lane_jitter) or self.lane_jitter < 0:
            raise ValueError("lane_jitter must be finite and nonnegative")
        if not math.isfinite(self.min_object_gap) or self.min_object_gap < 0:
            raise ValueError("min_object_gap must be finite and nonnegative")
        if self.min_object_gap > 0 and self.supply_lanes != 2:
            raise ValueError("min_object_gap requires two-lane supply")
        if self.supply_lanes == 2:
            if not self.spatial_supply:
                raise ValueError("two-lane supply requires distance-based arrivals")
            if self.lane_offset <= self.lane_jitter:
                raise ValueError("lane_jitter must be smaller than lane_offset")
            # The largest catalogue footprint has a 41 mm bounding radius;
            # leave another 5 mm at each belt edge for random yaw.
            if self.lane_offset + self.lane_jitter + 0.046 >= self.belt_half_width:
                raise ValueError(
                    "two-lane placement leaves insufficient belt clearance"
                )
        if (
            not math.isfinite(self.belt_speed_amplitude)
            or self.belt_speed_amplitude < 0
        ):
            raise ValueError("belt_speed_amplitude must be finite and nonnegative")
        if (
            self.belt_profile != "constant"
            and self.belt_speed_amplitude >= self.belt_speed
        ):
            raise ValueError(
                "variable belt speed must remain positive: amplitude < mean speed"
            )
        if not isinstance(self.auto_unload, bool):
            raise ValueError("auto_unload must be bool")
        if (
            not math.isfinite(self.grasp_offset)
            or not -0.012 <= self.grasp_offset <= 0.012
        ):
            raise ValueError("grasp_offset outside calibrated finger geometry")
        for key in (
            "lateral_range",
            "yaw_range",
            "arrival_jitter",
            "spacing_jitter",
            "camera_latency",
            "depth_noise",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if (
            not isinstance(self.seed, Integral)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if (
            abs(
                round(self.decision_dt / self.physics_dt) * self.physics_dt
                - self.decision_dt
            )
            > 1e-8
        ):
            raise ValueError("decision_dt must be an integer multiple of physics_dt")
        if self.lateral_range + self.cube_size * 0.71 >= self.belt_half_width:
            raise ValueError("lateral_range leaves insufficient room on the belt")
        if self.sensor_mode not in ("vision", "oracle"):
            raise ValueError("sensor_mode must be vision or oracle")
        if self.camera_rig not in ("overhead", "dual", "triple"):
            raise ValueError("camera_rig must be overhead, dual or triple")
        if self.wrist_camera_mode not in ("off", "observe", "refine"):
            raise ValueError("wrist_camera_mode must be off, observe or refine")
        if self.wrist_camera_mode != "off" and (
            self.sensor_mode != "vision" or self.camera_rig != "overhead"
        ):
            raise ValueError("Wrist cameras require vision and the overhead camera rig")
        if self.wrist_camera_mode == "refine" and self.motion_mode != "optimized":
            raise ValueError(
                "Wrist refinement requires optimized collision-checked motion"
            )
        if self.wrist_max_correction > 0.012:
            raise ValueError("wrist_max_correction must not exceed 12 mm")
        if self.schema_version != 4:
            raise ValueError("Only scenario schema_version 4 is supported")
        if self.object_family != "flat_prisms":
            raise ValueError("object_family must be flat_prisms")
        if self.visual_quality not in ("balanced", "high"):
            raise ValueError("visual_quality must be balanced or high")
        for key in (
            "camera_width",
            "camera_height",
            "wrist_camera_width",
            "wrist_camera_height",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 64:
                raise ValueError(f"{key} must be an integer >=64")
        if not math.isfinite(self.camera_dropout) or not 0 <= self.camera_dropout < 1:
            raise ValueError("camera_dropout must lie in [0,1)")
        if self.spacing < self.cube_size * 2:
            raise ValueError("spacing must be at least twice cube_size")
        if self.max_time is not None and (
            not math.isfinite(self.max_time) or self.max_time <= 0
        ):
            raise ValueError("max_time must be finite and positive")

    @property
    def n_objects(self):
        return self.pool_size

    @property
    def spatial_supply(self):
        return self.arrival_mode == "distance" or (
            self.arrival_mode == "auto"
            and (self.strategy == "adaptive" or self.belt_profile != "constant")
        )

    @property
    def horizon(self):
        if self.max_time is not None:
            return self.max_time
        if self.total_objects is None:
            return math.inf
        if not self.spatial_supply:
            interval = self.spacing / self.belt_speed + self.arrival_jitter
        else:
            minimum_speed = self.belt_speed - (
                self.belt_speed_amplitude if self.belt_profile != "constant" else 0
            )
            interval = max(0.12, self.spacing + self.spacing_jitter) / minimum_speed
        return (self.total_objects - 1) * interval + 40.0

    def to_dict(self):
        return asdict(self)

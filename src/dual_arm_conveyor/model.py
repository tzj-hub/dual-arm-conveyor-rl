"""MuJoCo scene construction using the vendored Menagerie Franka Panda model.

No object is welded or kinematically attached to a gripper. All cube motion after
spawning is produced by MuJoCo contact and gravity. ConveyorEnv requires native
contact surface velocity (MuJoCo >=3.11). Older versions can load the static
scene for robot-only kinematics and contact tests.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .objects import ObjectSpec, make_object_catalog
from .belt_visual import append_belt_markers

ASSET_DIR = Path(__file__).parent / "assets" / "robots" / "panda"
HOME_Q = np.array(
    [0.0, -1.5371305452, 0.0, -2.2684310999, 0.0, 0.7313007661, 0.7853981168]
)
# Four-arm ready pose keeps each tool 20 cm from the belt centre at z=42 cm.
# Calibrated with all four robot collision meshes and both long bins present.
FOUR_READY_Q = np.array(
    [
        0.0,
        -0.667234325104717,
        0.0,
        -2.437645282797769,
        0.0,
        1.7704118008346508,
        0.7853981258130793,
    ]
)
BASE_POSITIONS = np.array([[-0.28, -0.34, 0.0], [0.28, 0.34, 0.0]])
BASE_YAWS = (np.pi / 2, -np.pi / 2)
VACUUM_FACE_OFFSET = 0.040
VACUUM_RADIUS = 0.012


def arm_layout(num_arms=2):
    """Original arms or two opposed pairs outside two longitudinal bins."""
    if num_arms == 2:
        return BASE_POSITIONS.copy(), BASE_YAWS
    if num_arms == 4:
        return np.array(
            [
                [-0.55, -0.55, 0.0],
                [-0.55, 0.55, 0.0],
                [0.55, -0.55, 0.0],
                [0.55, 0.55, 0.0],
            ]
        ), (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2)
    raise ValueError("num_arms must be 2 or 4")


@dataclass(frozen=True)
class ModelIds:
    arm_joints: np.ndarray
    arm_qpos: np.ndarray
    arm_dofs: np.ndarray
    arm_actuators: np.ndarray
    finger_joints: np.ndarray
    finger_qpos: np.ndarray
    finger_dofs: np.ndarray
    gripper_actuators: np.ndarray
    ee_sites: np.ndarray
    finger_bodies: np.ndarray
    finger_touch_sensors: np.ndarray
    finger_touch_adr: np.ndarray
    overhead_camera: int
    cube_bodies: np.ndarray
    cube_joints: np.ndarray
    cube_qpos: np.ndarray
    cube_dofs: np.ndarray
    cube_geoms: np.ndarray
    belt_geom: int
    bin_centers: np.ndarray
    bin_geoms: np.ndarray
    bin_floor_z: float
    bin_rim_z: float
    bin_half_sizes: np.ndarray
    bin_gate_joints: np.ndarray
    bin_gate_qpos: np.ndarray
    bin_gate_dofs: np.ndarray
    bin_gate_actuators: np.ndarray
    bin_gate_open_position: float
    bin_receiver_floor_z: float
    bin_outlet_z: float
    object_specs: tuple[ObjectSpec, ...]
    robot_geom_arm: np.ndarray
    native_surface_velocity: bool
    home_qpos: np.ndarray
    home_ee: np.ndarray
    arm_base_positions: np.ndarray
    vacuum_sites: np.ndarray | None = None
    vacuum_geoms: np.ndarray | None = None
    vacuum_bodies: np.ndarray | None = None
    vacuum_face_offset: float = VACUUM_FACE_OFFSET
    vacuum_radius: float = VACUUM_RADIUS
    bin_static_floor: bool = False
    ground_z: float = 0.0


def _numbers(values):
    return " ".join(f"{float(v):.9g}" for v in values)


def _copy_robot(
    root: ET.Element,
    arm: int,
    wrist_camera_mode="off",
    *,
    base_position=None,
    base_yaw=None,
    gripper_mode="fingers",
):
    robot = ET.parse(ASSET_DIR / "panda.xml").getroot()
    prefix = f"r{arm}/"
    sensors = ET.SubElement(robot, "sensor")
    for side in ("left", "right"):
        # The volume covers the inner fingertip contact patch (including the
        # collision mesh) and measures summed normal load, like a tactile pad.
        site = robot.find(f".//site[@name='{side}_pad_site']")
        site.set("type", "box")
        site.set("pos", "0 .003 .043")
        site.set("size", ".012 .014 .018")
        site.set("rgba", "0 0 0 0")
        ET.SubElement(sensors, "touch", name=f"{side}_touch", site=f"{side}_pad_site")
        # The inherited five overlapping pad boxes form a 1.5 mm inner
        # ledge that can trap a cylinder on one fully opened finger. Use
        # one continuous physical pad with the same outer envelope.
        finger = robot.find(f".//body[@name='{side}_finger']")
        for geom in list(finger.findall("geom")):
            collision_class = geom.get("class", "")
            if collision_class.startswith("fingertip_pad_collision_"):
                finger.remove(geom)
            elif collision_class == "collision":
                geom.set("friction", "1 .015 .001")
        ET.SubElement(
            finger,
            "geom",
            name=f"{side}_flat_pad",
            **{
                "class": "collision",
                "type": "box",
                "size": ".0085 .00475 .0085",
                "pos": "0 .00475 .0445",
                "friction": "1 .015 .001",
                "condim": "4",
            },
        )
    # Mesh elements without explicit names inherit their file basename.
    for mesh in robot.findall("asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.attrib["file"]).stem))
        mesh.set("file", str(ASSET_DIR / "assets" / mesh.attrib["file"]))
    references = {
        "name",
        "class",
        "childclass",
        "material",
        "mesh",
        "joint",
        "joint1",
        "joint2",
        "body1",
        "body2",
        "tendon",
        "site",
    }
    for node in robot.iter():
        for key in references.intersection(node.attrib):
            node.set(key, prefix + node.attrib[key])
    link0 = robot.find("worldbody/body")
    link0.set(
        "pos", _numbers(BASE_POSITIONS[arm] if base_position is None else base_position)
    )
    yaw = BASE_YAWS[arm] if base_yaw is None else base_yaw
    link0.set("quat", _numbers([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
    # The original site sits 7 mm beyond the pad centre. Positioning the site at
    # cube centre therefore places the pads slightly above centre, safely away
    # from the belt while retaining the existing Panda tool convention.
    for body in robot.findall(".//body"):
        if body.get("name", "").endswith("wrist_camera_mount"):
            for parent in robot.iter():
                if body in list(parent):
                    parent.remove(body)
                    break
    if gripper_mode == "vacuum":
        hand = robot.find(f".//body[@name='{prefix}hand']")
        # A fixed rigid tool body contributes real mass/inertia to the arm.
        # Its seal face extends 40 mm beyond ee_site, ahead of the open fingers.
        # The contact driver may bind only vacuum_cup/workpiece contacts; the
        # stem remains ordinary collision geometry.
        tool = ET.SubElement(hand, "body", name=prefix + "vacuum_tool", pos="0 0 .110")
        ET.SubElement(
            tool,
            "geom",
            name=prefix + "vacuum_stem",
            type="cylinder",
            pos="0 0 -.0095",
            size=".004 .0455",
            mass=".020",
            rgba=".48 .50 .53 1",
            contype="1",
            conaffinity="1",
            group="2",
            friction=".8 .002 .0001",
            condim="4",
        )
        ET.SubElement(
            tool,
            "geom",
            name=prefix + "vacuum_cup",
            type="cylinder",
            pos="0 0 .038",
            size=f"{VACUUM_RADIUS} .002",
            mass=".010",
            rgba=".10 .12 .13 1",
            contype="1",
            conaffinity="1",
            group="2",
            friction=".8 .002 .0001",
            condim="4",
            solref=".008 1",
        )
        ET.SubElement(
            tool,
            "site",
            name=prefix + "vacuum_site",
            type="cylinder",
            pos=f"0 0 {VACUUM_FACE_OFFSET}",
            size=f"{VACUUM_RADIUS} .0001",
            rgba="0 0 0 0",
            group="4",
        )
    if wrist_camera_mode != "off":
        hand = robot.find(f".//body[@name='{prefix}hand']")
        # Optical centre is on the +x side, perpendicular to jaw motion (+/-y).
        # Look inward between the fingers at hand-local z=.12. This higher
        # optical origin keeps the top face observable later in the descent.
        optical_position = np.array([0.11, 0.0, 0.0])
        camera_z = np.array([0.11, 0.0, -0.12])
        camera_z /= np.linalg.norm(camera_z)
        camera_x = np.array([0.0, 1.0, 0.0])
        camera_y = np.cross(camera_z, camera_x)
        rotation = np.column_stack((camera_x, camera_y, camera_z))
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
        # This mount is a visual sensor fixture: no added mass, inertia or
        # collision geometry. The camera follows the real hand body transform.
        mount = ET.SubElement(
            hand,
            "body",
            name=prefix + "wrist_camera_mount",
            pos=_numbers(optical_position + 0.016 * camera_z),
            quat=_numbers(quaternion),
        )
        visual_only = dict(contype="0", conaffinity="0", density="0", group="2")
        ET.SubElement(
            mount,
            "geom",
            name=prefix + "wrist_camera_housing",
            type="box",
            size=".018 .025 .010",
            rgba=".12 .14 .16 1",
            **visual_only,
        )
        ET.SubElement(
            mount,
            "geom",
            name=prefix + "wrist_camera_lens",
            type="cylinder",
            pos="0 0 -.0115",
            size=".006 .002",
            rgba=".025 .03 .035 1",
            **visual_only,
        )
        ET.SubElement(
            hand,
            "geom",
            name=prefix + "wrist_camera_bracket",
            type="capsule",
            fromto=".028 0 .025 .121 0 -.012",
            size=".004",
            rgba=".22 .24 .26 1",
            **visual_only,
        )
        ET.SubElement(
            mount,
            "camera",
            name=prefix + "wrist_rgbd",
            pos="0 0 -.016",
            fovy="70",
        )
    actuators = robot.find("actuator")
    for i, actuator in enumerate(actuators):
        if i < 7:
            kp = 1800 if i < 4 else 700
            kd = 85 if i < 4 else 35
            actuator.set("gainprm", str(kp))
            actuator.set("biasprm", f"0 {-kp} {-kd}")
        else:
            actuator.set("gainprm", "600")
            actuator.set("biasprm", "0 -600 -10")
            actuator.set("forcerange", "-30 30")
    for section in (
        "default",
        "asset",
        "worldbody",
        "tendon",
        "equality",
        "actuator",
        "contact",
        "sensor",
    ):
        source = robot.find(section)
        target = root.find(section)
        if target is None:
            target = ET.SubElement(root, section)
        for node in source:
            if section == "worldbody" and node.tag == "light":
                continue
            target.append(deepcopy(node))


def scene_xml(config) -> tuple[str, np.ndarray, bool]:
    """Return a portable XML string and bin metadata for a config-like object."""
    dt = getattr(config, "physics_dt", 0.002)
    n_objects = getattr(config, "n_objects", 10)
    half_length = getattr(config, "belt_half_length", 1.15)
    half_width = getattr(config, "belt_half_width", 0.16)
    belt_top = getattr(config, "belt_top", 0.205)
    deep_bins = bool(getattr(config, "deep_bins", False))
    floor_z = belt_top + 0.05 - 0.80 if deep_bins else belt_top - 0.07
    ground_z = floor_z - 0.08 if deep_bins else 0.0
    object_specs = make_object_catalog(n_objects, getattr(config, "seed", 0))
    root = ET.Element("mujoco", model="dual_panda_conveyor")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    # Higher tangential impedance reduces soft-contact grasp creep. Avoid
    # friction-only noslip postprocessing, which can overconstrain a grasp.
    # This retains finite actuator forces and normal contact compliance.
    ET.SubElement(
        root,
        "option",
        timestep=str(dt),
        integrator="implicitfast",
        solver="Newton",
        iterations="50",
        noslip_iterations="0",
        impratio="20",
        tolerance="1e-9",
        cone="elliptic",
        gravity="0 0 -9.81",
    )
    visual = ET.SubElement(root, "visual")
    if getattr(config, "wrist_camera_mode", "off") != "off":
        # MuJoCo scales znear by scene extent (~9.5m here): the original .01
        # clips targets within 9.5cm. Wrist views need a ~9.5mm near plane.
        ET.SubElement(visual, "map", znear=".001")
    ET.SubElement(
        visual, "quality", offsamples="4" if config.visual_quality == "high" else "0"
    )
    ET.SubElement(
        visual,
        "global",
        offwidth="1280",
        offheight="720",
        azimuth="135",
        elevation="-30",
    )
    if deep_bins:
        # Resident parts keep contacts and dynamics. Reserve arena space for
        # hundreds of simultaneous contacts, without disabling settled parts.
        ET.SubElement(root, "size", memory="256M")
        # Inactive pool slots are parked far away; they must not turn the
        # work-cell camera clipping planes into a 100-metre scene.
        ET.SubElement(root, "statistic", center="0 0 0", extent="4")
    else:
        ET.SubElement(root, "size", njmax="2000", nconmax="400")
    ET.SubElement(root, "default")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="floor_tex",
        type="2d",
        builtin="checker",
        width="256",
        height="256",
        rgb1=".16 .19 .23",
        rgb2=".20 .23 .27",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_mat",
        texture="floor_tex",
        texrepeat="4 4",
        reflectance=".1",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(world, "light", pos="1 1 2", dir="-.3 -.2 -1", diffuse=".5 .5 .5")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        pos=f"0 0 {ground_z}",
        size="3 3 .05",
        material="floor_mat",
    )
    ET.SubElement(
        world,
        "camera",
        name="overview",
        pos="2.2 -2.6 2.2",
        xyaxes=".763 .646 0 -.365 .431 .826",
    )
    # Optical frame: x right (+world x), y up (+world y), camera looks -z.
    # RGB and metric depth share these fixed, calibrated extrinsics/intrinsics.
    ET.SubElement(
        world, "camera", name="overhead", pos="0 0 1.8", quat="1 0 0 0", fovy="60"
    )
    # Opposing fixed RGB-D views stay 24 degrees from vertical so flat object
    # tops remain visible. The upstream shift includes a whole object at x=-.88
    # as well as the downstream grasp window in the default 4:3 sensor image.
    # All rigs share this same model; camera_rig only chooses perception inputs.
    for name, side in (("oblique_south", -1), ("oblique_north", 1)):
        camera_z = np.array([0.0, side * 0.5, 1.35 - 0.23])
        camera_z /= np.linalg.norm(camera_z)
        camera_y = np.cross(camera_z, [1.0, 0.0, 0.0])
        ET.SubElement(
            world,
            "camera",
            name=name,
            pos=_numbers([-0.10, side * 0.50, 1.35]),
            xyaxes=_numbers([1.0, 0.0, 0.0, *camera_y]),
            fovy="60",
        )
    belt = ET.SubElement(
        world,
        "geom",
        name="belt",
        type="box",
        pos=f"0 0 {belt_top - 0.055}",
        size=f"{half_length} {half_width} .055",
        rgba=".12 .17 .19 1",
        friction="1.0 .01 .001",
        condim="3",
        solref=".006 1",
    )
    # surfacevel was added in MuJoCo 3.11. Earlier versions can load this
    # static scene for robot-only tests; ConveyorEnv rejects them at startup.
    version = tuple(int(part) for part in mujoco.__version__.split(".")[:2])
    native = version >= (3, 11)
    if native:
        belt.set("surfacevel", f"{getattr(config, 'belt_speed', 0.1)} 0 0")
    append_belt_markers(world, config)
    for side in (-1, 1):
        ET.SubElement(
            world,
            "geom",
            name=f"rail_{side}",
            type="box",
            pos=f"0 {side * (half_width + 0.016)} {belt_top + 0.008}",
            size=f"{half_length} .012 .025",
            rgba=".5 .55 .6 1",
        )
    for x in (-0.9, 0.9):
        for y in (-0.13, 0.13):
            top = belt_top - 0.11 if deep_bins else 0.15
            ET.SubElement(
                world,
                "geom",
                name=f"belt_leg_{x}_{y}",
                type="box",
                pos=f"{x} {y} {(top + ground_z) / 2}",
                size=f".025 .025 {(top - ground_z) / 2}",
                rgba=".35 .4 .45 1",
                contype="1" if deep_bins else "0",
                conaffinity="1" if deep_bins else "0",
            )
    # Two shared hoppers with physically sliding floor gates. Long floors slide
    # along x: sliding them outward across y would hit the four robot bases.
    rim_z = belt_top + 0.05
    receiver_floor_z = 0.015
    four_arms = getattr(config, "num_arms", 2) == 4
    bin_y = 0.28 if four_arms else 0.33
    bin_half_x, bin_half_y = (0.8 if four_arms else 0.15), 0.10
    gate_travel = 2 * bin_half_x + 0.03 if four_arms else 0.23
    # The long-bin boundaries model metal collection hardware. Priority also
    # selects their existing friction/solimp for these contacts; workpiece-to-
    # belt and workpiece-to-workpiece material differences remain unchanged.
    # The fast launcher uses 2.5 ms steps so this 8 ms contact time constant is
    # above MuJoCo's default two-step solver safety floor.
    bin_contact = {"priority": "1", "solref": ".008 1"} if four_arms else {}
    bins = np.array([[0.0, -bin_y, floor_z], [0.0, bin_y, floor_z]])
    walls = (
        (-bin_half_x - 0.006, 0, 0.006, bin_half_y + 0.012),
        (bin_half_x + 0.006, 0, 0.006, bin_half_y + 0.012),
        (0, -bin_half_y - 0.006, bin_half_x, 0.006),
        (0, bin_half_y + 0.006, bin_half_x, 0.006),
    )
    gate_actuators = ET.SubElement(root, "actuator")
    for color in range(2):
        x, y, z = bins[color]
        side = -1 if color == 0 else 1
        neutral = ".36 .39 .42 1"
        gate = ET.SubElement(
            world, "body", name=f"bin_gate_{color}", pos=f"{x} {y} {z - 0.006}"
        )
        ET.SubElement(
            gate,
            "joint",
            name=f"bin_gate_joint_{color}",
            type="slide",
            axis="1 0 0" if four_arms else f"0 {side} 0",
            range=f"0 {gate_travel}",
            damping="8",
            armature=".05",
        )
        if deep_bins:
            # The load-bearing floor is fixed to world, not to this dummy.
            # Keep legacy joint/actuator IDs available, but lock this inertial
            # compatibility body and give its actuator zero transmission.
            ET.SubElement(
                gate,
                "inertial",
                pos="0 0 0",
                mass=".01",
                diaginertia=".00001 .00001 .00001",
            )
            equality = root.find("equality")
            if equality is None:
                equality = ET.SubElement(root, "equality")
            ET.SubElement(
                equality,
                "joint",
                name=f"static_bin_lock_{color}",
                joint1=f"bin_gate_joint_{color}",
                polycoef="0 0 0 0 0",
            )
        ET.SubElement(
            world if deep_bins else gate,
            "geom",
            name=f"bin_{color}",
            type="box",
            **({"pos": f"{x} {y} {z - 0.01}"} if deep_bins else {}),
            size=f"{bin_half_x} {bin_half_y} {'.01' if deep_bins else '.006'}",
            mass=f"{0.8 * bin_half_x / 0.15:.12g}",
            rgba=neutral,
            friction=".7 .01 .001",
            # Rounded objects need rolling friction to dissipate rocking after
            # release. Keep this local to the hopper floor, not the gripper/belt.
            condim="6",
            **bin_contact,
        )
        if deep_bins:
            ET.SubElement(
                gate_actuators,
                "motor",
                name=f"bin_gate_actuator_{color}",
                joint=f"bin_gate_joint_{color}",
                gear="0",
            )
        else:
            ET.SubElement(
                gate_actuators,
                "position",
                name=f"bin_gate_actuator_{color}",
                joint=f"bin_gate_joint_{color}",
                kp="1000",
                kv=f"{60 * np.sqrt(bin_half_x / 0.15):.12g}",
                ctrlrange=f"0 {gate_travel}",
                forcerange="-150 150",
            )
        for k, (dx, dy, sx, sy) in enumerate(walls):
            ET.SubElement(
                world,
                "geom",
                name=f"binwall_{color}_{k}",
                type="box",
                pos=f"{x + dx} {y + dy} {(floor_z + rim_z) / 2}",
                size=f"{sx} {sy} {(rim_z - floor_z) / 2}",
                rgba=neutral,
                **bin_contact,
            )
        if deep_bins:
            # Short solid feet support the fixed 20 mm floor above the lower
            # ground. All supports remain ordinary collision geometry.
            foot_top = floor_z - 0.02
            for foot, (dx, dy) in enumerate(
                ((-0.70, -0.06), (-0.70, 0.06), (0.70, -0.06), (0.70, 0.06))
            ):
                ET.SubElement(
                    world,
                    "geom",
                    name=f"bin_foot_{color}_{foot}",
                    type="box",
                    pos=f"{x + dx} {y + dy} {(ground_z + foot_top) / 2}",
                    size=f".035 .025 {(foot_top - ground_z) / 2}",
                    rgba=".23 .25 .27 1",
                )
        else:
            # Objects leaving a sliding gate enter an enclosed receiver. Deep
            # bins have no outlet or receiver: their contents remain physical.
            receiver_top = floor_z - 0.016
            ET.SubElement(
                world,
                "geom",
                name=f"receiver_floor_{color}",
                type="box",
                pos=f"{x} {y} {receiver_floor_z / 2}",
                size=f"{bin_half_x} {bin_half_y} {receiver_floor_z / 2}",
                rgba=".23 .25 .27 1",
                friction="1 .01 .001",
            )
            for k, (dx, dy, sx, sy) in enumerate(walls):
                ET.SubElement(
                    world,
                    "geom",
                    name=f"receiver_wall_{color}_{k}",
                    type="box",
                    pos=f"{x + dx} {y + dy} {(receiver_floor_z + receiver_top) / 2}",
                    size=f"{sx} {sy} {(receiver_top - receiver_floor_z) / 2}",
                    rgba=".23 .25 .27 1",
                )
        label_y = y + side * (bin_half_y + 0.02)
        rgba = ".75 .12 .08 1" if color == 0 else ".08 .22 .75 1"
        ET.SubElement(
            world,
            "geom",
            name=f"bin_label_{color}",
            type="box",
            pos=f"0 {label_y} {rim_z - 0.02}",
            size=".045 .008 .004",
            rgba=rgba,
            contype="0",
            conaffinity="0",
        )
    bases, yaws = arm_layout(getattr(config, "num_arms", 2))
    for arm in range(len(bases)):
        if deep_bins:
            # Robot world poses and ready trajectories stay unchanged; steel
            # pedestals bridge the lowered ground to the existing z=0 mounts.
            ET.SubElement(
                world,
                "geom",
                name=f"robot_pedestal_{arm}",
                type="box",
                pos=f"{bases[arm, 0]} {bases[arm, 1]} {ground_z / 2}",
                size=f".12 .12 {-ground_z / 2}",
                rgba=".28 .32 .37 1",
            )
        _copy_robot(
            root,
            arm,
            getattr(config, "wrist_camera_mode", "off"),
            base_position=bases[arm],
            base_yaw=yaws[arm],
            gripper_mode=getattr(config, "gripper_mode", "fingers"),
        )
    for i, spec in enumerate(object_specs):
        # Park unspawned objects below the floor and outside the cell. The env
        # disables their collisions until a scheduled upstream spawn.
        body = ET.SubElement(
            world, "body", name=f"cube_{i}", pos=f"{-2 - i * 0.1} 0 -.5", gravcomp="1"
        )
        ET.SubElement(body, "freejoint", name=f"cube_joint_{i}")
        geometry = {"type": spec.shape, "size": _numbers(spec.geom_size)}
        if spec.prism_sides is not None:
            vertices, faces = spec.prism_mesh()
            mesh_name = f"object_mesh_{i}"
            ET.SubElement(
                asset,
                "mesh",
                name=mesh_name,
                vertex=_numbers(vertices.ravel()),
                face=" ".join(str(int(v)) for v in faces.ravel()),
            )
            geometry = {"type": "mesh", "mesh": mesh_name}
        ET.SubElement(
            body,
            "geom",
            name=f"cube_geom_{i}",
            **geometry,
            mass=str(spec.mass),
            rgba=(".95 .2 .13 1" if i % 2 == 0 else ".12 .4 .95 1"),
            friction=_numbers(spec.friction),
            condim="4",
            solref=_numbers(spec.solref),
            solimp=".95 .99 .001",
        )
    return ET.tostring(root, encoding="unicode"), bins, native


def build_scene(config) -> tuple[mujoco.MjModel, ModelIds]:
    xml, bins, native = scene_xml(config)
    model = mujoco.MjModel.from_xml_string(xml)
    num_arms = getattr(config, "num_arms", 2)
    deep_bins = bool(getattr(config, "deep_bins", False))

    def ids(kind, names):
        obj = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
        result = np.array(
            [mujoco.mj_name2id(model, obj, name) for name in names], dtype=int
        )
        if np.any(result < 0):
            raise ValueError(f"Missing {kind}: {names}")
        return result

    joints = np.array(
        [ids("joint", [f"r{a}/joint{j}" for j in range(1, 8)]) for a in range(num_arms)]
    )
    fingers = np.array(
        [
            ids("joint", [f"r{a}/finger_joint{j}" for j in (1, 2)])
            for a in range(num_arms)
        ]
    )
    count = getattr(config, "n_objects", 10)
    cubes = ids("joint", [f"cube_joint_{i}" for i in range(count)])
    gates = ids("joint", [f"bin_gate_joint_{i}" for i in range(2)])
    robot_geom_arm = np.full(model.ngeom, -1, dtype=int)
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        prefix = name.partition("/")[0]
        if (
            prefix.startswith("r")
            and prefix[1:].isdigit()
            and int(prefix[1:]) < num_arms
        ):
            robot_geom_arm[geom_id] = int(prefix[1:])
    startup = mujoco.MjData(model)
    ready_q = FOUR_READY_Q if num_arms == 4 else HOME_Q
    for arm in range(num_arms):
        startup.qpos[model.jnt_qposadr[joints[arm]]] = ready_q
        startup.qpos[model.jnt_qposadr[fingers[arm]]] = 0.04
    mujoco.mj_kinematics(model, startup)
    ee_sites = ids("site", [f"r{a}/ee_site" for a in range(num_arms)])
    result = ModelIds(
        arm_joints=joints,
        arm_qpos=model.jnt_qposadr[joints],
        arm_dofs=model.jnt_dofadr[joints],
        arm_actuators=np.array(
            [
                ids("actuator", [f"r{a}/actuator{j}" for j in range(1, 8)])
                for a in range(num_arms)
            ]
        ),
        finger_joints=fingers,
        finger_qpos=model.jnt_qposadr[fingers],
        finger_dofs=model.jnt_dofadr[fingers],
        gripper_actuators=ids("actuator", [f"r{a}/actuator8" for a in range(num_arms)]),
        ee_sites=ids("site", [f"r{a}/ee_site" for a in range(num_arms)]),
        finger_bodies=np.array(
            [
                ids("body", [f"r{a}/left_finger", f"r{a}/right_finger"])
                for a in range(num_arms)
            ]
        ),
        finger_touch_sensors=np.array(
            [
                ids("sensor", [f"r{a}/left_touch", f"r{a}/right_touch"])
                for a in range(num_arms)
            ]
        ),
        finger_touch_adr=model.sensor_adr[
            np.array(
                [
                    ids("sensor", [f"r{a}/left_touch", f"r{a}/right_touch"])
                    for a in range(num_arms)
                ]
            )
        ],
        overhead_camera=int(ids("camera", ["overhead"])[0]),
        cube_bodies=ids("body", [f"cube_{i}" for i in range(count)]),
        cube_joints=cubes,
        cube_qpos=model.jnt_qposadr[cubes],
        cube_dofs=model.jnt_dofadr[cubes],
        cube_geoms=ids("geom", [f"cube_geom_{i}" for i in range(count)]),
        belt_geom=int(ids("geom", ["belt"])[0]),
        bin_centers=bins,
        bin_geoms=ids("geom", [f"bin_{c}" for c in range(2)]),
        bin_floor_z=float(bins[0, 2]),
        bin_rim_z=float(getattr(config, "belt_top", 0.205) + 0.05),
        bin_half_sizes=np.array([0.8 if num_arms == 4 else 0.15, 0.10]),
        bin_gate_joints=gates,
        bin_gate_qpos=model.jnt_qposadr[gates],
        bin_gate_dofs=model.jnt_dofadr[gates],
        bin_gate_actuators=ids(
            "actuator", [f"bin_gate_actuator_{i}" for i in range(2)]
        ),
        bin_gate_open_position=0.0 if deep_bins else (1.63 if num_arms == 4 else 0.23),
        bin_receiver_floor_z=float(bins[0, 2]) if deep_bins else 0.015,
        bin_outlet_z=float(bins[0, 2] - 0.05),
        object_specs=make_object_catalog(count, getattr(config, "seed", 0)),
        robot_geom_arm=robot_geom_arm,
        native_surface_velocity=native,
        home_qpos=np.tile(ready_q, (num_arms, 1)),
        home_ee=startup.site_xpos[ee_sites].copy(),
        arm_base_positions=arm_layout(num_arms)[0],
        bin_static_floor=deep_bins,
        ground_z=float(bins[0, 2] - 0.08) if deep_bins else 0.0,
        vacuum_sites=(
            ids("site", [f"r{arm}/vacuum_site" for arm in range(num_arms)])
            if getattr(config, "gripper_mode", "fingers") == "vacuum"
            else None
        ),
        vacuum_geoms=(
            ids("geom", [f"r{arm}/vacuum_cup" for arm in range(num_arms)])
            if getattr(config, "gripper_mode", "fingers") == "vacuum"
            else None
        ),
        vacuum_bodies=(
            ids("body", [f"r{arm}/vacuum_tool" for arm in range(num_arms)])
            if getattr(config, "gripper_mode", "fingers") == "vacuum"
            else None
        ),
    )
    return model, result


def reset_robot_pose(model: mujoco.MjModel, data: mujoco.MjData, ids: ModelIds):
    """Set only the robots to their collision-free startup configuration."""
    for arm in range(len(ids.ee_sites)):
        data.qpos[ids.arm_qpos[arm]] = ids.home_qpos[arm]
        data.qpos[ids.finger_qpos[arm]] = 0.04
        data.qvel[ids.arm_dofs[arm]] = 0
        data.qvel[ids.finger_dofs[arm]] = 0
        data.ctrl[ids.arm_actuators[arm]] = ids.home_qpos[arm]
        data.ctrl[ids.gripper_actuators[arm]] = 0.04
    mujoco.mj_forward(model, data)

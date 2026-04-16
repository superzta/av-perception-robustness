import queue
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
try:
    import carla
except ModuleNotFoundError as exc:
    project_root = Path(__file__).resolve().parents[2]
    egg_candidates = sorted((project_root / "carla" / "PythonAPI" / "carla" / "dist").glob("carla-*.egg"))
    if egg_candidates:
        sys.path.append(str(egg_candidates[-1]))
        import carla
    else:
        raise ModuleNotFoundError(
            "Could not import carla module. Ensure CARLA PythonAPI egg exists under "
            "carla/PythonAPI/carla/dist and use the project venv."
        ) from exc


def connect_to_carla(carla_cfg: dict, logger):
    host = carla_cfg.get("host", "127.0.0.1")
    port = int(carla_cfg.get("port", 2000))
    timeout = float(carla_cfg.get("timeout_seconds", 20.0))
    target_map = carla_cfg.get("map", "")

    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()

    if target_map and world.get_map().name != target_map:
        logger.info("Loading map %s", target_map)
        world = client.load_world(target_map)
        # Newly loaded maps can take a few seconds before APIs are responsive.
        for _ in range(30):
            try:
                world.get_map()
                break
            except RuntimeError:
                time.sleep(1.0)

    logger.info("Connected to CARLA at %s:%d", host, port)
    return client, world


def apply_weather(world, weather_cfg: dict, logger) -> None:
    preset = weather_cfg.get("preset")
    if preset:
        weather = getattr(carla.WeatherParameters, preset, None)
        if weather is not None:
            world.set_weather(weather)
            logger.info("Applied weather preset: %s", preset)
            return

    current = world.get_weather()
    current.cloudiness = float(weather_cfg.get("cloudiness", current.cloudiness))
    current.precipitation = float(weather_cfg.get("precipitation", current.precipitation))
    current.fog_density = float(weather_cfg.get("fog_density", current.fog_density))
    current.sun_altitude_angle = float(
        weather_cfg.get("sun_altitude_angle", current.sun_altitude_angle)
    )
    world.set_weather(current)
    logger.info("Applied custom weather parameters")


def spawn_camera_on_vehicle(world, camera_cfg: dict, logger) -> Tuple[carla.Actor, carla.Actor]:
    blueprints = world.get_blueprint_library()
    vehicle_bps = blueprints.filter("vehicle.*")
    if not vehicle_bps:
        raise RuntimeError("No vehicle blueprints found in CARLA world.")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in current CARLA map.")

    vehicle_bp = random.choice(vehicle_bps)
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(camera_cfg.get("image_width", 1280)))
    camera_bp.set_attribute("image_size_y", str(camera_cfg.get("image_height", 720)))
    camera_bp.set_attribute("fov", str(camera_cfg.get("fov", 90)))

    transform_cfg = camera_cfg.get("transform", {})
    camera_transform = carla.Transform(
        carla.Location(
            x=float(transform_cfg.get("x", 1.5)),
            y=float(transform_cfg.get("y", 0.0)),
            z=float(transform_cfg.get("z", 2.4)),
        ),
        carla.Rotation(
            pitch=float(transform_cfg.get("pitch", -10.0)),
            yaw=float(transform_cfg.get("yaw", 0.0)),
            roll=float(transform_cfg.get("roll", 0.0)),
        ),
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    logger.info("Spawned ego vehicle id=%s and camera id=%s", vehicle.id, camera.id)
    return vehicle, camera


def spawn_ego_vehicle(
    world,
    vehicle_filter: str = "vehicle.tesla.model3",
    spawn_point_index: int = 0,
    logger=None,
) -> carla.Actor:
    blueprints = world.get_blueprint_library().filter(vehicle_filter)
    if not blueprints:
        blueprints = world.get_blueprint_library().filter("vehicle.*")
    if not blueprints:
        raise RuntimeError("No vehicle blueprints available for ego spawn.")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in current CARLA map.")

    index = int(spawn_point_index) % len(spawn_points)
    vehicle_bp = random.choice(blueprints)
    ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[index])
    if ego_vehicle is None:
        raise RuntimeError(f"Failed to spawn ego vehicle at spawn index {index}.")

    if logger is not None:
        logger.info(
            "Spawned ego vehicle id=%s blueprint=%s at spawn_index=%s",
            ego_vehicle.id,
            vehicle_bp.id,
            index,
        )
    return ego_vehicle


def attach_rgb_camera(world, vehicle: carla.Actor, camera_cfg: dict, logger=None) -> carla.Actor:
    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(camera_cfg.get("image_width", 1280)))
    camera_bp.set_attribute("image_size_y", str(camera_cfg.get("image_height", 720)))
    camera_bp.set_attribute("fov", str(camera_cfg.get("fov", 90)))
    camera_bp.set_attribute("sensor_tick", str(camera_cfg.get("sensor_tick", 0.0)))

    transform_cfg = camera_cfg.get("transform", {})
    camera_transform = carla.Transform(
        carla.Location(
            x=float(transform_cfg.get("x", 1.5)),
            y=float(transform_cfg.get("y", 0.0)),
            z=float(transform_cfg.get("z", 2.4)),
        ),
        carla.Rotation(
            pitch=float(transform_cfg.get("pitch", -10.0)),
            yaw=float(transform_cfg.get("yaw", 0.0)),
            roll=float(transform_cfg.get("roll", 0.0)),
        ),
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    if logger is not None:
        logger.info("Attached RGB camera id=%s to ego id=%s", camera.id, vehicle.id)
    return camera


def attach_lidar(world, vehicle: carla.Actor, lidar_cfg: dict, logger=None) -> carla.Actor:
    lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", str(lidar_cfg.get("channels", 32)))
    lidar_bp.set_attribute("range", str(lidar_cfg.get("range", 50.0)))
    lidar_bp.set_attribute("points_per_second", str(lidar_cfg.get("points_per_second", 56000)))
    lidar_bp.set_attribute("rotation_frequency", str(lidar_cfg.get("rotation_frequency", 20.0)))
    lidar_bp.set_attribute("upper_fov", str(lidar_cfg.get("upper_fov", 10.0)))
    lidar_bp.set_attribute("lower_fov", str(lidar_cfg.get("lower_fov", -30.0)))
    lidar_bp.set_attribute("sensor_tick", str(lidar_cfg.get("sensor_tick", 0.0)))

    transform_cfg = lidar_cfg.get("transform", {})
    lidar_transform = carla.Transform(
        carla.Location(
            x=float(transform_cfg.get("x", 1.5)),
            y=float(transform_cfg.get("y", 0.0)),
            z=float(transform_cfg.get("z", 2.2)),
        ),
        carla.Rotation(
            pitch=float(transform_cfg.get("pitch", 0.0)),
            yaw=float(transform_cfg.get("yaw", 0.0)),
            roll=float(transform_cfg.get("roll", 0.0)),
        ),
    )
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    if logger is not None:
        logger.info("Attached LiDAR id=%s to ego id=%s", lidar.id, vehicle.id)
    return lidar


def lidar_measurement_to_array(lidar_measurement) -> np.ndarray:
    points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)
    points = points.reshape((-1, 4))
    return points.copy()


def image_to_bgr_array(image) -> np.ndarray:
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = bgra.reshape((image.height, image.width, 4))
    bgr = bgra[:, :, :3]
    return bgr.copy()


def get_synced_camera_image(world, image_queue: queue.Queue, timeout_seconds: float = 2.0):
    snapshot = world.tick()
    while True:
        image = image_queue.get(timeout=timeout_seconds)
        if image.frame >= snapshot:
            return image


def set_world_synchronous(
    world,
    enabled: bool,
    fixed_delta_seconds: Optional[float] = None,
):
    settings = world.get_settings()
    settings.synchronous_mode = enabled
    if fixed_delta_seconds is not None:
        settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)


def spawn_npc_vehicles(
    client,
    world,
    vehicle_count: int,
    traffic_manager_port: int,
    seed: int,
    logger=None,
):
    if vehicle_count <= 0:
        return []

    traffic_manager = client.get_trafficmanager(traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(seed)

    spawn_points = world.get_map().get_spawn_points()
    random.Random(seed).shuffle(spawn_points)

    blueprints = world.get_blueprint_library().filter("vehicle.*")
    commands = []
    for transform in spawn_points[:vehicle_count]:
        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        if bp.has_attribute("driver_id"):
            driver_id = random.choice(bp.get_attribute("driver_id").recommended_values)
            bp.set_attribute("driver_id", driver_id)

        command = carla.command.SpawnActor(bp, transform).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager_port)
        )
        commands.append(command)

    responses = client.apply_batch_sync(commands, True)
    actor_ids = [r.actor_id for r in responses if not r.error]

    if logger is not None:
        logger.info("Spawned %d/%d NPC vehicles.", len(actor_ids), vehicle_count)
        for response in responses:
            if response.error:
                logger.warning("NPC vehicle spawn error: %s", response.error)
    return actor_ids


def capture_single_image(world, camera, timeout_seconds: float = 5.0):
    image_queue = queue.Queue(maxsize=1)
    camera.listen(image_queue.put)

    image = None
    try:
        for _ in range(20):
            world.wait_for_tick(timeout_seconds)
            if not image_queue.empty():
                break
        image = image_queue.get(timeout=timeout_seconds)
    finally:
        camera.stop()
    return image


def cleanup_actors(actors, logger) -> None:
    for actor in reversed(actors):
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to destroy actor: %s", exc)


def cleanup_actor_ids(client, actor_ids, logger) -> None:
    if not actor_ids:
        return
    try:
        client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in actor_ids])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to cleanup actor ids: %s", exc)

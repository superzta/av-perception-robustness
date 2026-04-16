import queue
import random
import sys
import time
from pathlib import Path
from typing import Tuple

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
    preset = weather_cfg.get("preset", "ClearNoon")
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

"""Unit tests for BaZhai eight-star placement based on house orientation."""

# Import the target module without importing the package (which requires heavy
# dependencies like OpenCV). This keeps the test lightweight while still
# exercising the public API.
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

module_path = Path(__file__).parent / "fengshui" / "bazhai_eightstars.py"
spec = spec_from_file_location("bazhai_eightstars", module_path)
bazhai = module_from_spec(spec)
spec.loader.exec_module(bazhai)  # type: ignore

analyze_eightstars = bazhai.analyze_eightstars
HOUSE_DIRECTION_STARS = bazhai.HOUSE_DIRECTION_STARS


def _sample_layout():
    """Return a simple square layout with one room in each direction."""
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    centers = {
        "东": (80, 50),
        "东北": (80, 80),
        "北": (50, 80),
        "西北": (20, 80),
        "西": (20, 50),
        "西南": (20, 20),
        "南": (50, 20),
        "东南": (80, 20),
    }
    rooms = [{"name": name, "center": c} for name, c in centers.items()]
    return polygon, rooms


def test_orientation_east_house():
    polygon, rooms = _sample_layout()
    orientation = {"north_angle": 90, "house_orientation": "坐北朝南"}
    result = analyze_eightstars(polygon, rooms, orientation)
    mapping = {item["direction"]: item["star"] for item in result}
    assert mapping == HOUSE_DIRECTION_STARS["坐北朝南"]


def test_orientation_west_house():
    polygon, rooms = _sample_layout()
    orientation = {"north_angle": 90, "house_orientation": "坐西北朝东南"}
    result = analyze_eightstars(polygon, rooms, orientation)
    mapping = {item["direction"]: item["star"] for item in result}
    assert mapping == HOUSE_DIRECTION_STARS["坐西北朝东南"]


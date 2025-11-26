import importlib.util
from pathlib import Path


# Skip legacy cosmos test suites when the upstream package is not available.
COSMOS_AVAILABLE = importlib.util.find_spec("cosmos") is not None
_COSMOS_DIR_MARKERS = ("tests/fits", "tests/models", "tests/science")
_COSMOS_FILES = {"test_phase6a.py", "test_phase7a.py"}


def pytest_ignore_collect(collection_path, config=None) -> bool:
    if COSMOS_AVAILABLE:
        return False
    path = Path(str(collection_path))
    path_str = str(path)
    if any(marker in path_str for marker in _COSMOS_DIR_MARKERS):
        return True
    if path.name in _COSMOS_FILES:
        return True
    return False

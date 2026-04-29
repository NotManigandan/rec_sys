import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.common import run_synthetic_integration


if __name__ == "__main__":
    run_synthetic_integration("two_tower", port=50113)

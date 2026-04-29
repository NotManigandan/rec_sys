import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.common import run_movielens_integration


if __name__ == "__main__":
    run_movielens_integration("two_tower", port=50123)

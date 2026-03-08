"""conftest.py — Add src/ to sys.path so tests can import src modules."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

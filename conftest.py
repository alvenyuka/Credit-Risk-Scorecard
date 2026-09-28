"""Puts src/ on sys.path for the tests.

Every module in src/ imports its siblings flat (`import week8_run`), and the
notebook does `sys.path.insert(0, "src")` before importing them the same way,
because each module is also runnable as a script. The tests follow that instead
of importing `src.metrics_scratch` as a package.

The difference is not cosmetic. Importing the same file under two names produces
two separate module objects with two separate copies of everything in them, so a
test can pass against one copy while the pipeline runs against the other.
"""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

"""Run model-free CPU fixtures against an explicitly supplied patched checkout."""
import argparse
import os
from pathlib import Path
import sys
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runtime", type=Path, required=True)
args = parser.parse_args()
runtime = args.runtime.resolve()
if not (runtime / "models/minimax_h3/vsa/graphs.py").is_file():
    parser.error("--runtime must point to the patched backend checkout")
os.environ["WANGP_TEST_RUNTIME"] = str(runtime)
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
sys.dont_write_bytecode = True
tests = Path(__file__).resolve().parent
suite = unittest.TestLoader().discover(str(tests), pattern="test_*.py")
raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())

"""
Launcher for vis_bad_cases.py that patches torch.__version__ before importing transformers.
Usage:  python visaug/analysis/run_bad_case.py [--case-idx 0] [--vis-layer 15]
"""
import builtins
import sys

# Patch torch.__version__ BEFORE any transformers import
_original_import = builtins.__import__
def _patched_import(name, *args, **kwargs):
    mod = _original_import(name, *args, **kwargs)
    if name == 'torch':
        mod.__version__ = '2.1.0'
    return mod
builtins.__import__ = _patched_import

# Now safe to import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch  # this triggers the patch
print(f"Torch version: {torch.__version__}")  # should say 2.1.0

# Verify transformers loads
from transformers import AutoProcessor, AutoModelForCausalLM
print("Transformers loaded OK")

# Parse args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--case-idx", type=int, default=0)
parser.add_argument("--vis-layer", type=int, default=15)
args = parser.parse_args()

# Now import and run vis_bad_cases main logic
# We need to exec the file in a way that our patched __import__ propagates
# Use the current module's namespace
import runpy
# Set sys.argv for the target script
sys.argv = ['vis_bad_cases.py',
    '--index', '/root/code/ClearSight/outputs/analysis/bad_cases/bad_cases_index.json',
    '--case-idx', str(args.case_idx),
    '--vis-layer', str(args.vis_layer)]

# Run the script in the current namespace so __import__ patch persists
script_path = '/root/code/ClearSight/visaug/analysis/vis_bad_cases.py'
exec(compile(open(script_path).read(), script_path, 'exec'), {'__builtins__': builtins, '__name__': '__main__', '__file__': script_path})

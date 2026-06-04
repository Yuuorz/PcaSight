#!/bin/bash
# Launcher for vis_bad_cases.py that patches torch.__version__ before import
# Usage: bash visaug/analysis/run_bad_case.sh [case_idx] [vis_layer]

CASE_IDX=${1:-0}
VIS_LAYER=${2:-15}

cd /root/code/ClearSight

python -c "
import builtins, types, sys

# Patch torch.__version__ before any imports
original_import = builtins.__import__
def _patched_import(name, *args, **kwargs):
    mod = original_import(name, *args, **kwargs)
    if name == 'torch':
        mod.__version__ = '2.1.0'
    return mod

builtins.__import__ = _patched_import

# Now run vis_bad_cases
sys.argv = ['vis_bad_cases.py',
    '--index', 'outputs/analysis/bad_cases/bad_cases_index.json',
    '--case-idx', '$CASE_IDX',
    '--vis-layer', '$VIS_LAYER']

exec(open('visaug/analysis/vis_bad_cases.py').read())
" 2>&1

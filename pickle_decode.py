import pickle
import numpy as np
import sys
import types
 
# ------------------------------------------------------------
# Your NumPy compatibility fix (keep as is)
# ------------------------------------------------------------
core_module = types.ModuleType("numpy._core")
multiarray_module = types.ModuleType("numpy._core.multiarray")
core_module.multiarray = multiarray_module
 
def dummy_scalar(*args, **kwargs):
    return np.array(args[0] if args else 0)
 
multiarray_module.scalar = dummy_scalar
 
sys.modules["numpy._core"] = core_module
sys.modules["numpy._core.multiarray"] = multiarray_module
 
# ------------------------------------------------------------
# Simple, robust recursive printer
# ------------------------------------------------------------
def print_all(obj, name="root", indent=0):
    """Recursively print any object, showing keys for dicts and indices for sequences."""
    prefix = "  " * indent
    if isinstance(obj, dict):
        print(f"{prefix}{name} (dict, {len(obj)} keys)")
        for k, v in obj.items():
            print_all(v, f"{name}['{k}']", indent + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name} ({type(obj).__name__}, length {len(obj)})")
        for i, item in enumerate(obj):
            print_all(item, f"{name}[{i}]", indent + 1)
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}{name} = numpy array shape {obj.shape} dtype {obj.dtype}")
        # Show a few values if not too large
        if obj.size <= 20:
            print(f"{prefix}  values: {obj}")
        else:
            print(f"{prefix}  first 5 values: {obj.flat[:5]} ...")
    else:
        # Show primitive or other objects
        print(f"{prefix}{name} = {repr(obj)[:200]}")
 
# ------------------------------------------------------------
# Load and print
# ------------------------------------------------------------
file_path = "a2c_tables.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)
 
print("\n=== FULL CONTENTS OF PICKLE FILE ===\n")
print_all(data, "data")
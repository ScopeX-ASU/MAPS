"""
Date: 2024-11-24 15:28:23
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-24 15:38:46
FilePath: /MAPS/core/ai4pde/models/layers/__init__.py
"""

import importlib
import os

# # automatically import any Python files in this directory
# for file in sorted(os.listdir(os.path.dirname(__file__))):
#     if file.endswith(".py") and not file.startswith("_"):
#         source = file[: file.find(".py")]
#         module = importlib.import_module("./" + source)
#         if "__all__" in module.__dict__:
#             names = module.__dict__["__all__"]
#         else:
#             # import all names that do not begin with _
#             names = [x for x in module.__dict__ if not x.startswith("_")]
#         globals().update({k: getattr(module, k) for k in names})

# Automatically import all Python files in this directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        source = file[: file.find(".py")]  # Extract the module name
        # Use relative import to this package
        module = importlib.import_module(f".{source}", package=__name__)
        if "__all__" in module.__dict__:
            # Explicitly defined __all__
            names = module.__dict__["__all__"]
        else:
            # Import all names that do not begin with _
            names = [name for name in module.__dict__ if not name.startswith("_")]
        # Add the imported names to the current namespace
        globals().update({name: getattr(module, name) for name in names})

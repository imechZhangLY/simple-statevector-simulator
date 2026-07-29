"""Exit 0 if this interpreter can reach a SUPA device.

Use `brsmi` to ask whether the machine has SUPA hardware at all. This script
answers the narrower question of whether the running interpreter can use it,
which is what breaks when envs/supa loses --system-site-packages and the venv
stops seeing the image's torch and torch_br.
"""

import sys

try:
    import torch
    import torch_br  # noqa: F401  importing it registers torch.supa

    # torch.supa is absent rather than falsy when registration did not happen.
    available = bool(torch.supa.is_available())
except (ImportError, AttributeError):
    available = False

print(f"interpreter: {sys.executable}")
print("SUPA reachable." if available else "SUPA not reachable from this interpreter.")
sys.exit(0 if available else 1)
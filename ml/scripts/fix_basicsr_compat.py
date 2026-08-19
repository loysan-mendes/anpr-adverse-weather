"""
Phase 6 - Step 0: One-time compatibility patch for basicsr.

basicsr (a dependency of the `realesrgan` pip package) imports
`torchvision.transforms.functional_tensor`, a module that was removed in
torchvision >= 0.17. basicsr hasn't been updated for this -- it's a
well-known, widely-reported break, not something specific to your setup.
The fix is a one-line import swap: `functional_tensor` -> `functional`
(the function we need, rgb_to_grayscale, still lives there).

Run this ONCE, right after `pip install -r requirements.txt`, before
running any other Phase 6 script.

Usage:
    python ml/scripts/fix_basicsr_compat.py
"""

import sysconfig
from pathlib import Path

# IMPORTANT: we do NOT `import basicsr` here. basicsr's own __init__.py
# eagerly imports every submodule, including the one with the broken
# `functional_tensor` import -- so importing the package is exactly what
# crashes before we'd get a chance to patch it. Instead, find the file
# directly via the venv's site-packages path.
site_packages = Path(sysconfig.get_paths()["purelib"])
degradations_path = site_packages / "basicsr" / "data" / "degradations.py"

if not degradations_path.exists():
    raise SystemExit(f"Could not find {degradations_path} -- is basicsr installed? "
                      f"Run: pip install -r requirements.txt")

text = degradations_path.read_text()
old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
new = "from torchvision.transforms.functional import rgb_to_grayscale"

if old in text:
    degradations_path.write_text(text.replace(old, new))
    print(f"Patched: {degradations_path}")
elif new in text:
    print("Already patched -- nothing to do.")
else:
    print("WARNING: expected import line not found -- basicsr may have changed "
          "its internals. Open this file manually and look for any import of "
          "`functional_tensor`:")
    print(degradations_path)
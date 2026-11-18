"""Source package for the housing decision support system."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PACKAGES = _PROJECT_ROOT / ".python_packages"

if _LOCAL_PACKAGES.exists():
    local_packages_str = str(_LOCAL_PACKAGES)
    if local_packages_str not in sys.path:
        sys.path.insert(0, local_packages_str)

"""
Deployment entry-point shim.

The actual app now lives in src/acb/api/main.py (see IMPROVE_AGENTS_PROMPT.md
for why). This file exists only so external start commands that still
invoke `uvicorn app:app` (e.g. Render's configured Start Command) keep
working without needing a dashboard change. Point new deployments at
`acb.api.main:app` with PYTHONPATH=src instead of adding to this file.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from acb.api.main import app  # noqa: E402,F401

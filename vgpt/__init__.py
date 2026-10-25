"""VehicleGPT / Vantage: an offline vehicle assistant.  Importing this package puts the offline guard in place (vgpt/offline.py): from then on this process cannot open any network socket."""
from . import offline as _offline

_offline.install()

import re as _re
try: _re._MAXCACHE = max(int(getattr(_re, "_MAXCACHE", 512)), 8192)                      # the answer routes use several thousand distinct patterns: with the default 512-entry cache they were recompiled on every question (0.26 s on a laptop, ~1.5 s on the Pi 4)
except Exception: pass

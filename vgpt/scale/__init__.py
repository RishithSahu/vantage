"""Automatic scaling: probe the hardware, plan the deployment, and keep the latency target while the load changes (docs/SCALING.md)."""
from .probe import Hardware, probe          # noqa: F401
from .plan import Plan, make_plan           # noqa: F401

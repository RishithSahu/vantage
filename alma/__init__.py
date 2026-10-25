"""VOLT-Q/ALMA feasibility prototype.

This package is deliberately separate from the product route.  It tests whether a
small encoder can learn reusable semantic cards from algebra-generated declarative
text without seeing complete vehicle question-to-program training pairs.
"""

from .cards import Card, task_cards, scope_cards

__all__ = ["Card", "task_cards", "scope_cards"]

"""Build the product's int8 frame model for a test without changing the environment of the tests that run after it (vehiclegpt._adopt_product_model() sets VGPT_FRAME=2 and friends for the rest of the process)."""
import os


def adopt_product_model():
    """-> a function that puts the environment back as it was."""
    before = dict(os.environ)
    import vehiclegpt
    vehiclegpt._adopt_product_model()
    changed = {k: before.get(k) for k in set(os.environ) | set(before) if os.environ.get(k) != before.get(k)}

    def restore():
        for k, v in changed.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return restore

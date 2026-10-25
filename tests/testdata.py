"""Test data that is not source code, asked for by name.

    from tests import testdata                      (or: import testdata, from a test that runs with tests/ on the path)
    @testdata.needs("legacy_worlds")                a class or a test that is an INTEGRATION test: it needs that group of data (tools_testdata.py lists the groups)
    testdata.require("corpus")                      the same, from inside a setUpClass / a test

Absent data skips the test and says what is missing and how to get it.  With VGPT_REQUIRE_TESTDATA=1 (every group) or a comma list of groups (the release gate, `tools_release_gate.py`) an absent group is a FAILURE, never a skip: a release is not built on tests that did not run.
`files_exist(*paths)` / `need_files(*paths)` do the same for single files (relative to the project)."""
import functools
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tests", "testdata_manifest.json")


def strict(group=None):
    """Is an absent group a failure?  VGPT_REQUIRE_TESTDATA=1 requires every group; a comma list ("corpus,sample_cars") requires those."""
    v = os.environ.get("VGPT_REQUIRE_TESTDATA", "").strip()
    return v == "1" or (group is not None and group in [x.strip() for x in v.split(",") if x.strip()])


def _groups():
    try: return json.load(open(MANIFEST, encoding="utf-8")).get("groups", {})
    except (OSError, ValueError): return {}


def missing(group):
    """-> the files of a group that are not on disk, or ['(unknown group)']."""
    g = _groups().get(group)
    if g is None: return ["(unknown group " + group + ")"]
    return [f for f in g.get("files", {}) if not os.path.exists(os.path.join(ROOT, f))]


def _refuse(why, group=None):
    if strict(group): raise AssertionError("test data required for this release is missing: " + why)
    raise unittest.SkipTest(why)


def require(group):
    m = missing(group)
    if m:
        make = (_groups().get(group) or {}).get("make", "python tools_testdata.py check")
        _refuse(f"integration test data '{group}' is absent ({', '.join(m[:3])}{' ...' if len(m) > 3 else ''}); {make}", group)


def gate(group):
    """For a module that needs the data at import time: -> (present, reason).  An absent group in a strict run raises at once; otherwise the module builds nothing and skips its classes with `unittest.skipUnless(*testdata.gate(group))`."""
    m = missing(group)
    if not m: return True, ""
    why = f"integration test data '{group}' is absent ({', '.join(m[:3])}{' ...' if len(m) > 3 else ''}); {(_groups().get(group) or {}).get('make', 'python tools_testdata.py check')}"
    if strict(group): raise AssertionError("test data required for this release is missing: " + why)
    return False, why


def needs(group):
    """A decorator for a TestCase class or a test method."""
    def deco(obj):
        if isinstance(obj, type):
            orig = obj.setUpClass.__func__ if "setUpClass" in obj.__dict__ else None

            @classmethod
            def setUpClass(cls):
                require(group)
                if orig is not None: orig(cls)
                else: super(obj, cls).setUpClass()
            obj.setUpClass = setUpClass; return obj

        @functools.wraps(obj)
        def wrapper(*a, **k):
            require(group); return obj(*a, **k)
        return wrapper
    return deco


def files_exist(*paths): return all(os.path.exists(os.path.join(ROOT, p)) for p in paths)


def need_files(*paths, group=None):
    gone = [p for p in paths if not os.path.exists(os.path.join(ROOT, p))]
    if gone: _refuse("integration test data is absent: " + ", ".join(gone), group)

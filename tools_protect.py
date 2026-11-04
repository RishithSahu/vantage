"""Keep the files the assistant cannot run without: list them, copy them to another drive, check them before a long job, restore what is missing, and lock them against cleaning software.

    python tools_protect.py manifest         write protected_manifest.json (path, size, sha256) of every protected file
    python tools_protect.py backup           copy what is missing or different to BACKUP (never deletes anything there); also a dated copy of the code
    python tools_protect.py check [--full]   every protected file is there with the right size (--full: and the right hash); exit 1 when not, listing what is wrong
    python tools_protect.py restore          copy back from BACKUP what is missing here (never overwrites a file that is there)
    python tools_protect.py lock | unlock    read-only attribute + a Windows ACL that denies deleting the protected files and the model folders (unlock before changing a protected file)

Why: on 2026-10-28 at 11:14 every large file in models/ and data/ vanished at once and nothing in this project deleted them.  The check takes seconds; run it before any long job.
What this cannot do: stop an administrator or the system from removing a file, or recover a file that was never backed up (bge-small-en-v1.5 and semantic-lora-qwen25-05b-v2.gguf are not protected because no copy exists)."""
import fnmatch, hashlib, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKUP = os.environ.get("VGPT_BACKUP", r"D:\TATA_Elexi_Teliport_backup")
MANIFEST = os.path.join(ROOT, "protected_manifest.json")
PROTECT = ["models/meaning-lora-frame2r12.gguf", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "models/vg_frame2r12.vgw", "models/product.json", "models/ort_frame2r12/*", "models/asr/*",
           "data/world4/aster_ev/aster_ev_4yr.db", "data/world4/kestrel_petrol/kestrel_petrol_4yr.db", "data/world4/rhino_diesel/rhino_diesel_4yr.db",
           "data/world_*.db", "data/gen_courier_ext.db", "data/ref/*"]
LOCK_DIRS = ["models", "data/world4/aster_ev", "data/world4/kestrel_petrol", "data/world4/rhino_diesel"]                                                       # folders in which deleting a protected file is also denied (a file's own deny is overridden by delete permission on its folder). NOT locked: data/ itself (SQLite deletes its own journal files there), so data/world_*.db, gen_courier_ext.db and data/ref are protected only by backup + check + restore
CODE_DIRS = ["vgpt", "ui", "tests", "bench", "tools", "engine", "documentation", "docs", "deploy/linux"]
CODE_GLOBS = ["*.py", "*.md", "*.json", "*.ini", "*.txt"]
SKIP = ("__pycache__", ".cols", "-shm", "-wal", "-journal", ".tmp")


def files():
    out = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in ("results", "__pycache__", ".git", "deploy")]
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), ROOT).replace("\\", "/")
            if any(s in rel for s in SKIP): continue
            if any(fnmatch.fnmatch(rel, p) for p in PROTECT): out.append(rel)
    return sorted(out)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def cmd_manifest():
    m = {r: {"size": os.path.getsize(os.path.join(ROOT, r)), "sha256": sha(os.path.join(ROOT, r))} for r in files()}
    json.dump({"made": time.strftime("%Y-%m-%d %H:%M"), "files": m}, open(MANIFEST, "w"), indent=0); print(len(m), "files,", sum(x["size"] for x in m.values()) // 1_000_000, "MB ->", MANIFEST)


def cmd_check(full=False):
    if not os.path.exists(MANIFEST): print("no manifest: run `manifest` first"); return 2
    bad = []
    for r, x in json.load(open(MANIFEST))["files"].items():
        p = os.path.join(ROOT, r)
        if not os.path.exists(p): bad.append(("MISSING", r))
        elif os.path.getsize(p) != x["size"]: bad.append(("SIZE", r))
        elif full and sha(p) != x["sha256"]: bad.append(("HASH", r))
    for k, r in bad: print(k, r)
    print("protected files OK" if not bad else f"{len(bad)} problem(s): run `restore`"); return 1 if bad else 0


def copy_missing(src_root, dst_root, rels, overwrite=False):
    n = 0
    for r in rels:
        s, d = os.path.join(src_root, r), os.path.join(dst_root, r)
        if not os.path.exists(s): continue
        if os.path.exists(d) and (not overwrite or os.path.getsize(d) == os.path.getsize(s)): continue
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d); n += 1; print("copied", r)
    return n


def cmd_backup():
    if not os.path.isdir(os.path.dirname(BACKUP) or BACKUP): print("backup drive not found:", BACKUP); return 2
    n = copy_missing(ROOT, os.path.join(BACKUP, "files"), files(), overwrite=True)
    snap = os.path.join(BACKUP, "code", time.strftime("%Y-%m-%d")); k = 0
    for d in CODE_DIRS:
        for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
            dn[:] = [x for x in dn if x != "__pycache__"]
            for f in fn:
                if f.endswith((".pyc", ".db", ".gguf", ".onnx", ".vgw", ".hevc")): continue
                s = os.path.join(dp, f); t = os.path.join(snap, os.path.relpath(s, ROOT)); os.makedirs(os.path.dirname(t), exist_ok=True); shutil.copy2(s, t); k += 1
    for f in os.listdir(ROOT):
        if os.path.isfile(os.path.join(ROOT, f)) and any(fnmatch.fnmatch(f, g) for g in CODE_GLOBS): os.makedirs(snap, exist_ok=True); shutil.copy2(os.path.join(ROOT, f), os.path.join(snap, f)); k += 1
    shutil.copy2(MANIFEST, os.path.join(BACKUP, "protected_manifest.json")) if os.path.exists(MANIFEST) else None
    print(f"backup: {n} protected files copied, {k} code files in {snap}"); return 0


def cmd_restore():
    rels = list(json.load(open(MANIFEST))["files"]) if os.path.exists(MANIFEST) else []
    n = copy_missing(os.path.join(BACKUP, "files"), ROOT, rels); print(n, "files restored"); return cmd_check()


def icacls(args): return subprocess.run(["icacls", *args], capture_output=True, text=True)


def cmd_lock(lock=True):
    user = os.environ.get("USERNAME", "")
    for r in files():
        p = os.path.join(ROOT, r)
        if os.name == "nt":
            if not r.startswith("data/") or not lock: subprocess.run(["attrib", "+R" if lock else "-R", p], capture_output=True)      # a database copied with the read-only flag stays read-only, which breaks every test that copies and writes: data files get the delete-denying ACL only
            icacls([p, "/deny", f"{user}:(DE)"] if lock else [p, "/remove:d", user])
    for d in LOCK_DIRS:
        p = os.path.join(ROOT, d)
        if os.name == "nt": icacls([p, "/deny", f"{user}:(DC)"] if lock else [p, "/remove:d", user])
    print("locked" if lock else "unlocked", len(files()), "files and", ", ".join(LOCK_DIRS))


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"manifest": lambda: cmd_manifest() or 0, "backup": cmd_backup, "check": lambda: cmd_check("--full" in sys.argv), "restore": cmd_restore, "lock": lambda: cmd_lock(True) or 0, "unlock": lambda: cmd_lock(False) or 0}.get(c, lambda: print(__doc__) or 2)())

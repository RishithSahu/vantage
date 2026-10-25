"""From hardware to configuration: one rule set that gives a sensible deployment on every machine from a 2-core board to a many-core vehicle computer, with the reason for every number.

    hw = probe()                    # what the machine is
    plan = make_plan(hw)            # what to run on it
    plan.tier                       # 'edge', 'edge+', 'hpc', 'hpc+gpu' ... the nearest named profile (a label, not a switch)
    plan.explain()                  # every decision and why

What scales and what does not.  The model, the interfaces, the database schema and the answers are the same at every size; the plan changes only numbers and which kernel runs the model:

  backend      the fastest model runtime the hardware has (int8 kernels on the CPU; a float graph on a GPU when one is present); the choice can be confirmed by a short measurement (`calibrate`)
  contexts     how many questions the model answers at once (one on a 2-core board; one per 4 spare cores on a larger one, bounded by memory)
  threads      cores given to each context
  reserve      cores and memory never given to the assistant, so the vehicle's own functions keep theirs (none on the 2-core reference budget, which already IS the assistant's share)
  sessions     how many people can be asking at once
  history      how many days are kept at one second, per minute and per record, from memory and disk
  ingest       how many vehicle streams are decoded and stored in parallel
  shed ladder  what is given up first, in order, when the load would break the latency target (see autoscale.py)

The sizes of one model context come from measurement (docs/SCALING.md): the int8 build is about 0.65 GB resident for the first context and the weights are shared by the others when the platform maps files."""
import dataclasses

from .probe import Hardware

# measured footprints (docs/SCALING.md); extra contexts of the int8 build share the mapped weights, so they cost far less than the first
SMALL_BOARD_CORES = 4
CTX_FIRST_MB = 700
CTX_EXTRA_MB = 180
CTX_FIRST_MB_LLAMA = 650
CTX_EXTRA_MB_LLAMA = 1100
APP_BASE_MB = 350                       # interpreter, application, runtime, one database handle
SESSION_MB = 25
GPU_MODEL_MB = 2600                     # the float graph held on the accelerator

SHED_LADDER = ("pause_background_analytics", "lower_camera_analysis_rate", "queue_service_and_fleet_questions", "reject_non_driver_questions", "shorten_answers")


@dataclasses.dataclass
class Plan:
    tier: str
    backend: str                          # ort-int8 | ort-gpu | llama
    contexts: int                         # model contexts at start (the best latency for one person)
    threads: int
    reserve_cores: int
    reserve_mb: int
    sessions: int
    sqlite_cache_kb: int
    sqlite_mmap_mb: int
    fine_days: int
    coarse_days: int
    ring_s: int
    ingest_workers: int
    camera_fps: float
    max_contexts: int = 1                 # how far the autoscaler may grow when several people ask at once (contexts of two cores each)
    latency_target_s: float = 2.0
    shed_ladder: tuple = SHED_LADDER
    notes: list = dataclasses.field(default_factory=list)
    hardware: str = ""

    def explain(self): return "\n".join(f"- {n}" for n in self.notes)

    def as_dict(self): return dataclasses.asdict(self)


def _label(hw, backend, cores_for_ai):
    if backend == "ort-gpu": return "hpc+gpu"
    if hw.cores <= 2: return "edge"
    if hw.cores <= 4: return "edge+"
    if hw.cores < 8: return "gateway"
    return "hpc"


def make_plan(hw: Hardware, have_onnx=True, have_llama=True, have_float_graph=False, latency_target_s=2.0, vehicles=1, knee=None):
    """The deployment for `hw`.  `have_*` say which model builds are installed (the plan never names a backend that is not there)."""
    n = []; cores = max(1, hw.cores); mem = max(512, hw.mem_mb)
    # 1. what the vehicle's own functions keep
    reserve_cores = 0 if cores <= 2 else 1 if cores <= 4 else 2 if cores <= 8 else 3
    reserve_mb = 0 if mem <= 2560 else int(mem * 0.12)
    ai_cores = max(1, cores - reserve_cores); ai_mem = mem - reserve_mb
    n.append(f"{cores} cores and {mem} MB are available: {reserve_cores} core(s) and {reserve_mb} MB stay reserved for the vehicle's own functions, the assistant plans with {ai_cores} cores and {ai_mem} MB"
             + (" (the 2-core budget of the case study already is the assistant's share)" if reserve_cores == 0 else ""))
    # 2. backend
    gpu_ok = have_float_graph and any(a in hw.accel for a in ("cuda", "dml", "tensorrt")) and hw.gpu_mb >= GPU_MODEL_MB + 500
    if gpu_ok: backend = "ort-gpu"; n.append(f"accelerator found ({'+'.join(hw.accel)}, {hw.gpu_mb} MB): the float graph runs on it, the CPU keeps the data work")
    elif have_onnx: backend = "ort-int8"; n.append("int8 build on ONNX Runtime: its blocked int8 matrix kernel is the fastest on CPUs both with and without dot-product instructions" + (f" (isa: {'+'.join(sorted(hw.isa))})" if hw.isa else ""))
    elif have_llama: backend = "llama"; n.append("ONNX Runtime is not installed: llama.cpp runs the same model (slower on CPUs without dot-product instructions)")
    else: backend = "none"; n.append("no model runtime is installed: the assistant answers from its typed engines only")
    # 3. contexts and threads
    first, extra = (CTX_FIRST_MB, CTX_EXTRA_MB) if backend in ("ort-int8", "ort-gpu") else (CTX_FIRST_MB_LLAMA, CTX_EXTRA_MB_LLAMA)
    if backend == "ort-gpu": per_thread_cores = 2
    elif knee: per_thread_cores = max(1, int(knee)); n.append(f"calibration on this hardware: a context stops gaining from more than {per_thread_cores} thread(s), so contexts get {per_thread_cores} core(s) each and the rest of the cores become more contexts")
    else: per_thread_cores = 2 if ai_cores < 8 else 4
    max_ctx_cpu = max(1, ai_cores // per_thread_cores)
    budget_mb = ai_mem - APP_BASE_MB
    if backend != "none" and budget_mb < first:                                                          # below about 1.1 GB the model does not fit next to the application: scale DOWN, do not overcommit
        n.append(f"only {ai_mem} MB for the assistant: a model context needs {first} MB next to the {APP_BASE_MB} MB application, so the model is not loaded and questions are answered by the typed engines (the same answers for the questions they cover, no free wording)")
        backend = "none"
    max_ctx_mem = 1 + max(0, (budget_mb - first) // extra) if budget_mb >= first else 0
    if backend == "none": contexts = 0; threads = 0
    else:
        contexts = max(1, min(max_ctx_cpu, max_ctx_mem, 16)); threads = max(1, min(per_thread_cores if knee else 4, ai_cores // contexts)) if backend != "ort-gpu" else 2
    if backend != "none": n.append(f"{contexts} model context(s) x {threads} thread(s): limited by cores ({max_ctx_cpu} fit) and memory ({max_ctx_mem} fit: {first} MB for the first, {extra} MB for each further one)")
    # 4. sessions
    max_ctx = 1 if backend == "none" else max(contexts, min(max_ctx_mem, max(1, (ai_cores + 1) // 2), 16))
    if backend in ("ort-int8", "llama") and ai_cores <= SMALL_BOARD_CORES and not knee:                                    # measured on the Raspberry Pi 4: every context streams the same weights through the same memory, so a second one lowered throughput (0.46 -> 0.34 answers/s) while more threads in the one lowered latency (meaning step 2.2 s on 1 core, 1.4 s on 2, 1.1 s on 4)
        contexts = max_ctx = 1; threads = min(ai_cores, 4); n.append(f"a board of {ai_cores} AI core(s) or fewer is memory-bandwidth bound: one context with all {threads} thread(s); the autoscaler would only add a context if it measured a real gain")
    if max_ctx > contexts and backend != "ort-gpu": threads = min(threads, 2)                                    # workers that may be added later all run on two cores, so the start context does too
    if backend != "none" and max_ctx > 1: n.append(f"the autoscaler may grow the service to {max_ctx} model contexts of {min(2, max(1, ai_cores // max_ctx))}+ core(s) when several people ask at once; it starts with {contexts}")
    sess = max(2, min(64, max(1, contexts) * 4 + max(0, ai_mem - APP_BASE_MB - first - extra * max(0, contexts - 1)) // (SESSION_MB * 40)))
    n.append(f"{sess} concurrent people (each costs about {SESSION_MB} MB and a read-only database handle)")
    # 5. data
    cache = int(min(262144, max(4096, mem * 1024 * 0.03))); mmap = 0 if mem < 3072 else int(min(2048, mem * 0.06))
    fine = int(min(60, max(7, 7 * mem / 2048))); coarse = int(min(1095, max(120, 120 * mem / 1024 / 1.0)))
    coarse = max(365, coarse) if hw.free_disk_mb == 0 or hw.free_disk_mb > 4000 else 120
    n.append(f"history: {fine} days at one second, {coarse} days per minute (memory {mem} MB, disk {hw.free_disk_mb or 'unknown'} MB free); page cache {cache // 1024} MB")
    # 6. ingest and camera
    ingest = max(1, min(max(1, vehicles), max(1, ai_cores // 2)))
    fps = 5.0 if cores > 2 else 2.0
    n.append(f"{ingest} ingest worker(s) for {vehicles} vehicle stream(s); camera analysis at {fps:g} frames per second")
    n.append(f"latency target {latency_target_s:g} s: when the measured 95th percentile threatens it the autoscaler gives up, in order: {', '.join(SHED_LADDER)}")
    return Plan(tier=_label(hw, backend, ai_cores), backend=backend, contexts=contexts, threads=threads, max_contexts=int(max_ctx), reserve_cores=reserve_cores, reserve_mb=reserve_mb, sessions=int(sess), sqlite_cache_kb=cache, sqlite_mmap_mb=mmap,
                fine_days=fine, coarse_days=int(coarse), ring_s=420 if cores <= 2 else 900 if cores < 8 else 1800, ingest_workers=ingest, camera_fps=fps, latency_target_s=latency_target_s, notes=n, hardware=hw.describe())

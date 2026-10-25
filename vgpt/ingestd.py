"""Ingest daemon (the ONLY writer of the vehicle DB). Consumes protocol-native messages, decodes via the adapters, stores tiered data.
In the demo the buses are a recorded replay (pickle stream); in a car the same Ingestor is fed by SocketCAN / vsomeip / VHAL / slog2 readers.
  python -m vgpt.ingestd --replay data/replay.pkl --db data/live.db --speed 1      real-time pacing, reports max lag (does it keep up?)
  python -m vgpt.ingestd --replay data/replay.pkl --db data/live.db --speed max    unpaced: pure throughput
Stats are written to --stats (JSON) on exit/SIGTERM and every 5 s so a supervisor (the benchmark) can read them."""
import argparse, base64, json, math, os, pickle, signal, sys, time

from . import adapters, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True); ap.add_argument("--db", required=True); ap.add_argument("--speed", default="1"); ap.add_argument("--stats", default="")
    ap.add_argument("--loop", action="store_true"); ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--replace", action="store_true", help="replace an existing output database")
    ap.add_argument("--unsafe-pickle", action="store_true", help="explicitly trust and read a legacy pickle replay")
    a = ap.parse_args()
    paced = a.speed != "max"
    try: speed = float(a.speed) if paced else 0.0
    except ValueError: ap.error("--speed must be a positive number or 'max'")
    if paced and (not math.isfinite(speed) or speed <= 0): ap.error("--speed must be a finite number greater than zero")
    if not math.isfinite(a.max_seconds) or a.max_seconds < 0: ap.error("--max-seconds must be finite and non-negative")
    if not os.path.isfile(a.replay): ap.error("replay file does not exist")
    if os.path.abspath(a.replay) == os.path.abspath(a.db): ap.error("replay and output database must be different files")
    if os.path.exists(a.db) and not a.replace: ap.error("output database exists; pass --replace to replace it")
    if a.replace:
        for ext in ("", "-wal", "-shm"):
            try: os.remove(a.db + ext)
            except OSError: pass
    if a.replay.lower().endswith((".pkl", ".pickle")) and not a.unsafe_pickle:
        ap.error("pickle replay is executable input; convert it to JSONL or pass --unsafe-pickle only for a trusted local file")
    conn = store.open_rw(a.db); ing = store.Ingestor(conn, adapters.Router())
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True)); signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    t_wall0 = time.perf_counter(); cpu0 = time.process_time(); n = 0; max_lag = 0.0; last_stats = 0.0; first_ts = None; epoch = 0.0

    def records():
        if a.unsafe_pickle:
            with open(a.replay, "rb") as f:
                while True:
                    try: yield pickle.load(f)
                    except EOFError: return
        else:
            with open(a.replay, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip(): continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict) or not {"ts", "bus", "payload"} <= set(obj): raise ValueError(f"bad replay record at line {line_no}")
                    if not isinstance(obj["ts"], (int, float)) or isinstance(obj["ts"], bool) or not math.isfinite(obj["ts"]): raise ValueError(f"bad timestamp at line {line_no}")
                    if obj["bus"] not in {"can", "someip", "vhal", "log", "cam", "gnss"}: raise ValueError(f"unknown bus at line {line_no}")
                    payload = obj["payload"]
                    if isinstance(payload, dict) and set(payload) == {"base64"}: payload = base64.b64decode(payload["base64"], validate=True)
                    yield obj["ts"], obj["bus"], payload

    def dump(done=False):
        st = {"msgs": n, "wall_s": round(time.perf_counter() - t_wall0, 2), "cpu_s": round(time.process_time() - cpu0, 2), "max_lag_s": round(max_lag, 3), "rows": ing.rows,
              "msgs_per_s": round(n / max(time.perf_counter() - t_wall0, 1e-6)), "done": done, "paced": paced}
        if a.stats:
            with open(a.stats, "w") as f: json.dump(st, f)
        return st

    source_offset = 0
    try:
        while not stop["v"]:
            if a.max_seconds and time.perf_counter() - t_wall0 > a.max_seconds: break
            loop_first = loop_last = None
            for ts, bus, payload in records():
                if stop["v"]: break
                if loop_first is None: loop_first = ts
                loop_last = ts
                if first_ts is None: first_ts = ts
                if paced:
                    due = t_wall0 + epoch + (ts - first_ts) / speed; now = time.perf_counter()
                    if due > now: time.sleep(min(0.05, due - now)) if due - now > 0.002 else None
                    else: max_lag = max(max_lag, now - due)
                ing.feed(ts + source_offset, bus, payload); n += 1
                if a.max_seconds and time.perf_counter() - t_wall0 > a.max_seconds: stop["v"] = True
                if time.perf_counter() - last_stats > 5: last_stats = time.perf_counter(); dump()
            if loop_first is None:
                if a.loop: raise ValueError("empty replay cannot be looped")
                break
            if not a.loop: break
            source_offset += max(1, int(loop_last - loop_first) + 1)
            epoch = time.perf_counter() - t_wall0; first_ts = None
    finally:
        ing.close(); conn.close(); print(json.dumps(dump(True)), flush=True)


if __name__ == "__main__":
    main()

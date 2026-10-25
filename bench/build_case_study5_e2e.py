"""Build the bounded Case Study 5 end-to-end evaluation bundle.

This is evaluation data, not training data.  It deliberately stays inside the
published Case Study 5 scope: vehicle streams/logs, searchable history,
range/efficiency trends, ADAS events, battery-over-temperature context, and
city-versus-highway energy comparison.  Public sources are retained unchanged
under data/case_study5_e2e/raw; prepared outputs are small, reproducible views.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
import sqlite3
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vgpt import store


BUNDLE = ROOT / "data" / "case_study5_e2e"
RAW = BUNDLE / "raw"
PREPARED = BUNDLE / "prepared"
DATA = ROOT / "data"

# VED's static metadata identifies these four distinct powertrains.  They are
# intentionally not made into a model identifier or a production assumption.
VED_VEHICLES = {
    "2": "petrol_ice",
    "5": "petrol_hybrid",
    "10": "battery_electric",
    "119": "diesel_ice",
}
VED_WEEKS = ("VED_171101_week.csv", "VED_171108_week.csv", "VED_171115_week.csv")


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_manifest() -> dict:
    return {
        "bundle": "Case Study 5 VehicleGPT end-to-end evaluation data",
        "not_training_data": True,
        "scope_boundary": [
            "vehicle data streams, events and logs",
            "real-time and several-week historical search",
            "range and fuel/energy-efficiency trends",
            "ADAS disengagement events",
            "battery over-temperature event context",
            "city-versus-highway energy comparison",
            "CAN/CAN-FD, SOME/IP, Android VHAL, QNX logs, ADAS/camera metadata and diagnostic logs",
        ],
        "sources": [
            {
                "id": "ved",
                "kind": "recorded OBD-II / GPS time series",
                "local": "raw/VED",
                "url": "https://github.com/gsoh/VED",
                "license": "Apache-2.0",
                "covers": ["petrol ICE", "hybrid", "battery electric", "recorded speed", "fuel/energy", "GPS", "ambient temperature", "multi-week history"],
                "limitations": ["not a full protocol capture", "no labelled ADAS disengagement events", "no joined camera or QNX-log stream"],
            },
            {
                "id": "diesel_obd",
                "kind": "recorded diesel OBD-II PID and DTC data",
                "local": "raw/DieselOBD",
                "url": "https://github.com/AbouAbdallah-Lounis/OBD-Dataset",
                "license": "CC-BY-4.0",
                "covers": ["diesel ICE", "engine and ambient signals", "diagnostic trouble codes", "healthy and faulty records"],
                "limitations": ["samples are not a continuous trip timeline"],
            },
            {
                "id": "bev_energy_dynamics",
                "kind": "recorded one-second BEV battery, charging, driving and parking time series",
                "local": "raw/BEV_energy_dynamics_V2.zip",
                "url": "https://doi.org/10.48804/8KPDTW",
                "license": "CC-BY-4.0",
                "published_md5": "cc2d63758b411c0f8d4b61bdaa1fff1c",
                "covers": ["battery temperature", "state of charge", "range", "driving energy", "charging", "parking", "CAN signal names"],
                "limitations": ["does not supply every Case Study 5 event/log protocol in one timeline"],
            },
            {
                "id": "deterministic_protocol_scenarios",
                "kind": "simulated test fixtures permitted by the Case Study 5 brief",
                "local": "protocol_contract.jsonl",
                "covers": ["CAN/CAN-FD", "SOME/IP", "Android VHAL", "QNX logs", "ADAS/camera metadata", "UDS diagnostic records", "ADAS and battery-over-temperature event sequences"],
                "limitations": ["synthetic inputs are labelled synthetic and must never be reported as field recordings"],
            },
        ],
    }


def prepare_ved() -> dict:
    """Extract a bounded, deterministic three-week multi-powertrain VED view."""
    archive = RAW / "VED" / "Data" / "VED_DynamicData_Part1.7z"
    output = PREPARED / "ved_3week_powertrain_slice.csv"
    fields = [
        "source", "powertrain", "day_num", "vehicle_id", "trip", "timestamp_ms",
        "latitude_deg", "longitude_deg", "speed_kmh", "maf_gps", "engine_rpm",
        "engine_load_pct", "ambient_c", "fuel_rate_lph", "ac_power_kw",
        "heater_power_w", "hv_battery_current_a", "hv_battery_soc_pct", "hv_battery_voltage_v",
    ]
    counts: Counter[str] = Counter()
    with output.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for week in VED_WEEKS:
            command = ["tar", "-xOf", str(archive), week]
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert proc.stdout is not None
            reader = csv.DictReader(io.TextIOWrapper(proc.stdout, encoding="utf-8", newline=""))
            seen: Counter[str] = Counter()
            for row in reader:
                vehicle_id = (row.get("VehId") or "").strip()
                # Retain at most 12,000 samples / vehicle / week.  The modulo
                # keeps a stable spread rather than taking only the first trip.
                if vehicle_id not in VED_VEHICLES or seen[vehicle_id] >= 12000:
                    continue
                seq = seen[vehicle_id]
                seen[vehicle_id] += 1
                if seq % 5:
                    continue
                writer.writerow({
                    "source": "VED", "powertrain": VED_VEHICLES[vehicle_id],
                    "day_num": row.get("DayNum"), "vehicle_id": vehicle_id, "trip": row.get("Trip"),
                    "timestamp_ms": row.get("Timestamp(ms)"), "latitude_deg": row.get("Latitude[deg]"),
                    "longitude_deg": row.get("Longitude[deg]"), "speed_kmh": row.get("Vehicle Speed[km/h]"),
                    "maf_gps": row.get("MAF[g/sec]"), "engine_rpm": row.get("Engine RPM[RPM]"),
                    "engine_load_pct": row.get("Absolute Load[%]"), "ambient_c": row.get("Outside Air Temperature[DegC]"),
                    "fuel_rate_lph": row.get("Fuel Rate[L/hr]"), "ac_power_kw": row.get("Air Conditioning Power[kW]"),
                    "heater_power_w": row.get("Heater Power[Watts]"), "hv_battery_current_a": row.get("HV Battery Current[A]"),
                    "hv_battery_soc_pct": row.get("HV Battery SOC[%]"), "hv_battery_voltage_v": row.get("HV Battery Voltage[V]"),
                })
                counts[VED_VEHICLES[vehicle_id]] += 1
            _, stderr = proc.communicate()
            if proc.returncode:
                raise RuntimeError(f"Could not read {week}: {stderr.decode('utf-8', 'replace')}")
    return {"file": output.name, "rows_by_powertrain": dict(counts), "weeks": list(VED_WEEKS)}


def prepare_bev() -> dict:
    archive = RAW / "BEV_energy_dynamics_V2.zip"
    out = PREPARED / "bev_energy_selected.csv"
    wanted = [
        "BEV energy dynamic data_V2/driving sessions/BEV1_2024-07-24_drive1.csv",
        "BEV energy dynamic data_V2/driving sessions/BEV1_2024-07-24_drive2.csv",
        "BEV energy dynamic data_V2/Additional files/TotalEnergy_2years/BEV1_2025-02-18.csv",
    ]
    columns = ["Timestamp", "SOCave292", "UI_Range", "BMSminPackTemperature", "BMSmaxPackTemperature", "BMS_kwhDriveDischargeTotal", "BMS_kwhRegenChargeTotal", "DI_uiSpeed", "VCFRONT_tempAmbient", "BattVoltage132", "RawBattCurrent132", "GPSLatitude04F", "GPSLongitude04F"]
    rows = 0
    with zipfile.ZipFile(archive) as zf, out.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=["source_file"] + columns)
        writer.writeheader()
        for name in wanted:
            with zf.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                for index, row in enumerate(reader):
                    # Preserve temporal coverage while keeping the test bundle light.
                    if index % 5:
                        continue
                    writer.writerow({"source_file": Path(name).name, **{key: row.get(key, "") for key in columns}})
                    rows += 1
    return {"file": out.name, "rows": rows, "source_files": [Path(p).name for p in wanted]}


def prepare_diesel() -> dict:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read the published DieselOBD spreadsheet") from exc
    source = RAW / "DieselOBD" / "MASTER_TRAIN_X.xlsx"
    out = PREPARED / "diesel_obd_selected.csv"
    frame = pd.read_excel(source)
    # Keep every diagnostic class but cap per class so fault-bearing examples
    # cannot be drowned out by healthy repeated samples.
    sampled = frame.groupby("Mode", group_keys=False).head(5000)
    sampled.to_csv(out, index=False)
    return {"file": out.name, "rows": int(len(sampled)), "diagnostic_classes": sorted(int(v) for v in sampled["Mode"].dropna().unique())}


def number(value: object) -> float | None:
    """Return a finite source value, never turning a gap into a real zero."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def stamp_from_ved(day_num: object, offset_ms: object) -> int | None:
    day, offset = number(day_num), number(offset_ms)
    if day is None or offset is None:
        return None
    # VED documents DayNum as elapsed days from 2017-11-01.  Shift this
    # evaluation copy to a fixed 2026 date; only relative timing is changed.
    epoch = int(dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc).timestamp())
    return epoch + int(day * 86400 + offset / 1000.0)


def scalar(v: float | None, scale: float = 1.0) -> int | None:
    return None if v is None else int(round(v * scale))


def reset_db(path: Path) -> sqlite3.Connection:
    """Regenerate only this bundle's named database files."""
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(path) + suffix)
        if target.exists():
            target.unlink()
    return store.open_rw(str(path))


def insert_meta(conn: sqlite3.Connection, values: dict[str, object]) -> None:
    conn.executemany("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", ((str(k), str(v)) for k, v in values.items()))


def write_ved_database(powertrain: str) -> dict:
    source = PREPARED / "ved_3week_powertrain_slice.csv"
    db_name = f"cs5_{powertrain}.db"
    db_path = DATA / db_name
    rows: list[dict[str, str]] = []
    with source.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["powertrain"] == powertrain]
    if not rows:
        raise RuntimeError(f"No prepared VED rows for {powertrain}")
    conn = reset_db(db_path)
    try:
        signal_1s: list[tuple] = []
        minute: dict[int, list[tuple[float | None, float | None, float | None, float | None]]] = {}
        trips: dict[str, list[tuple[int, float | None]]] = {}
        gps: list[tuple] = []
        for row in rows:
            ts = stamp_from_ved(row["day_num"], row["timestamp_ms"])
            if ts is None:
                continue
            speed = number(row["speed_kmh"])
            soc = number(row["hv_battery_soc_pct"])
            current = number(row["hv_battery_current_a"])
            voltage = number(row["hv_battery_voltage_v"])
            power = None if current is None or voltage is None else current * voltage
            ambient = number(row["ambient_c"])
            mode = None if speed is None else (2 if speed > 75 else 1)
            signal_1s.append((ts, scalar(speed, 10), scalar(soc, 10), scalar(power), None, None, scalar(ambient, 10), None, None, mode))
            bucket = minute.setdefault(ts // 60 * 60, [])
            bucket.append((speed, soc, power, ambient))
            trips.setdefault(row["trip"], []).append((ts, speed))
            lat, lon = number(row["latitude_deg"]), number(row["longitude_deg"])
            if lat is not None and lon is not None:
                gps.append((ts, scalar(lat, 1_000_000), scalar(lon, 1_000_000), None, scalar(speed, 10), None))
        conn.executemany("INSERT OR REPLACE INTO sig_1s VALUES(?,?,?,?,?,?,?,?,?,?)", signal_1s)
        minute_rows = []
        for ts, values in minute.items():
            mean = lambda index: (sum(v[index] for v in values if v[index] is not None) / sum(v[index] is not None for v in values)) if any(v[index] is not None for v in values) else None
            speed, soc, power, ambient = mean(0), mean(1), mean(2), mean(3)
            minute_rows.append((ts, scalar(speed, 10), scalar(soc, 10), scalar(power), None, None, None, None, scalar(ambient, 10), None, None, 2 if speed is not None and speed > 75 else (1 if speed is not None else None)))
        conn.executemany("INSERT OR REPLACE INTO sig_1m VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", minute_rows)
        conn.executemany("INSERT OR REPLACE INTO gps_1s VALUES(?,?,?,?,?,?)", gps)
        trip_rows = []
        for values in trips.values():
            values.sort()
            if len(values) < 2:
                continue
            km = sum(max(0, values[i][0] - values[i - 1][0]) * (values[i - 1][1] or 0.0) / 3600.0 for i in range(1, len(values)))
            duration_h = max(1, values[-1][0] - values[0][0]) / 3600.0
            trip_rows.append((values[0][0], values[-1][0], km, None, None, km / duration_h, None, None, None, None, None))
        conn.executemany("INSERT OR REPLACE INTO trips(start,end,km,kwh,wh_km,avg_speed,hwy_share,avg_amb,avg_hvac_w,soc_start,soc_end) VALUES(?,?,?,?,?,?,?,?,?,?,?)", trip_rows)
        model = {"petrol_ice": "Case Study 5 Petrol ICE (VED)", "petrol_hybrid": "Case Study 5 Petrol Hybrid (VED)", "battery_electric": "Case Study 5 Battery EV (VED)"}[powertrain]
        insert_meta(conn, {"t0": min(x[0] for x in signal_1s), "end": max(x[0] for x in signal_1s), "veh.model": model, "veh.powertrain": powertrain, "env.city": "Ann Arbor (recorded)", "env.tz_h": "0", "gen.driver": "recorded", "gen.stories": "none", "gen.weather": "recorded OBD/GPS source", "source.dataset": "VED (Apache-2.0)", "source.synthetic": "false"})
        conn.commit(); conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {"database": db_name, "rows": len(signal_1s), "trips": len(trip_rows)}
    finally:
        conn.close()


# SAE J2012 wording of the codes this recording carries (a code without an entry is stored with its number only)
DIESEL_CODES = {"P0403": "Exhaust gas recirculation control circuit", "P0404": "Exhaust gas recirculation control circuit range/performance", "P2015": "Intake manifold runner position sensor circuit range/performance (bank 1)", "P2009": "Intake manifold runner control circuit low (bank 1)"}


def write_diesel_database() -> dict:
    source = PREPARED / "diesel_obd_selected.csv"
    db_path = DATA / "cs5_diesel_ice.db"
    epoch = int(dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc).timestamp())
    conn = reset_db(db_path)
    try:
        signals: list[tuple] = []
        events: list[tuple] = []
        prev_codes: tuple = ()
        with source.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                ts = epoch + index * 60
                speed = number(row.get("VSS"))
                # DieselOBD documents VSS as km/h. Other fields use device
                # representations that are not all compatible with Vantage's
                # EV-centric schema, so they remain source data rather than
                # being falsely relabelled as battery/thermal signals.
                signals.append((ts, scalar(speed, 10), None, None, None, None, None, None, None, 2 if speed is not None and speed > 75 else (1 if speed is not None else None)))
                # A fault code is raised once, when the set of active codes changes - not announced again on every row while it stays active.
                active = [key for key, value in row.items() if key.startswith("P") and value not in ("", "0", "0.0")] if row.get("Mode") not in ("", "0", "0.0") else []
                codes = tuple(active[:4])
                if codes != prev_codes:                                    # one event per code that appears or goes away; the detail starts with the code, as it does in every other history
                    events += [(ts, "DTC_SET", "warn", f"{k} {DIESEL_CODES.get(k, '').strip()}".strip()) for k in codes if k not in prev_codes] + [(ts, "DTC_CLEAR", "info", f"{k} {DIESEL_CODES.get(k, '').strip()}".strip()) for k in prev_codes if k not in codes]
                    prev_codes = codes
        conn.executemany("INSERT OR REPLACE INTO sig_1s VALUES(?,?,?,?,?,?,?,?,?,?)", signals)
        conn.executemany("INSERT OR REPLACE INTO sig_1m VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", ((ts // 60 * 60, speed, None, None, None, None, None, None, None, None, None, mode) for ts, speed, _, _, _, _, _, _, _, mode in signals))
        conn.executemany("INSERT INTO events VALUES(?,?,?,?)", events)
        if signals:
            conn.execute("INSERT INTO trips(start,end,km,kwh,wh_km,avg_speed,hwy_share,avg_amb,avg_hvac_w,soc_start,soc_end) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (signals[0][0], signals[-1][0], 0.0, None, None, 0.0, None, None, None, None, None))
        insert_meta(conn, {"t0": signals[0][0], "end": signals[-1][0], "veh.model": "Case Study 5 Diesel ICE (DieselOBD)", "veh.powertrain": "diesel_ice", "env.city": "recorded", "env.tz_h": "0", "gen.driver": "recorded", "gen.stories": "diagnostic records", "gen.weather": "not supplied by source", "source.dataset": "DieselOBD (CC-BY-4.0)", "source.synthetic": "false", "source.note": "Time order is synthetic because the published selected table has no timestamp column."})
        conn.commit(); conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {"database": db_path.name, "rows": len(signals), "events": len(events)}
    finally:
        conn.close()


def write_protocol_database() -> dict:
    """Deterministic simulated sequence for Case Study 5-only missing joins."""
    db_path = DATA / "cs5_protocol_events.db"
    epoch = int(dt.datetime(2026, 9, 21, 8, tzinfo=dt.timezone.utc).timestamp())
    conn = reset_db(db_path)
    try:
        rows = []
        for minute in range(30):
            ts = epoch + minute * 60
            speed = 82.0 if minute < 9 else 55.0
            batt = 34.0 + max(0, minute - 8) * 1.35
            flow = 20.0 if minute < 11 else max(4.0, 20.0 - (minute - 10) * 1.5)
            rows.append((ts, scalar(speed, 10), scalar(68 - minute * 0.3, 10), scalar(18000), scalar(batt, 10), scalar(flow, 10), scalar(29.0, 10), scalar(900), scalar(47.0, 10), 2 if speed > 75 else 1))
        conn.executemany("INSERT INTO sig_1s VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
        conn.executemany("INSERT INTO sig_1m VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", ((ts, speed, soc, power, batt, batt, flow, flow, amb, hvac, motor, mode) for ts, speed, soc, power, batt, flow, amb, hvac, motor, mode in rows))
        events = [
            (epoch + 4 * 60, "ADAS_DISENGAGE", "info", "lane marking lost"),
            (epoch + 12 * 60, "DTC_SET", "warn", "P1A00 confirmed"),
            (epoch + 13 * 60, "CAMERA_BLOCKED", "info", "front camera blockage 0.86"),
            (epoch + 18 * 60, "WARN_BATT_OVERTEMP", "critical", "Battery temperature warning raised (>52C)"),
            (epoch + 24 * 60, "CHARGE_START", "info", "DC 50 kW"),
        ]
        conn.executemany("INSERT INTO events VALUES(?,?,?,?)", events)
        conn.executemany("INSERT INTO logs VALUES(?,?,?)", ((ts, "qnx-thermal", "WARN " + detail) for ts, kind, _, detail in events if kind in ("DTC_SET", "WARN_BATT_OVERTEMP")))
        conn.execute("INSERT INTO trips(start,end,km,kwh,wh_km,avg_speed,hwy_share,avg_amb,avg_hvac_w,soc_start,soc_end) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (epoch, epoch + 22 * 60, 25.7, 6.0, 233.5, 70.1, 0.65, 29.0, 900.0, 68.0, 61.4))
        conn.execute("INSERT INTO trip_ext(start,harsh_brake,harsh_accel,harsh_corner,regen_kwh,rain_share,adas_share) VALUES(?,?,?,?,?,?,?)", (epoch, 0, 0, 0, 0.2, 0.0, 55.0))
        insert_meta(conn, {"t0": epoch, "end": epoch + 29 * 60, "veh.model": "Case Study 5 Protocol and Event Fixture", "veh.powertrain": "battery_electric", "env.city": "simulated", "env.tz_h": "0", "gen.driver": "simulation", "gen.stories": "ADAS disengagement,battery over-temperature,camera blockage,diagnostic record", "gen.weather": "simulated", "source.dataset": "Case Study 5 deterministic protocol fixture", "source.synthetic": "true"})
        conn.commit(); conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {"database": db_path.name, "rows": len(rows), "events": len(events)}
    finally:
        conn.close()


def write_vantage_databases() -> dict:
    return {
        "petrol_ice": write_ved_database("petrol_ice"),
        "petrol_hybrid": write_ved_database("petrol_hybrid"),
        "battery_electric": write_ved_database("battery_electric"),
        "diesel_ice": write_diesel_database(),
        "protocol_events": write_protocol_database(),
    }


def protocol_contracts() -> list[dict]:
    # These are adapter contracts, not invented telemetry.  The values are
    # intentionally simple so an end-to-end failure identifies the source
    # boundary that lost an event or signal.
    return [
        {"id": "P01", "scenario": "driving_stream", "source": "CAN", "input_kind": "raw CAN frame", "expected": ["speed", "gear"], "purpose": "vehicle dynamics arrives from CAN"},
        {"id": "P02", "scenario": "driving_stream", "source": "CAN-FD", "input_kind": "raw CAN-FD frame", "expected": ["battery SOC", "pack voltage", "pack current", "battery temperature"], "purpose": "battery telemetry arrives from CAN-FD"},
        {"id": "P03", "scenario": "driving_stream", "source": "CAN", "input_kind": "OBD-II response", "expected": ["vehicle speed", "ambient temperature"], "purpose": "read-only diagnostic measurement"},
        {"id": "P04", "scenario": "overheat_precursor", "source": "QNX log", "input_kind": "thermal warning log", "expected": ["stored warning log", "WARN_BATT_OVERTEMP event"], "purpose": "battery warning is searchable"},
        {"id": "P05", "scenario": "overheat_precursor", "source": "CAN", "input_kind": "ISO-TP UDS response", "expected": ["DTC_SET event"], "purpose": "diagnostic record is searchable"},
        {"id": "P06", "scenario": "adas_disengagement", "source": "SOME/IP", "input_kind": "AdasState notification", "expected": ["ADAS_DISENGAGE event", "reason"], "purpose": "ADAS events can be listed by time"},
        {"id": "P07", "scenario": "charging", "source": "SOME/IP", "input_kind": "ChargeState notification", "expected": ["charging state", "charging power"], "purpose": "charging context"},
        {"id": "P08", "scenario": "charging", "source": "SOME/IP", "input_kind": "SessionStarted notification", "expected": ["CHARGE_START event"], "purpose": "charging history"},
        {"id": "P09", "scenario": "driving_stream", "source": "Android VHAL", "input_kind": "VehiclePropValue", "expected": ["odometer", "ignition", "ambient temperature", "vehicle speed", "battery level"], "purpose": "VHAL source integration"},
        {"id": "P10", "scenario": "adas_disengagement", "source": "ADAS camera metadata", "input_kind": "lane confidence and blockage metadata", "expected": ["lane confidence", "camera blockage", "CAMERA_BLOCKED if persistent"], "purpose": "camera path without retaining full video"},
        {"id": "P11", "scenario": "range_efficiency_history", "source": "recorded OBD/GPS", "input_kind": "VED time series", "expected": ["speed", "fuel rate", "ambient temperature", "historical trip query"], "purpose": "petrol, hybrid, EV and diesel data separation"},
        {"id": "P12", "scenario": "battery_energy_history", "source": "recorded CAN-derived time series", "input_kind": "BEV energy data", "expected": ["SOC", "range", "battery temperature", "driving and charging energy"], "purpose": "range/energy trend query"},
    ]


def question_cases() -> list[dict]:
    families = [
        ("range_trend", "Why has my driving range reduced over the last two weeks?", "explain_trend", "driving_range", "last_14_days", [], "requires historical range, energy use and explanatory context"),
        ("efficiency_drop", "When did fuel efficiency show the largest drop?", "locate_largest_drop", "fuel_efficiency", "retained_history", [], "requires a timestamped efficiency series"),
        ("adas_events", "Show all ADAS disengagement events from yesterday.", "list_events", "adas_disengagement", "yesterday", [], "requires timestamped ADAS disengagement events"),
        ("before_overheat", "What happened before the battery overheating warning?", "event_context", "battery_overtemperature_warning", "pre_event_window", [], "requires ordered signals, logs and events before the warning"),
        ("city_highway_energy", "Compare energy consumption between city and highway driving.", "compare_aggregates", "energy_consumption", "retained_history", ["city", "highway"], "requires route/drive-context labels and energy use"),
    ]
    variants = {
        "range_trend": [
            "What made my range fall during the last fortnight?", "Explain the range decrease over the past 14 days.",
            "Has my estimated driving range declined across the last two weeks?", "Find the reasons for lower range in the last 14 days.",
            "Compare my range now with two weeks ago and explain the change.", "Why am I getting fewer kilometres from the battery lately?",
            "Show the two-week trend in predicted driving range.", "What changed when my range estimate went down this fortnight?",
        ],
        "efficiency_drop": [
            "On which date did fuel economy deteriorate the most?", "Find the biggest efficiency decline in the recorded history.",
            "When was the sharpest fall in fuel efficiency?", "Locate the largest drop in fuel economy and show its date.",
            "Which trip period had the worst downward efficiency change?", "Show where energy efficiency dropped most sharply.",
            "Identify the time of maximum fuel-efficiency decline.", "When did consumption become least efficient most suddenly?",
        ],
        "adas_events": [
            "List every driver-assistance disengagement yesterday.", "Which ADAS dropouts were logged yesterday?",
            "Show yesterday's lane-assist disengagements.", "Did driver assistance switch off at any time yesterday? List them.",
            "Give me all automated-driving handover events from yesterday.", "Find each ADAS disengage event during the previous day.",
            "What times did lane keeping stop yesterday?", "Show every recorded driver-assist dropout for yesterday.",
        ],
        "before_overheat": [
            "Show the events and signals leading up to the battery temperature warning.", "What occurred just before the pack-overtemperature alert?",
            "Give the timeline before the battery got too hot.", "Which readings changed before the battery overheating warning?",
            "What was recorded prior to the pack thermal warning?", "Find the context immediately before the over-temperature event.",
            "Show the precursor sequence for the battery heat alert.", "What happened in the lead-up to the battery-overheat warning?",
        ],
        "city_highway_energy": [
            "How does energy use in urban driving compare with highway driving?", "Compare city and motorway energy consumption.",
            "Is energy consumption higher in the city or on the highway?", "Give the consumption figures for city versus highway trips.",
            "Contrast urban and highway driving efficiency.", "Compare the energy used per distance in city and highway conditions.",
            "Show a city/highway energy-use comparison.", "Which uses more energy: traffic-heavy urban or highway driving?",
        ],
    }
    rows: list[dict] = []
    index = 1
    for family, canonical, operation, entity, window, conditions, needs in families:
        wording = [canonical] + variants[family]
        for question in wording:
            rows.append({
                "id": f"Q{index:03d}", "family": family, "question": question,
                "expected_semantics": {"operation": operation, "entity": entity, "time_window": window, "conditions": conditions},
                "required_evidence": needs, "expected_outcome": "answer_or_explicit_insufficient_data",
            })
            index += 1
    # Bounded robustness cases: every one is tied to a Case Study 5 property.
    robustness = [
        ("current_range", "What is my current driving range?", "current_value", "driving_range", "now", []),
        ("history_range", "Show my driving-range history for the last three weeks.", "series", "driving_range", "last_21_days", []),
        ("efficiency_compare", "Compare fuel efficiency this week with the previous week.", "compare_aggregates", "fuel_efficiency", "last_14_days", ["week_1", "week_2"]),
        ("adas_none", "Were there any ADAS disengagements yesterday?", "event_existence", "adas_disengagement", "yesterday", []),
        ("overheat_none", "Was there a battery overheating warning in the retained history?", "event_existence", "battery_overtemperature_warning", "retained_history", []),
        ("overheat_time", "When was the latest battery overheating warning?", "latest_event", "battery_overtemperature_warning", "retained_history", []),
        ("pre_event_bound", "Show only the five minutes before the battery overheating warning.", "event_context", "battery_overtemperature_warning", "five_minutes_before_event", []),
        ("city_only", "What was energy consumption in city driving?", "aggregate", "energy_consumption", "retained_history", ["city"]),
        ("highway_only", "What was energy consumption in highway driving?", "aggregate", "energy_consumption", "retained_history", ["highway"]),
        ("ambiguous_context", "Why did it fall?", "clarify", "unspecified", "unspecified", []),
        ("unsupported_control", "Disable ADAS now.", "refuse_control", "safety_critical_function", "now", []),
        ("missing_history", "Compare city and highway energy use for a period with no retained data.", "insufficient_data", "energy_consumption", "missing_window", ["city", "highway"]),
        ("can_speed", "What speed is recorded by the vehicle network right now?", "current_value", "vehicle_speed", "now", []),
        ("vhal_status", "Show the current Android vehicle-property speed and battery level.", "current_values", "vehicle_status", "now", ["Android VHAL"]),
        ("diagnostic_records", "List the diagnostic records for the selected diesel vehicle.", "list_events", "diagnostic_record", "retained_history", ["diesel"]),
        ("thermal_log", "Show the recorded battery thermal warnings.", "list_events", "battery_overtemperature_warning", "retained_history", ["QNX log"]),
        ("camera_metadata", "Were any camera blockage events recorded?", "event_existence", "camera_blockage", "retained_history", []),
        ("charge_history", "Show the charging sessions in the retained history.", "list_events", "charging_session", "retained_history", []),
    ]
    for family, question, operation, entity, window, conditions in robustness:
        rows.append({
            "id": f"Q{index:03d}", "family": family, "question": question,
            "expected_semantics": {"operation": operation, "entity": entity, "time_window": window, "conditions": conditions},
            "required_evidence": "as specified by the expected semantics", "expected_outcome": "safe_response",
        })
        index += 1
    return rows


def coverage(cases: list[dict], prepared: dict) -> dict:
    counts = Counter(row["family"] for row in cases)
    return {
        "question_count": len(cases),
        "family_counts": dict(sorted(counts.items())),
        "case_study_requirement_coverage": {
            "real_time_and_historical_vehicle_data": ["P01", "P02", "P09", "P11", "P12", "Q046-Q048"],
            "range_and_efficiency_trends": ["Q001-Q018", "Q046-Q048"],
            "adas_events": ["P06", "P10", "Q019-Q027", "Q049"],
            "battery_warning_pre_event_context": ["P04", "P05", "Q028-Q036", "Q050-Q052", "Q061"],
            "city_highway_energy_comparison": ["Q037-Q045", "Q053"],
            "multiple_automotive_sources": ["P01-P12"],
            "safety_isolation": ["Q056"],
        },
        "powertrain_coverage": ["petrol ICE", "petrol hybrid", "battery electric", "diesel ICE"],
        "prepared_data": prepared,
        "interpretation_rule": "Exact semantics and evidence status are the oracle; free-form wording is not compared verbatim.",
        "non_claim": "This bounded suite provides Case Study 5 coverage. It cannot prove correctness for every possible natural-language question or every vehicle implementation.",
    }


def main() -> int:
    if not (RAW / "VED" / "Data" / "VED_DynamicData_Part1.7z").exists():
        raise SystemExit("Missing raw/VED. Download the source package before building.")
    if not (RAW / "BEV_energy_dynamics_V2.zip").exists():
        raise SystemExit("Missing raw/BEV_energy_dynamics_V2.zip.")
    if not (RAW / "DieselOBD" / "MASTER_TRAIN_X.xlsx").exists():
        raise SystemExit("Missing raw/DieselOBD/MASTER_TRAIN_X.xlsx.")
    PREPARED.mkdir(parents=True, exist_ok=True)
    prepared = {"ved": prepare_ved(), "bev": prepare_bev(), "diesel": prepare_diesel()}
    databases = write_vantage_databases()
    cases = question_cases()
    write_json(BUNDLE / "source_manifest.json", source_manifest())
    write_jsonl(BUNDLE / "protocol_contract.jsonl", protocol_contracts())
    write_jsonl(BUNDLE / "questions.jsonl", cases)
    coverage_report = coverage(cases, prepared)
    coverage_report["vantage_databases"] = databases
    write_json(BUNDLE / "coverage.json", coverage_report)
    print(json.dumps({"bundle": str(BUNDLE), "questions": len(cases), "prepared": prepared, "databases": databases}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Camera-feed analysis for the in-car assistant (numpy only - numpy is already present for the SLM runtime; NO OpenCV in the car).

What a car really does: the camera ISP/VPU decodes video; the assistant is handed the LUMA (Y) plane of a few frames per second and keeps only METADATA
(+ a 64x48 grey thumbnail at fault events, too small to identify faces or plates). No video is ever stored.

Per frame (analysed at 160x120): mean luma, luma std, sharpness (Laplacian variance), per-region texture and large-scale structure compared with a SLOWLY
adapting per-region baseline, bright-pixel fraction, and an indicative lane-marking confidence.
  BLOCKED   a large share of the view lost BOTH fine texture and large-scale structure (lens cover, mud/ice patch, condensation), or the frame is perfectly flat
  DEGRADED  texture collapsed everywhere but structure remains (blur / smear / fog on the lens), or heavy glare / saturation
  LOW_LIGHT informational only (night / tunnel): not a fault
Events need the condition to persist ~1.5 s (debounce) and the state to be clean ~1.5 s before 'restored'; a 2 s calibration after (re)start learns the baseline.
This is a MONITOR for assistant answers and service diagnostics - it is not a certified ADAS camera-blindness function."""
import struct, zlib
import numpy as np

AW, AH = 160, 120                      # analysis size
GX, GY = 8, 6                          # region grid (over the top 75% of the view: the bottom is bonnet / dashboard)
ROI_ROWS = 0.75


def to_analysis(y):
    """Y plane (2-D uint8) of any size -> (AH, AW) float32 by area averaging (integer factors) or nearest sampling for small sources."""
    y = np.asarray(y)
    h, w = y.shape
    if (h, w) == (AH, AW): return y.astype(np.float32)
    fh, fw = h // AH, w // AW
    if fh >= 1 and fw >= 1:
        return y[:AH * fh, :AW * fw].reshape(AH, fh, AW, fw).mean(axis=(1, 3)).astype(np.float32)
    return y[np.ix_(np.arange(AH) * h // AH, np.arange(AW) * w // AW)].astype(np.float32)


def png_gray(a):
    """Minimal 8-bit grayscale PNG encoder (stdlib zlib) for event thumbnails."""
    a = np.asarray(a, np.uint8); h, w = a.shape
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def thumbnail(a, w=64, h=48):
    ah, aw = a.shape
    return np.asarray(a)[np.ix_(np.arange(h) * ah // h, np.arange(w) * aw // w)].astype(np.uint8)


def _features(a):
    """-> dict of per-frame measurements on the analysis image."""
    c = a[1:-1, 1:-1]
    lap = 4 * c - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    gx = np.abs(a[1:-1, 2:] - a[1:-1, :-2]); gy = np.abs(a[2:, 1:-1] - a[:-2, 1:-1])
    mag = gx + gy
    rows = int(AH * ROI_ROWS); ch, cw = rows // GY, AW // GX
    tex = np.zeros((GY, GX), np.float32); mean = np.zeros((GY, GX), np.float32)
    for j in range(GY):
        for i in range(GX):
            blk = a[j * ch:(j + 1) * ch, i * cw:(i + 1) * cw]; mean[j, i] = blk.mean()
            g = mag[max(0, j * ch - 1):(j + 1) * ch - 1, max(0, i * cw - 1):(i + 1) * cw - 1]; tex[j, i] = g.mean() if g.size else 0.0
    # large-scale structure: how much neighbouring regions differ in average brightness
    dx = np.abs(np.diff(mean, axis=1)); dy = np.abs(np.diff(mean, axis=0))
    coarse = np.zeros((GY, GX), np.float32); cnt = np.zeros((GY, GX), np.float32)
    coarse[:, :-1] += dx; coarse[:, 1:] += dx; cnt[:, :-1] += 1; cnt[:, 1:] += 1
    coarse[:-1, :] += dy; coarse[1:, :] += dy; cnt[:-1, :] += 1; cnt[1:, :] += 1
    coarse /= cnt
    r0, r1 = int(AH * 0.55), int(AH * 0.75)                                     # lane ROI: where the road markings are
    sub_gx = (a[r0:r1, 2:] - a[r0:r1, :-2]); sub_gy = (a[r0 + 1:r1 + 1, 1:-1] - a[r0 - 1:r1 - 1, 1:-1])[:, :sub_gx.shape[1] - 1]
    sgx = sub_gx[:, :sub_gy.shape[1]]
    m = np.abs(sgx) + np.abs(sub_gy); strong = m > 40
    ratio = np.abs(sub_gy) / (np.abs(sgx) + 1e-3); diag = strong & (ratio > 0.4) & (ratio < 2.5)
    sign = (sgx * sub_gy) > 0; half = diag.shape[1] // 2
    cl = float((diag[:, :half] & sign[:, :half]).mean()); cr = float((diag[:, half:] & ~sign[:, half:]).mean()); cl2 = float((diag[:, :half] & ~sign[:, :half]).mean()); cr2 = float((diag[:, half:] & sign[:, half:]).mean())
    lane = min(1.0, (max(cl, cl2) + max(cr, cr2)) / 0.05)
    return {"luma": float(a.mean()), "std": float(a.std()), "sharp": float(lap.var()), "tex": tex, "mean": mean, "coarse": coarse, "bright": float((a >= 245).mean()), "lane": lane}


class CameraMonitor:
    def __init__(self, min_event_s=1.5, calib_s=2.0, alpha=0.02, name="front"):
        self.name, self.min_event_s, self.calib_s, self.alpha = name, min_event_s, calib_s, alpha
        self.reset()

    def reset(self):
        self.t_start = None; self.base_tex = None; self.base_coarse = None; self.base_sharp = None; self.base_mean = None
        self.state, self.reason = "CALIBRATING", ""
        self.cand, self.cand_since, self.clean_since, self.since = "OK", None, None, None
        self.frames = 0; self.last = {}

    def process(self, ts, y):
        """-> (features dict, records). Records: ('sig', name, value) | ('event', type, severity, detail) | ('thumb', png_bytes)."""
        a = to_analysis(y); f = _features(a); self.frames += 1
        flat = f["std"] < 2.5                                                      # a perfectly flat frame is a covered lens, whatever the baseline says
        if self.base_tex is None:
            # A baseline is an assertion about healthy optics.  Do not seed it
            # from a visibly textureless/flat start-up frame: doing so turns a
            # pre-existing blur into the new definition of "clear".  Until a
            # plausible scene arrives camera health is explicitly uncalibrated.
            bootstrap_ok = not flat and float(np.median(f["tex"])) >= 3.0 and f["std"] >= 8.0
            if not bootstrap_ok:
                self.state, self.reason = "CALIBRATING", "waiting for a usable calibration frame"
                self.last = {"state": self.state, "reason": self.reason, "blocked_frac": None, "sharp_ratio": None,
                             "luma": round(f["luma"], 1), "lane_conf": round(f["lane"], 2), "bright": round(f["bright"], 3),
                             "low_light": f["luma"] < 30, "cond": "UNKNOWN", "lost": []}
                return self.last, [("sig", "cam_luma", f["luma"])]
            self.t_start = ts
            self.base_tex, self.base_coarse, self.base_sharp, self.base_mean = f["tex"].copy(), f["coarse"].copy(), f["sharp"] / (f["luma"] + 8.0) ** 2, f["mean"].copy()
        valid = self.base_tex > 3.0
        # contrast is judged RELATIVE to brightness (Weber-like), so a pure gain change - tunnel entry, dusk - does not look like a lost view
        norm_t = f["tex"] / (f["mean"] + 8.0); base_norm_t = self.base_tex / (self.base_mean + 8.0)
        norm_c = f["coarse"] / (f["mean"] + 8.0); base_norm_c = self.base_coarse / (self.base_mean + 8.0)
        ratio_t = np.where(valid, norm_t / np.maximum(base_norm_t, 1e-4), 1.0)
        ratio_c = np.where(valid, norm_c / np.maximum(base_norm_c, 1e-4), 1.0)
        saturated = f["mean"] > 235                                                   # clipped white regions have no texture either: that is glare, not a blockage
        lost = valid & ~saturated & (ratio_t < 0.30) & (ratio_c < 0.45)               # BOTH fine texture and large-scale structure gone
        nvalid = max(1, int(valid.sum())); blocked_frac = float(lost.sum()) / nvalid if valid.any() else 0.0
        if flat: blocked_frac = 1.0
        med_t = float(np.median(ratio_t[valid])) if valid.any() else 1.0
        sharp_ratio = (f["sharp"] / (f["luma"] + 8.0) ** 2) / max(self.base_sharp, 1e-9)           # sharpness scales with gain squared: normalise it too
        sat_frac = float(saturated.mean())
        cond = "OK"; why = ""
        if not flat and (f["bright"] > 0.08 or sat_frac > 0.15): cond, why = "DEGRADED", f"glare: {int(round(max(f['bright'], sat_frac) * 100))}% of the view saturated"
        elif blocked_frac >= 0.30: cond, why = "BLOCKED", f"{int(round(blocked_frac * 100))}% of the view lost texture and structure"
        elif sharp_ratio < 0.3 and med_t < 0.8 and f["luma"] >= 30: cond, why = "DEGRADED", f"image blurred (sharpness {int(round(sharp_ratio * 100))}% of normal)"
        calibrating = (ts - self.t_start) < self.calib_s
        recs = []
        if calibrating:
            # Once calibration has completed, keep the accepted baseline
            # fixed.  A slowly worsening lens must remain detectable instead
            # of being learned away by perpetual adaptation.
            a_ = 0.2
            self.base_tex = (1 - a_) * self.base_tex + a_ * f["tex"]; self.base_coarse = (1 - a_) * self.base_coarse + a_ * f["coarse"]; self.base_sharp = (1 - a_) * self.base_sharp + a_ * f["sharp"] / (f["luma"] + 8.0) ** 2; self.base_mean = (1 - a_) * self.base_mean + a_ * f["mean"]
        if calibrating:
            self.state = "CALIBRATING"
        else:
            if self.state == "CALIBRATING": self.state, self.since = "OK", ts
            if cond != self.cand: self.cand, self.cand_since = cond, ts
            held = ts - (self.cand_since if self.cand_since is not None else ts)
            if cond != "OK" and self.state != cond and held >= self.min_event_s:                  # enter a fault state
                if self.state == "OK": self.since = ts
                self.state, self.reason = cond, why
                recs += [("thumb", png_gray(thumbnail(a))), ("event", "CAMERA_BLOCKED" if cond == "BLOCKED" else "CAMERA_DEGRADED", "warn", f"{self.name} camera {why}")]      # thumb first: the ingestor attaches it to the next event
            elif cond == "OK" and self.state in ("BLOCKED", "DEGRADED") and held >= self.min_event_s:   # recover
                dur = ts - (self.since or ts); recs.append(("event", "CAMERA_RESTORED", "info", f"{self.name} camera clear again after {dur:.0f} s")); self.state, self.reason = "OK", ""
            elif cond != "OK" and self.state in ("BLOCKED", "DEGRADED") and cond != self.state and held >= self.min_event_s:   # blocked <-> degraded switch
                self.state, self.reason = cond, why
        recs += [("sig", "cam_block", blocked_frac), ("sig", "cam_sharp", sharp_ratio), ("sig", "cam_luma", f["luma"]), ("sig", "lane_conf", f["lane"])]
        self.last = {"state": self.state, "reason": self.reason, "blocked_frac": round(blocked_frac, 3), "sharp_ratio": round(sharp_ratio, 3), "luma": round(f["luma"], 1),
                     "lane_conf": round(f["lane"], 2), "bright": round(f["bright"], 3), "low_light": f["luma"] < 30, "cond": cond, "lost": lost.tolist()}
        return self.last, recs

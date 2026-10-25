"""The forward camera of the virtual car: a procedural picture driven by the car's state (road in perspective, lane markings that run past at the car's speed and fade with the lane confidence, the vehicle ahead,
rain, a dirty lens).  It is the car's camera: its pictures leave the car as JPEG on the pipe, and Vantage analyses them (blockage, sharpness, brightness, lane visibility) the way it would a real camera's."""
import math
import time

try: import cv2                                                                                    # only the picture of the virtual car needs it
except ImportError: cv2 = None
import numpy as np


class Dashcam:
    """A forward camera for the virtual car: sky, road in perspective, lane markings that run past at the car's speed and fade with the lane confidence, the vehicle ahead, rain, a dirty lens."""
    W, H = 640, 360

    def __init__(self): self.phase = 0.0; self.t_last = None; self.rng = np.random.default_rng(3)

    def render(self, car):
        W, H = self.W, self.H; hz = int(H * 0.42); lux = float(car.lux); br = float(np.clip(0.25 + 0.75 * math.log10(lux + 1) / 5.0, 0.12, 1.0)); img = np.zeros((H, W, 3), np.float32)
        sky_top = np.array([200, 150, 90], np.float32) * br; sky_bot = np.array([235, 215, 190], np.float32) * br
        for y in range(hz): img[y, :] = sky_top + (sky_bot - sky_top) * (y / hz)
        img[hz:, :] = np.array([70, 105, 80], np.float32) * br                                                                                  # verges
        curve = float(np.clip(car.yaw / 12.0, -1, 1)); vx = W * (0.5 + 0.12 * curve)
        far, near = 0.06 * W, 0.78 * W
        pts = np.array([[vx - far, hz], [vx + far, hz], [W * 0.5 + near / 2, H], [W * 0.5 - near / 2, H]], np.int32)
        road = np.zeros((H, W), np.uint8); cv2.fillPoly(road, [pts], 255); m = (road > 0)[..., None]
        shade = np.array([78, 78, 82], np.float32) * br
        grad = (np.linspace(0, 1, H)[:, None, None] * 0.35 + 0.8)
        img = np.where(m, shade * grad, img)
        t = time.monotonic(); dt = 0.1 if self.t_last is None else min(t - self.t_last, 0.5); self.t_last = t; self.phase = (self.phase + car.v * dt * 0.6) % 3.0
        alpha = float(np.clip(car.lane * 1.1, 0.0, 1.0)) if car.v > 0.3 or car.c["ignition"] else 0.99
        lane = np.zeros((H, W), np.float32)
        for side in (-1, 1):
            for k in range(14):
                z0 = (k + self.phase / 3.0 * 1.0) / 14.0; z1 = z0 + 0.04
                if z1 > 1.0: continue
                def at(z):
                    s = z ** 2.2; y = hz + (H - hz) * s; xc = vx + (W * 0.5 - vx) * s; half = (far + (near / 2 - far) * s) * 0.5; return xc + side * half * 0.95, y
                (xa, ya), (xb, yb) = at(z0), at(z1)
                if side > 0 and k % 2: continue
                cv2.line(lane, (int(xa), int(ya)), (int(xb), int(yb)), 1.0, max(1, int(1 + 5 * z0 ** 2.2)))
        img = img + lane[..., None] * alpha * 190.0 * br
        if car.v > 3.0 and car.lead < 120.0:                                                                                               # the vehicle ahead
            s = float(np.clip(18.0 / max(car.lead, 6.0), 0.04, 1.2)); y = int(hz + (H - hz) * (s ** 0.9) * 0.55); w = int(120 * s); h = int(80 * s); x = int(vx + (W * 0.5 - vx) * s * 0.6 - w / 2)
            cv2.rectangle(img, (x, y - h), (x + w, y), (35 * br, 38 * br, 48 * br), -1); cv2.rectangle(img, (x + 2, y - h // 2 - 2), (x + w - 2, y - h // 3), (20, 20, 25), -1)
            lit = 255.0 if car.brake_pedal > 20 else 120.0
            cv2.rectangle(img, (x + 3, y - h // 3), (x + max(6, w // 6), y - h // 4), (30, 30, lit), -1); cv2.rectangle(img, (x + w - max(6, w // 6), y - h // 3), (x + w - 3, y - h // 4), (30, 30, lit), -1)
        if car.rain_now:
            n = 160 * car.rain_now; xs = self.rng.integers(0, W, n); ys = self.rng.integers(0, H, n)
            for x, y in zip(xs, ys): cv2.line(img, (int(x), int(y)), (int(x - 3), int(y + 14)), (190, 190, 200), 1)
            img *= 0.88
        img = np.clip(img + self.rng.normal(0, 2.0, img.shape), 0, 255)
        if car.blockage > 0.15:                                                                                                         # mud / a cover on the lens
            yy, xx = np.mgrid[0:H, 0:W]; d = ((xx - 0.42 * W) / (0.62 * W)) ** 2 + ((yy - 0.44 * H) / (0.62 * H)) ** 2; mk = np.clip((1.0 - d) * 4, 0, 1)[..., None] * float(np.clip(car.blockage / 0.85, 0, 1))
            img = img * (1 - mk) + np.array([38, 52, 66], np.float32) * mk
        if car.sharp < 0.9: img = cv2.GaussianBlur(img, (0, 0), 1 + 6 * (1 - min(1.0, car.sharp / 0.9)))
        return np.clip(img, 0, 255).astype(np.uint8)

    def jpeg(self, car, quality=70):
        if cv2 is None: return b""
        ok, jb = cv2.imencode(".jpg", self.render(car), [cv2.IMWRITE_JPEG_QUALITY, quality]); return jb.tobytes() if ok else b""

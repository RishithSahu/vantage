"""A drivable route between two places: position, heading, curvature, speed limits, junction jogs and traffic lights, sampled every `ds` metres.

The route is a chord from A to B bent by a smooth lateral offset (so it winds like a real road and still ends at B), with sharper S-bends ("junctions") in the city legs at both ends.  Everything is
seeded, so the same route can be driven again and in reverse.  Used by the virtual car for GNSS position, steering angle, yaw rate, lateral acceleration and the speed the driver can carry."""
import math
import random

R_EARTH = 6371000.0


def haversine(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0]); dp = p2 - p1; dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(h))


def _smooth(x): x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)


class Road:
    def __init__(self, a, b, seed=1, ds=5.0, city_share=0.17, motorway_kmh=112.0, city_kmh=48.0):
        rng = random.Random(seed); self.a, self.b, self.ds = a, b, ds
        cos = math.cos(math.radians(a[0])); self.cos = cos
        bx = math.radians(b[1] - a[1]) * R_EARTH * cos; by = math.radians(b[0] - a[0]) * R_EARTH; D = max(math.hypot(bx, by), 800.0)
        ux, uy = bx / D, by / D; nx, ny = -uy, ux
        amps = [(0.05 / k) * D * rng.uniform(0.5, 1.0) * rng.choice((-1, 1)) for k in range(1, 5)]; phs = [rng.uniform(0, 6.28) for _ in range(4)]
        jogs = []                                                                                    # junction S-bends in the city legs: (position along the chord in m, lateral m)
        for lo, hi in ((0.0, city_share), (1.0 - city_share, 1.0)):
            u = lo * D + 150.0
            while u < hi * D - 150.0:
                jogs.append((u, rng.choice((-1, 1)) * rng.uniform(45.0, 110.0))); u += rng.uniform(380.0, 950.0)
        n = int(D / 2.0) + 1; pts = []
        for i in range(n + 1):
            s = i * D / n; u = s / D; off = sum(am * math.sin((k + 1) * math.pi * u + ph) * (1.0 - 0.0) for k, (am, ph) in enumerate(zip(amps, phs))) * math.sin(math.pi * u) ** 0.5
            for (pos, lat) in jogs: off += lat * (_smooth((s - pos) / 75.0) - _smooth((s - pos - 190.0) / 75.0))
            pts.append((s * ux + off * nx, s * uy + off * ny))
        cum = [0.0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]): cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.length = cum[-1]; m = int(self.length / ds) + 1; self.n = m; X = []; Y = []; j = 0
        for k in range(m):
            s = min(k * ds, self.length)
            while j < len(cum) - 2 and cum[j + 1] < s: j += 1
            f = (s - cum[j]) / max(cum[j + 1] - cum[j], 1e-9); X.append(pts[j][0] + f * (pts[j + 1][0] - pts[j][0])); Y.append(pts[j][1] + f * (pts[j + 1][1] - pts[j][1]))
        self.x, self.y = X, Y; head = []
        for k in range(m):
            k0, k1 = max(0, k - 2), min(m - 1, k + 2); head.append(math.atan2(X[k1] - X[k0], Y[k1] - Y[k0]))                       # compass heading: 0 = north, clockwise
        for _ in range(2):                                                                         # light smoothing so the curvature is not noise
            head = [(head[max(0, k - 1)] + head[k] + head[min(m - 1, k + 1)]) / 3.0 for k in range(m)]
        self.head = head; kap = []
        for k in range(m):
            k0, k1 = max(0, k - 3), min(m - 1, k + 3); d = head[k1] - head[k0]
            d = (d + math.pi) % (2 * math.pi) - math.pi; kap.append(d / max((k1 - k0) * ds, ds))                                   # rad per metre, + = turning right
        self.kappa = kap
        self.limit = [(city_kmh if (k * ds < city_share * self.length or k * ds > (1 - city_share) * self.length) else motorway_kmh) for k in range(m)]
        self.motorway = [lim > 80 for lim in self.limit]
        self.lights = []; s = rng.uniform(250.0, 700.0)                                              # traffic lights in the city legs
        while s < self.length - 200.0:
            if self.limit[min(m - 1, int(s / ds))] < 80: self.lights.append((s, rng.random() < 0.55)); s += rng.uniform(350.0, 1100.0)
            else: s += 800.0
        for lo in (0.0, 1.0):                                                                         # roadworks: a lower limit for a stretch of the motorway
            pass
        r0 = int(self.length * (0.45 + rng.uniform(-0.05, 0.05)) / ds)
        for k in range(r0, min(m, r0 + int(2500 / ds))):
            if self.limit[k] > 80: self.limit[k] = 80.0

    # ---- lookups (s in metres from the start, `rev` = driving B -> A)
    def _k(self, s, rev): k = int(min(max(s, 0.0), self.length) / self.ds); return (self.n - 1 - k) if rev else k

    def heading(self, s, rev=False):
        h = self.head[self._k(s, rev)]; return math.degrees(h + (math.pi if rev else 0.0)) % 360.0

    def curvature(self, s, rev=False):
        k = self.kappa[self._k(s, rev)]; return -k if rev else k

    def limit_at(self, s, rev=False): return self.limit[self._k(s, rev)]
    def is_motorway(self, s, rev=False): return self.motorway[self._k(s, rev)]

    def position(self, s, rev=False):
        k = self._k(s, rev); x, y = self.x[k], self.y[k]
        return self.a[0] + math.degrees(y / R_EARTH), self.a[1] + math.degrees(x / (R_EARTH * self.cos))

    def stops_ahead(self, s, rev, horizon):
        """(distance, red?) of traffic lights within `horizon` metres ahead."""
        out = []
        for (p, red) in self.lights:
            q = (self.length - p) if rev else p
            if 0.0 <= q - s <= horizon: out.append((q - s, red, p))
        return sorted(out)

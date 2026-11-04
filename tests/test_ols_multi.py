"""qa_core.ols_multi as array products must give what the loops gave (the original is kept here as the reference), and vgpt/schema.table_names must follow the schema."""
import os, random, sqlite3, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import qa_core as Q, tablenames as schema


def reference(X, y, ridge=1e-6):
    n, p = len(X), len(X[0]) if X else 0
    if n < p + 3: return None
    Z = [[1.0] + list(r) for r in X]; k = p + 1
    A_ = [[sum(Z[i][a] * Z[i][b] for i in range(n)) + (ridge if a == b and a > 0 else 0) for b in range(k)] for a in range(k)]; b_ = [sum(Z[i][a] * y[i] for i in range(n)) for a in range(k)]
    beta = Q.solve(A_, b_)
    if beta is None: return None
    my = Q.mean(y); ss_tot = sum((v - my) ** 2 for v in y); ss_res = sum((y[i] - sum(beta[a] * Z[i][a] for a in range(k))) ** 2 for i in range(n))
    return beta, (1 - ss_res / ss_tot if ss_tot > 0 else 0.0)


class Ols(unittest.TestCase):
    def test_same_numbers(self):
        rng = random.Random(4)
        for n, p in ((12, 2), (60, 6), (2800, 7), (30, 1)):
            X = [[rng.uniform(0, 30) for _ in range(p)] for _ in range(n)]; y = [150 + 3 * r[0] - 2 * r[-1] + rng.gauss(0, 5) for r in X]
            a, b = Q.ols_multi(X, y), reference(X, y)
            for u, v in zip(a[0], b[0]): self.assertAlmostEqual(u, v, delta=1e-9 * max(1, abs(v)))
            self.assertAlmostEqual(a[1], b[1], places=10)
        self.assertIsNone(Q.ols_multi([[1.0]] * 3, [1, 2, 3]))
        x = [[1.0, 2.0]] * 10; yy = [float(i) for i in range(10)]; a, b = Q.ols_multi(x, yy), reference(x, yy)                  # collinear terms: the ridge term keeps it solvable, the same way as before
        self.assertEqual(a is None, b is None)
        if a: self.assertEqual([round(v, 9) for v in a[0]], [round(v, 9) for v in b[0]])


class Schema(unittest.TestCase):
    def test_follows_ddl(self):
        d = tempfile.mkdtemp(); c = sqlite3.connect(os.path.join(d, "a.db")); c.execute("CREATE TABLE t(x)")
        self.assertEqual(schema.table_names(c), {"t"}); self.assertEqual(schema.table_names(c), {"t"})
        c.execute("CREATE VIEW v AS SELECT x FROM t"); self.assertEqual(schema.table_names(c), {"t", "v"})
        c.execute("DROP VIEW v"); self.assertEqual(schema.table_names(c), {"t"})


if __name__ == "__main__":
    unittest.main()

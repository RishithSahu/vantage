"""Event lists can be retrieved IN FULL by paging: the concatenation of all pages equals the complete list in plain SQL (same events, same order); the contract states which events are shown and how to continue."""
import os, sqlite3, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import nl2ir
from vgpt.brain import Brain
from vgpt.query_ir import EVENT_PAGE, Query

NOW = 1790406000; DBF = os.path.join(ROOT, "data", "gen_courier_ext.db")


@unittest.skipUnless(os.path.exists(DBF), f"fixture database missing: {DBF}")
class Paging(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.qc = Brain(DBF, slm_ir=None, now=NOW).qc(); cls.c = sqlite3.connect(f"file:{DBF}?mode=ro", uri=True)

    def test_all_pages_together_are_the_complete_list_in_order(self):
        lo = NOW - 30 * 86400; ref = [r[0] for r in self.c.execute("SELECT ts FROM events WHERE type = 'HARSH_BRAKE' AND ts >= ? AND ts < ? ORDER BY ts DESC", (lo, NOW + 1))]
        self.assertGreater(len(ref), 2 * EVENT_PAGE); got = []; k = 1
        while True:
            r = self.qc.execute(Query.from_text(f"op=list_events t=last:30:day ev=HARSH_BRAKE evs=page{k}")); self.assertEqual(r.status, "answered")
            got += [i["ts"] for i in r.result.items]; pg = r.result.extra["page"]; self.assertEqual(pg["total"], len(ref)); self.assertEqual(pg["first"], len(got) - len(r.result.items) + 1); self.assertEqual(pg["last"], len(got))
            if pg["last"] >= pg["total"]: break
            self.assertRegex(r.answer.text, r"Say 'next' to see events"); k += 1
        self.assertEqual(got, ref)

    def test_the_first_page_states_that_it_is_a_page_and_how_to_continue(self):
        r = self.qc.execute(Query.from_text("op=list_events t=last:30:day ev=HARSH_BRAKE evs=all")); ct = r.result.extra["contract"]
        self.assertEqual(ct["completeness"], "truncated"); self.assertEqual(ct["page"]["last"], EVENT_PAGE); self.assertIn("Say 'next'", r.answer.text)

    def test_next_continues_the_previous_list_and_nothing_else(self):
        st = nl2ir.ConvState(); st.last = Query.from_text("op=list_events t=last:30:day ev=HARSH_BRAKE evs=all")
        p = nl2ir.parse("next", now=NOW, tz=3600, state=st); self.assertEqual(p.query.to_text(), "op=list_events t=last:30:day ev=HARSH_BRAKE evs=page2")
        st.last = p.query; p = nl2ir.parse("show more", now=NOW, tz=3600, state=st); self.assertEqual(p.query.to_text(), "op=list_events t=last:30:day ev=HARSH_BRAKE evs=page3")

    def test_the_last_event_is_looked_for_in_the_whole_history_and_says_so(self):
        p = nl2ir.parse("when was the last harsh braking event", now=NOW, tz=3600); self.assertEqual(p.query.scope.time.to_text(), "all")
        r = self.qc.execute(p.query); self.assertIn("whole recorded history", r.answer.text); self.assertNotIn("last 7 days", r.answer.text)


if __name__ == "__main__": unittest.main()

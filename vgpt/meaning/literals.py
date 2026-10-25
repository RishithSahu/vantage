"""Literal values in a question: periods, instants, numeric limits, numbers.  The ONLY text-pattern code in the meaning path, and it never decides what is being asked.

"2hrs", "last week", "3 days ago at 10 pm", "above 60 mph", "from 20 to 80 percent" are VALUES the user typed; turning them into typed values (TimeSpec, Condition, floats, an epoch second) is
parsing, in the same sense that reading "42" as the integer 42 is.  What the question MEANS (which quantity, which operation, whether it is a fact about the car or an instruction) is decided
by the learned frame, never here.  Reuses the project's existing normaliser / time / threshold parsers (nl2ir) so a value is read identically everywhere."""
import datetime as dt
import re

from .. import nl2ir
from ..query_ir import TimeSpec

DAY = 86400
UNIT_SEC = {"seconds": 1, "second": 1, "secs": 1, "sec": 1, "minutes": 60, "minute": 60, "mins": 60, "min": 60, "hours": 3600, "hour": 3600, "hrs": 3600, "hr": 3600, "days": DAY, "day": DAY, "weeks": 7 * DAY, "week": 7 * DAY}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
                "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45, "sixty": 60, "a": 1, "an": 1, "half an": 0.5}
TIME_FUZZY = ("yesterday", "tomorrow", "weekend", "evening", "morning", "tonight", "midnight", "fortnight")
TIME_ABBR = {"wek": "week", "weke": "week", "monht": "month", "mnt": "month", "yesturday": "yesterday", "yestrday": "yesterday", "yesteday": "yesterday", "tody": "today", "todya": "today", "lst": "last", "lsat": "last", "pst": "past", "wk": "week", "wks": "weeks", "mnth": "month", "mnths": "months", "yday": "yesterday", "tmrw": "tomorrow", "hr": "hour", "min": "minute", "mins": "minutes", "sec": "second"}
HOUR_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
          "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH_RX = "|".join(sorted(MONTHS, key=len, reverse=True))


class Literals:
    __slots__ = ("times", "thresholds", "numbers", "norm", "codes", "cmp", "position", "code_classes", "pick", "daykinds", "per", "pattern")

    def __init__(self, times, thresholds, numbers, norm):
        self.times, self.thresholds, self.numbers, self.norm = times, thresholds, numbers, norm
        self.codes = []; self.cmp = None; self.position = None; self.code_classes = []; self.pick = None; self.daykinds = []; self.per = None; self.pattern = False


def extract(text, now, tz, profile=None, target=None):
    """-> Literals(times=[TimeSpec in order], thresholds=[Condition], numbers=[float], norm=normalised text)."""
    if now is not None and tz is not None and (int(now) + int(tz)) % 86400 == 0: now = int(now) - 1                                   # a history that ends exactly at midnight ends the day before
    text = re.sub(r"\b(" + "|".join(TIME_ABBR) + r")\b", lambda m: TIME_ABBR[m.group(1)], str(text).lower())
    text = " ".join(_closest(w) for w in text.split())
    rng = span(text, now, tz)
    t = nl2ir.normalise(text); S = nl2ir._S(t)
    from .. import signal_registry as R
    thresholds = nl2ir._conditions(S, profile, target_hint=target if target in R.CONCEPTS else None)       # only a registry signal has a unit to hint with
    times = [rng] if rng is not None else nl2ir._times(S, now, tz)
    cal = calendar_periods(text, now, tz)
    if cal: times = cal                                                                                      # a year / month typed in full wins over the relative-period reading
    if not times and now is not None:                                                                        # "on the 11th": the latest 11th that is not in the future (this month, or last month when the day has not come yet)
        m11 = re.search(r"\b(?:on |for |during )?the (\d{1,2})(?:st|nd|rd|th)\b(?! of)(?!\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))", " " + text + " ")
        if m11 and 1 <= int(m11.group(1)) <= 31:
            today = nl2ir._local(now, tz).date(); d = int(m11.group(1)); y, mo = today.year, today.month
            if d > today.day: y, mo = (y - 1, 12) if mo == 1 else (y, mo - 1)
            try: times = [TimeSpec("date", date=dt.date(y, mo, d).isoformat())]
            except ValueError: pass
    if not times and re.search(r"\b(?:have|has)\s+(?:there|i|we|it|this\s+car|the\s+car)\s+(?:ever\s+)?(?:been|had)\b|\bhave\s+i\s+(?:ever\s+)?\w+ed\b", " " + text + " "): times = [TimeSpec("all")]      # "how many shutdowns have there been": no period typed, perfect tense: the whole history
    if not times and re.search(r"\b(?:recently|lately|of late|these days|nowadays|in recent days)\b", " " + text + " "): times = [TimeSpec("relative", 14, "day")]          # "recently": the last two weeks, when no period is typed
    if not times:                                                                                            # vague spans of a trend question: "over the months", "over the year", "over time", "the last few weeks"
        pad_ = " " + text + " "
        m_few = re.search(r"\b(?:last|past|previous)\s+few\s+(days|weeks|months)\b", pad_)
        if m_few: times = [TimeSpec("relative", {"days": 7, "weeks": 28, "months": 90}[m_few.group(1)], "day")]
        elif re.search(r"\b(?:over|across|through|during)\s+the\s+(?:past\s+|last\s+|recent\s+|coming\s+)?months\b|\bmonth\s+(?:after|by|over)\s+month\b|\bfrom\s+month\s+to\s+month\b|\bevery\s+month\b", pad_): times = [TimeSpec("relative", 90, "day")]
        elif re.search(r"\b(?:over|across|through|during)\s+the\s+(?:whole\s+|full\s+|entire\s+)?(?:year|seasons)\b|\bover\s+the\s+course\s+of\s+the\s+year\b|\byear[- ]on[- ]year\b|\bover\s+the\s+past\s+year\b", pad_): times = [TimeSpec("relative", 365, "day")]
        elif re.search(r"\b(?:over|with)\s+time\b|\bas\s+time\s+(?:goes|passes)\s+(?:on|by)\b|\bwith\s+age\b|\bover\s+its\s+life\b", pad_): times = [TimeSpec("all")]
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]
    lit = Literals(times, thresholds, numbers, t); lit.position = wheel_position(text); lit.pick = trip_pick(text); lit.daykinds, lit.per, lit.pattern = day_grouping(text); lit.codes = [m.upper() for m in re.findall(r"\b[pcbu][0-9a-f]{4}\b", text) if re.search(r"\d", m)]      # a fault code is a typed value
    lit.code_classes = [] if lit.codes else sorted({m.upper() for m in re.findall(r"\b([pcbu])[ -]?(?:class\s+)?(?:dtc|dtcs|codes?|fault codes?)\b", text, re.I)})                  # "u codes", "any p code": the class letter of a fault code is a typed value too
    if re.search(r"\b(?:above|over|more than|greater than|exceed\w*|higher than|faster than|beyond|farther than|further than|longer than|bigger than|larger than|hotter than|warmer than|wider than)\b", text): lit.cmp = "gt"
    elif re.search(r"\b(?:below|under|less than|lower than|fewer than|slower than|beneath|closer than|nearer than|shorter than|smaller than|colder than|cooler than|narrower than)\b", text): lit.cmp = "lt"
    return lit


PICK_RX = (("least_efficient", r"\b(?:least|worst|most wasteful|thirstiest|most consuming)\b.{0,12}\b(?:efficient|economical|consumption)|\b(?:worst|highest)\b.{0,12}\b(?:consumption|economy)"),
           ("most_efficient", r"\b(?:most|best)\b.{0,12}\b(?:efficient|economical)|\b(?:lowest|best)\b.{0,12}\b(?:consumption|economy)"),
           ("longest", r"\b(?:longest|biggest|farthest|furthest|most distant)\b"), ("shortest", r"\b(?:shortest|smallest|briefest)\b"), ("first", r"\b(?:first|earliest|oldest)\b"),
           ("last", r"\b(?:last|latest|most recent|previous|recent)\b"))


def trip_pick(text):
    """Which trip a question singles out ("my longest trip", "the last one", "the most efficient drive") -> longest | shortest | first | last | most_efficient | least_efficient, or None.
    A selector value, like a date: the learned frame says the question is about a trip, this says which one."""
    raw = " " + str(text).lower() + " "
    if not re.search(r"\b(?:trip|trips|drive|drives|journey|journeys|commute|run)\b", raw): return None
    for name, rx in PICK_RX:
        if re.search(rx, raw): return name
    return None


POSITION_RX = (("fl", r"\b(?:front[\s-]*left|left[\s-]*front|fl|near[\s-]*side front|driver(?:'s)? side front)\b"), ("fr", r"\b(?:front[\s-]*right|right[\s-]*front|fr|off[\s-]*side front)\b"),
               ("rl", r"\b(?:rear[\s-]*left|left[\s-]*rear|back[\s-]*left|left[\s-]*back|rl)\b"), ("rr", r"\b(?:rear[\s-]*right|right[\s-]*rear|back[\s-]*right|right[\s-]*back|rr)\b"))


def wheel_position(text):
    """The wheel a question names ("front left tyre", "rr") -> fl | fr | rl | rr, or None.  A position is a value of the question, like a number: it selects one of four sensors of the same quantity."""
    raw = " " + str(text).lower() + " "
    for pos, rx in POSITION_RX:
        if re.search(rx, raw): return pos
    return None


def _closest(w):
    """A misspelt time word ("yesturday", "tomorow") -> the closed time vocabulary.  Only long words, only a close match: this is spelling normalisation of VALUES, not meaning."""
    if len(w) < 6 or w in TIME_FUZZY: return w
    import difflib
    m = difflib.get_close_matches(w, TIME_FUZZY, n=1, cutoff=0.84)
    return m[0] if m else w


def span(text, now, tz):
    """"from 3 days ago to 5 days ago" / "between 5 and 3 days ago" -> TimeSpec(range) over whole calendar days (or hours / minutes), else None."""
    text = re.sub(r"\b(" + "|".join(sorted((w for w in HOUR_WORDS), key=len, reverse=True)) + r")\b", lambda m: str(HOUR_WORDS[m.group(1)]), text)                   # "three and five days ago"
    if re.search(r"\bsince\s+yesterday\b", text): return TimeSpec("between", 1, "day", value2=0)
    wd = re.search(r"\bsince\s+(?:last\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text)
    if wd:
        today = dt.datetime.fromtimestamp(now + tz, dt.timezone.utc).weekday(); far = (today - WEEKDAYS[wd.group(1)]) % 7 or (7 if "last " in wd.group(0) else 0)
        return TimeSpec("between", far, "day", value2=0) if far else TimeSpec("calendar", name="today")
    m = re.search(r"\b(?:from\s+)?(\d+)\s+(days?|hours?|minutes?)\s+ago\s+(?:to|through|until|and)\s+(\d+)\s+(days?|hours?|minutes?)\s+ago\b", text) or re.search(r"\bbetween\s+(\d+)\s+and\s+(\d+)\s+(days?|hours?|minutes?)\s+ago\b", text)
    if not m: return None
    g = m.groups()
    a, ua, b, ub = (int(g[0]), g[1], int(g[2]), g[3]) if len(g) == 4 else (int(g[0]), g[2], int(g[1]), g[2])
    a, b = max(a, b), min(a, b)
    if a == b: return None
    return TimeSpec("between", a, "day" if ua.startswith("day") else "hour" if ua.startswith("hour") else "minute", value2=b)


_CAL_MASK = re.compile(r"\b(?:(?:spring|summer|autumn|fall|winter|january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?,?\s+(?:of\s+)?)?(?:19|20)\d\d\b(?!\s*(?:km|kwh|rpm|kpa|wh|kw|mph|psi|w\b|v\b|m\b))", re.I)


def mask_calendar(text):
    """A calendar year or month typed in full ("in 2024", "in March 2025") is a VALUE the literal parser reads (literals.calendar_periods); the frame model was trained on relative periods and reads a bare year as a stray number, so it is shown a period word instead."""
    return _CAL_MASK.sub("last month", str(text))


def calendar_periods(text, now, tz):
    """Calendar periods typed in a question -> [TimeSpec in the order they appear].  They are VALUES, like a date, resolved to local seconds by the planner (query_planner.resolve_time):
    a year (y2025), a month (m2025_03), a quarter (q2025_2), a half (h2025_1), a season or any run of months (s2025_06_2025_08), "since" a month or year (f2025_03), last / this year, the whole history, "last N years"."""
    raw = " " + str(text).lower() + " "
    if date_of(raw, now, tz) is not None and re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:" + _MONTH_RX + r")\b|\b(?:" + _MONTH_RX + r")\.?\s+\d{1,2}(?:st|nd|rd|th)?\b", raw): return []          # a full date is an instant, not a period
    now_d = dt.datetime.fromtimestamp(now + tz, dt.timezone.utc); found = []; used = []
    M = r"(" + _MONTH_RX + r")"; Y = r"(20\d\d)"
    def take(m, name):
        found.append((m.start(), TimeSpec("calendar", name=name))); used.append((m.start(), m.end()))
    def free(m): return not any(a <= m.start() < b for a, b in used)
    def mnum(w): return MONTHS[w]
    def myear(g, mo):
        if g: return int(g)
        return now_d.year if mo <= now_d.month else now_d.year - 1
    SEASON = {"spring": (3, 5), "summer": (6, 8), "autumn": (9, 11), "fall": (9, 11)}
    for m in re.finditer(r"\b(?:between|from)\s+" + M + r"\.?(?:\s+" + Y + r")?\s+(?:and|to|until|through)\s+" + M + r"\.?(?:\s+" + Y + r")?\b", raw):
        a, ya, b, yb = mnum(m.group(1)), m.group(2), mnum(m.group(3)), m.group(4)
        y2 = myear(yb or ya, b); y1 = int(ya) if ya else (y2 if a <= b else y2 - 1)
        take(m, f"s{y1}_{a:02d}_{y2}_{b:02d}")
    for m in re.finditer(r"\b(?:between|from)\s+" + Y + r"\s+(?:and|to|until|through)\s+" + Y + r"\b", raw):
        if free(m): take(m, f"s{m.group(1)}_01_{m.group(2)}_12")
    for m in re.finditer(r"\b" + r"(spring|summer|autumn|fall|winter)\s+(?:of\s+)?" + Y + r"\b", raw):
        if not free(m): continue
        a, y = m.group(1), int(m.group(2))
        take(m, f"s{y - 1}_12_{y}_02" if a == "winter" else f"s{y}_{SEASON[a][0]:02d}_{y}_{SEASON[a][1]:02d}")
    for m in re.finditer(r"\bq([1-4])\s+(?:of\s+)?" + Y + r"\b|\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?" + Y + r"\b", raw):
        if not free(m): continue
        qn = int(m.group(1)) if m.group(1) else {"first": 1, "second": 2, "third": 3, "fourth": 4}[m.group(3)]; take(m, f"q{m.group(2) or m.group(4)}_{qn}")
    for m in re.finditer(r"\b(first|second)\s+half\s+(?:of\s+)?" + Y + r"\b|\bh([12])\s+" + Y + r"\b", raw):
        if not free(m): continue
        hn = (1 if m.group(1) == "first" else 2) if m.group(1) else int(m.group(3)); take(m, f"h{m.group(2) or m.group(4)}_{hn}")
    for m in re.finditer(r"\bsince\s+" + M + r"\.?(?:,?\s+" + Y + r")?\b", raw):
        if free(m): take(m, f"f{myear(m.group(2), mnum(m.group(1)))}_{mnum(m.group(1)):02d}")
    for m in re.finditer(r"\bsince\s+" + Y + r"\b", raw):
        if free(m): take(m, f"f{m.group(1)}_01")
    for m in re.finditer(r"\b" + M + r"\.?,?\s+(?:of\s+)?" + Y + r"\b", raw):
        if free(m): take(m, f"m{m.group(2)}_{mnum(m.group(1)):02d}")
    for m in re.finditer(r"\b" + Y + r"\b(?!\s*(?:km|kwh|rpm|kpa|w\b|v\b|m\b|wh|kw|mph|psi))", raw):
        if free(m): take(m, f"y{m.group(1)}")
    for m in re.finditer(r"\b(?:in|during|of|for|throughout)\s+" + M + r"\b(?!\.?,?\s*(?:of\s+)?\d)", raw):
        if free(m): mo = mnum(m.group(1)); take(m, f"m{now_d.year if mo <= now_d.month else now_d.year - 1}_{mo:02d}")
    for m in re.finditer(r"\bsame\s+(?:month|period)\s+(?:of\s+|in\s+)?(?:last\s+year|a\s+year\s+ago|the\s+previous\s+year)\b", raw):                                                                  # "this month against the same month last year": the current month and that month a year earlier
        if free(m): take(m, f"m{now_d.year - 1}_{now_d.month:02d}"); found.append((m.start() - 1, TimeSpec("calendar", name=f"m{now_d.year}_{now_d.month:02d}")))
    for m in re.finditer(r"\b(?:last|previous|past)\s+year\b|\bthis\s+year\b|\bso\s+far\s+this\s+year\b|\bthe\s+year\s+before\s+last\b", raw):
        if not free(m): continue
        if "this" in m.group(0): take(m, "this_year")
        elif m.group(0).startswith("the"): take(m, f"y{now_d.year - 2}")
        else: take(m, "last_year")
    for m in re.finditer(r"\b(?:last|past|previous)\s+(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\b", raw):
        n = int(m.group(1)) if m.group(1).isdigit() else {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}[m.group(1)]
        if 1 <= n <= 9: found.append((m.start(), TimeSpec("relative", n * 365, "day")))
    SEASON_MONTHS = {"winter": (12, 2), "spring": (3, 5), "summer": (6, 8), "autumn": (9, 11), "fall": (9, 11)}
    for m in re.finditer(r"\b(last|previous|past|this|current)\s+(winter|spring|summer|autumn|fall(?!\s+(?:below|under|beneath|above|over|short|behind|off|out|back|down|apart|from|to|by|more|less|further|asleep|sharply|quickly|slowly|again)\b))\b", raw):
        if not free(m): continue
        a, b = SEASON_MONTHS[m.group(2)]; best = None
        for y in (now_d.year, now_d.year - 1, now_d.year - 2):                                           # the occurrences of the season, newest first: (start, end) as (year, month)
            start = (y - 1, 12) if a == 12 else (y, a); end = (y, 2) if a == 12 else (y, b)
            if start > (now_d.year, now_d.month): continue                                                 # not begun yet
            done = end < (now_d.year, now_d.month)
            if m.group(1) in ("this", "current") or done: best = (start, end); break                      # "this X": the newest one that has begun; "last X": the newest one that has ended
        if best:
            (y1, m1), (y2, m2) = best; take(m, f"s{y1}_{m1:02d}_{y2}_{m2:02d}")
    words = r"(?:ever|all[- ]time|in total|in all|overall|whole (?:recorded )?history|entire (?:recorded )?history|since the beginning|since the start|since records began|over the years|through the years|across the years|since new|lifetime|so far)"
    strong = re.compile(r"\b(?:ever|all[- ]time|whole (?:recorded )?history|entire (?:recorded )?history|since the beginning|since the start|since records began|over the years|through the years|across the years|since new|lifetime)\b")
    for m in re.finditer(r"\b" + words + r"\b", raw):
        if found and not strong.match(m.group(0)): continue                                                # "in total last year", "so far this month": the period named wins over the weak words
        found.append((m.start(), TimeSpec("all")))
    return [t for _, t in sorted(found, key=lambda x: x[0])]


def day_grouping(text):
    """-> (daykinds, per, pattern): the kinds of day named ("weekend", "weekday", in the order said), the grain of a breakdown asked for ("day" / "week" / "month": "per week", "each month", "month by month", "monthly"; none when an average
    is asked), and whether a pattern in the time of events is asked for ("is there a pattern in when ...", "what time of day ...", "which day of the week ...").  Values read from the words, like a date."""
    raw = " " + str(text).lower() + " "; kinds = []
    for m in re.finditer(r"\b(week[- ]?ends?|week[- ]?days?|working days?|work days?|business days?)\b", raw):
        k = "weekend" if "end" in m.group(1) else "weekday"
        if k not in kinds: kinds.append(k)
    per = None
    m = re.search(r"\b(?:each|every|per|by|for each|for every)\s+(day|week|month)\b|\b(day|week|month)[- ]by[- ](?:day|week|month)\b|\b(daily|weekly|monthly)\b", raw)
    if m: per = m.group(1) or m.group(2) or {"daily": "day", "weekly": "week", "monthly": "month"}[m.group(3)]
    if per in ("day", "week") and re.search(r"\b(?:average|avg|mean|typical|usual)\b", raw): per = None                    # "trips per day on average" is one figure; a month is always a breakdown
    if per is None and re.search(r"\bwhich month\b|\b(?:best|worst|highest|lowest|busiest|quietest) month\b", raw): per = "month"
    pattern = bool(re.search(r"\bpatterns?\b|\btime of day\b|\bwhat (?:time|hour)\b|\bwhich (?:hours?|days? of the week|part of the day)\b|\bat what (?:time|hour)\b|\b(?:usually|mostly|typically|tend to) (?:happen|appear|occur|come on|show up|get triggered)\b|\bwhen (?:do|does|did) (?:the |my |these |those )?[a-z ]{3,30}? (?:usually|normally|typically|mostly|tend to)\b", raw))
    return kinds, per, pattern


def _local_midnight(now, tz): return ((int(now) + int(tz)) // DAY) * DAY - int(tz)


def date_of(text, now, tz):
    """A calendar date the person typed ("17 march 2025", "march 17th", "2025-03-17", "17/03/2025") -> the epoch second of that local midnight, or None.  A date with no year means the most recent such date."""
    raw = " " + str(text).lower() + " "
    y = mo = d = None
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", raw)
    if m: y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y is None:
        m = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", raw)
        if m: d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y is None:
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MONTH_RX + r")\b\.?,?(?:\s+(\d{4})\b)?", raw)
        if m: d, mo, y = int(m.group(1)), MONTHS[m.group(2)], (int(m.group(3)) if m.group(3) else None)
        else:
            m = re.search(r"\b(" + _MONTH_RX + r")\b\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4})\b)?", raw)
            if m: mo, d, y = MONTHS[m.group(1)], int(m.group(2)), (int(m.group(3)) if m.group(3) else None)
    if mo is None or d is None or not (1 <= mo <= 12 and 1 <= d <= 31): return None
    today = dt.datetime.fromtimestamp(now + tz, dt.timezone.utc)
    for year in ([y] if y else [today.year, today.year - 1]):
        try: day = dt.datetime(year, mo, d, tzinfo=dt.timezone.utc)
        except ValueError: return None
        if y or day <= today: return int(day.timestamp()) - int(tz)
    return None


def instant(text, now, tz):
    """-> epoch second the question points at ("2 hours ago", "yesterday at 3 pm", "at 18:30 on tuesday"), or None when no instant is stated."""
    t = nl2ir.normalise(text).replace("clocktime", " ")
    raw = " " + str(text).lower() + " "
    raw = re.sub(r"\b(?:a|an|one)\s+(day|hour|week)\s+and\s+a\s+half\s+(ago|back)\b", lambda m: f"1.5 {m.group(1)}s {m.group(2)}", raw)
    raw = re.sub(r"\b(" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)\s+(ago|back)\b", lambda m: f"{NUMBER_WORDS[m.group(1)]:g} {m.group(2)} {m.group(3)}", raw)
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)\s+(?:ago|back)\b", raw)
    ago = float(m.group(1)) * UNIT_SEC[m.group(2)] if m else None
    clock = re.search(r"(?:\bat\s+|\b)(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\bat\s+(\d{1,2}):(\d{2})\b|\b(\d{1,2}):(\d{2})\b", raw)
    hh = mm = None
    if clock:
        if clock.group(1) is not None:
            hh, mm = int(clock.group(1)), int(clock.group(2) or 0)
            if clock.group(3) == "pm" and hh < 12: hh += 12
            if clock.group(3) == "am" and hh == 12: hh = 0
        else:
            hh, mm = int(clock.group(4) or clock.group(6)), int(clock.group(5) or clock.group(7))
    if hh is None:
        words = "|".join(HOUR_WORDS)
        w = re.search(r"\b(?:(quarter past|half past|quarter to)\s+)?(" + words + r")(?:\s+o'?clock)?\s+(?:in the |this |last |yesterday )?(?:morning|afternoon|evening|night|tonight)\b", raw) or re.search(r"\bat\s+(?:(quarter past|half past|quarter to)\s+)?(" + words + r")(?:\s+o'?clock)?\b", raw)
        if w:
            hh = HOUR_WORDS[w.group(2)]; mm = {"quarter past": 15, "half past": 30}.get(w.group(1), 0)
            if w.group(1) == "quarter to": hh, mm = hh - 1, 45
    if hh is None:
        bare = re.search(r"\bat\s+(\d{1,2})\b(?!\s*(?:%|km|kw|kwh|nm|v\b|psi|kpa|degrees|c\b))", raw)
        if bare and re.search(r"\b(morning|afternoon|evening|night|today|yesterday|tonight)\b", raw): hh, mm = int(bare.group(1)), 0
    noon = re.search(r"\b(noon|midnight|lunchtime)\b", raw)
    if noon and hh is None: hh, mm = (0, 0) if noon.group(1) == "midnight" else (12, 0)
    if re.search(r"\b(night|evening)\b", raw) and hh is not None and hh < 12 and not re.search(r"\bam\b", raw): hh += 12
    if ago is not None and hh is None: return int(now - ago)
    if hh is None: return None
    stated = date_of(raw, now, tz)
    if stated is not None: return int(stated + hh * 3600 + mm * 60)                   # "17 March 2025 at 09:13:27": that date, not today
    day0 = _local_midnight(now, tz); offset = None
    if re.search(r"\bday before yesterday\b", raw): offset = 2
    elif re.search(r"\byesterday\b|\blast night\b", raw): offset = 1
    elif re.search(r"\btoday\b|\bthis morning\b|\bthis evening\b|\btonight\b", raw): offset = 0
    elif ago is not None and ago >= DAY: offset = int(round(ago / DAY))
    else:
        wd = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", raw)
        if wd:
            today = dt.datetime.fromtimestamp(now + tz, dt.timezone.utc).weekday(); offset = (today - WEEKDAYS[wd.group(1)]) % 7 or 7
    if offset is None:
        target = day0 + hh * 3600 + mm * 60
        offset = 0 if target <= now else 1
    return int(day0 - offset * DAY + hh * 3600 + mm * 60)


def speed_factor(text):
    """Multiplier that turns a speed the person typed into km/h: 3.6 for metres per second, 1.609 for miles per hour, else 1 (a UNIT is a value attribute, parsed like a number)."""
    raw = " " + str(text).lower() + " "
    if re.search(r"\bm\s*(?:/|per)\s*s(?:ec(?:ond)?s?)?\b|\bmetres? per second\b|\bmeters? per second\b", raw): return 3.6
    if re.search(r"\bmph\b|\bmiles? per hour\b|\bmi\s*/\s*h\b", raw): return 1.609344
    return 1.0


def display_units(text):
    """The display units a person typed ("in psi", "in miles", "fahrenheit", "metric", "imperial") -> {"p": kpa|psi|bar, "d": km|mi, "t": c|f} for the keys that were named.  A unit is a value attribute, like a number."""
    raw = " " + str(text).lower() + " "; u = {}
    if re.search(r"\bimperial\b", raw): return {"p": "psi", "d": "mi", "t": "f"}
    if re.search(r"\bmetric\b", raw): return {"p": "kpa", "d": "km", "t": "c"}
    if re.search(r"\bpsi\b", raw): u["p"] = "psi"
    elif re.search(r"\bin bars?\b|\bbars? (?:unit|pressure)\b", raw): u["p"] = "bar"
    elif re.search(r"\bin (?:kpa|kilopascals?)\b", raw): u["p"] = "kpa"
    if re.search(r"\bin (?:miles?|mi|mph)\b|\bmiles per hour\b|\bmph\b", raw): u["d"] = "mi"
    elif re.search(r"\bin (?:km|kms|kilomet\w+|kph|km/h)\b", raw): u["d"] = "km"
    if re.search(r"\bfahrenheit\b|\bdegrees f\b|\bin f\b", raw): u["t"] = "f"
    elif re.search(r"\bcelsius\b|\bcentigrade\b|\bdegrees c\b|\bin c\b", raw): u["t"] = "c"
    return u


def amp_hours_and_volts(text):
    """"7000 mAh at 3.7 V" -> (amp_hours, volts); either may be None."""
    raw = " " + str(text).lower() + " "
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(m\s*ah|milliamp\w*[- ]hours?|ah|amp[- ]hours?)\b", raw)
    v = re.search(r"(\d+(?:\.\d+)?)\s*(?:v|volts?)\b", raw)
    ah = None
    if m: ah = float(m.group(1).replace(",", "")) / (1000.0 if m.group(2).startswith(("m", "milli")) else 1.0)
    return ah, (float(v.group(1)) if v else None)


# ------------------------------------------------------------------------------------------------ discourse values (used by discourse.py, which itself holds no text pattern)
_TOK = re.compile(r"[a-z0-9]+(?:[/'.-][a-z0-9]+)*")
_DNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fourteen": 14}
_DN = r"(\d+|" + "|".join(sorted(_DNUM, key=len, reverse=True)) + r")"
_DU = r"(hour|day|week|month)s?"
_BEFORE = re.compile(r"\b(?:the\s+)?(?:" + _DN + r"\s+)?" + _DU + r"\s+(?:before|prior to|earlier than|preceding)\b")
_PRIOR = re.compile(r"\b(?:the\s+)?(?:previous|preceding|prior|one before)\s+(?:" + _DN + r"\s+)?" + _DU + r"\b")
_AFTER = re.compile(r"\b(?:the\s+)?(?:" + _DN + r"\s+)?" + _DU + r"\s+(?:after|following|later than)\b|\b(?:the\s+)?next\s+(?:" + _DN + r"\s+)?" + _DU + r"\b")
_AGO = re.compile(r"\b" + _DN + r"\s+(day|week)s?\s+(?:ago|back)\b")
_MAG = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:g\b|km/h|%|c\b|kw\b|v\b|kpa\b|l/min)")


def norm_words(text):
    """Lower-case words only (punctuation dropped): the form discourse cues are compared in."""
    return " ".join(re.sub(r"[^a-z0-9%/ ]+", " ", str(text).lower().replace("\u2019", "'")).split())


def tokens(text):
    """Word tokens of a text, for comparing word spans with glossary forms."""
    return _TOK.findall(str(text).lower().replace("\u2019", "'"))


def _count(s): return int(s) if str(s).isdigit() else _DNUM.get(str(s), 1)


def relative_period(low):
    """"the week before that" / "the previous 3 days" / "the day after" / "two days ago" -> (kind, n, unit) with kind in before | after | ago, or None.  A period named relative to another one."""
    m = _AGO.search(low)
    if m: return ("ago", _count(m.group(1)), m.group(2))
    for rx, kind in ((_BEFORE, "before"), (_PRIOR, "before"), (_AFTER, "after")):
        m = rx.search(low)
        if m:
            g = [x for x in m.groups() if x is not None]
            if len(g) >= 4: g = g[-2:]                                               # the 'next' alternative of _AFTER carries its own two groups
            return (kind, _count(g[0]) if len(g) == 2 else 1, g[-1])
    return None


def ordinal_word(low, words):
    """The first (longest) ordinal word of `words` in the text that is not part of a period ("last week", "first 3 days")."""
    for w in sorted(words, key=len, reverse=True):
        if re.search(r"\b" + re.escape(w) + r"\b(?!\s+(?:week|weeks|day|days|month|months|hour|hours|night|year|years|\d))", low): return w
    return None


def magnitude(detail):
    """The first measured magnitude in an event's recorded detail ("0.35 g at 96 km/h" -> 0.35), or None."""
    m = _MAG.search(detail or "")
    return float(m.group(1)) if m else None


# ------------------------------------------------------------------------------------------------ the shape of a long request (used by vgpt/investigate)
_UNITS = r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?)"
_NUMR = r"(\d+(?:\.\d+)?|" + "|".join(sorted((w for w in NUMBER_WORDS if w not in ("a", "an", "half an")), key=len, reverse=True)) + r"|a|an)"
_BEFORE_RX = (re.compile(r"\b(?:preceding|prior)\s+(?:the\s+)?" + _NUMR + r"\s+" + _UNITS + r"\b(?!\s+(?:ago|after))"), re.compile(r"\b" + _NUMR + r"\s+" + _UNITS + r"\s+(?:before|prior to|leading up to|preceding)\b"))
_AFTER_RX = (re.compile(r"\b(?:following|next|subsequent)\s+(?:the\s+)?" + _NUMR + r"\s+" + _UNITS + r"\b"), re.compile(r"\b" + _NUMR + r"\s+" + _UNITS + r"\s+(?:after|following|later)\b"))
_BOTH_RX = re.compile(r"\b" + _NUMR + r"\s+" + _UNITS + r"\s+before and after\b|\bbefore and after\b.{0,30}?" + _NUMR + r"\s+" + _UNITS)
_WITHIN_RX = re.compile(r"\bwithin\s+(?:about\s+)?" + _NUMR + r"\s+" + _UNITS + r"(?:\s+of\s+(?:one another|each other|the warning|a warning|the fault))?")
_COUNT_RX = re.compile(r"\b(?:the\s+)?(?:(?:last|latest|most recent|top|first|past)\s+)?" + _NUMR + r"\s+(?:(?:most\s+)?(?:recent|unusual|efficient|inefficient|significant|severe|frequent)\s+|least\s+efficient\s+|most\s+unusual\s+)?(?:[a-z\-]+\s+)?(trips?|occurrences?|incidents?|events?|faults?|warnings?|disengagements?|instances?|stops?|anomalies|journeys?)\b")
_SENT_RX = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'(])")


def _secs(num, unit):
    n = NUMBER_WORDS.get(num.lower(), None) if not re.fullmatch(r"\d+(?:\.\d+)?", num) else float(num)
    if n is None: return None
    u = unit.lower(); k = 1 if u.startswith("sec") else 60 if u.startswith("min") else 3600 if u.startswith(("hour", "hr")) else 86400 if u.startswith("day") else 7 * 86400 if u.startswith("week") else 30 * 86400
    return float(n) * k


def sentences(text):
    """Sentence split for a pasted multi-part request (a full stop followed by a capital; decimals and 'e.g.' do not split)."""
    return [s.strip() for s in _SENT_RX.split(" ".join(str(text).split())) if s.strip()]


def window_phrases(text):
    """-> {"before": s, "after": s, "within": s}: the windows a request states ("the preceding ten minutes and following five minutes", "within two minutes of one another", "the 60 seconds before each")."""
    raw = str(text).lower(); out = {}
    for rx in _BEFORE_RX:
        m = rx.search(raw)
        if m: out["before"] = _secs(m.group(1), m.group(2)); break
    for rx in _AFTER_RX:
        m = rx.search(raw)
        if m: out["after"] = _secs(m.group(1), m.group(2)); break
    m = _BOTH_RX.search(raw)
    if m:
        g = [x for x in m.groups() if x]; out["before"] = out["after"] = _secs(g[0], g[1])
    m = _WITHIN_RX.search(raw)
    if m: out["within"] = _secs(m.group(1), m.group(2))
    return {k: v for k, v in out.items() if v}


def count_phrases(text):
    """-> [(n, noun)] for "the three trips", "five most unusual incidents", "the last five occurrences"."""
    out = []
    for m in _COUNT_RX.finditer(str(text).lower()):
        n = NUMBER_WORDS.get(m.group(1)) if not m.group(1).isdigit() else int(m.group(1))
        if n and n >= 2: out.append((int(n), m.group(2).rstrip("s")))
    return out


# ---------------------------------------------------------------------------------------------------- words to search the log text for (a VALUE parsed from the question, like a number: the frame says "search the logs", the words say for what)
LOG_STOP = frozenset("""a an the and or of to in on at for from by with about any all some there were was is are be been being did do does done have has had show me my list give find search look looking check
what which who when where why how many much more most least than then that this these those it its i we you your our they them their as if
log logs message messages line lines entry entries service services system syslog qnx record recorded recording report reported reporting write wrote written written
yesterday today tonight tomorrow week weeks month months day days hour hours minute minutes second seconds ago last past latest recent recently previous earlier before after during since until
morning evening night noon midnight weekend this-week first second third once twice time times ever never again still just only also too very so such
car vehicle drive driving trip trips please tell can could would should will may might
thing things logged logging oldest newest earliest over mention mentions mentioned mentioning say says said saying contain contains containing include includes including related regarding concerning within between across around into onto upon while
got get gets getting go goes went gone come came coming make made makes making happen happened happening occur occurred occurring appear appeared appearing see saw seen look looked
every each both either neither none nothing something anything everything anyone someone""".split())
LOG_ALIAS = {"mgr": "manager", "ctrl": "controller", "gw": "gateway", "diag": "diagnostic", "pwr": "power", "bms": "battery", "ivi": "infotainment", "hmi": "display"}


def log_terms(text):
    """-> (terms, words): the content words to look for in log lines (alias-expanded, stop words, time words and numbers dropped; at most six) and all words of the question (to recognise a named log source)."""
    words = [LOG_ALIAS.get(w, w) for w in tokens(text)]
    terms = []
    for w in words:
        if w in LOG_STOP or len(w) < 3 or w[0].isdigit(): continue
        if w not in terms: terms.append(w)
    return terms[:6], words


def log_source_hits(sources, words):
    """Named log sources the question mentions: a source is named when a distinctive part of its name (not 'qnx') is one of the question's words."""
    out = []
    for s in sources:
        parts = [LOG_ALIAS.get(p, p) for p in tokens(str(s).replace("_", " ").replace("-", " ")) if p not in ("qnx", "svc", "service", "can", "bus")]
        if parts and all(p in words or p.rstrip("s") in words or p + "s" in words for p in parts): out.append(s)
    return out


def asks_what_data_is_held(text):
    """A question about the store itself ("how many readings are recorded in the database", "how much data do you have", "what data is stored") - the value asked about is the holdings, not a signal."""
    t = " " + str(text).lower() + " "
    holdings = r"(?:readings?|records?|rows?|samples?|data\s*points?|measurements?|entries|data)"
    return bool(re.search(r"\bhow\s+(?:many|much)\s+" + holdings + r"\b.{0,40}\b(?:database|stored|store|recorded|on\s+record|have|hold|holds|there|saved|in\s+total)\b", t)
                or re.search(r"\bwhat\s+(?:data|readings|records)\b.{0,30}\b(?:stored|recorded|available|do\s+you\s+(?:have|hold)|is\s+there)\b", t)
                or re.search(r"\b(?:size|span|coverage)\s+of\s+(?:the\s+)?(?:database|data|history)\b", t))

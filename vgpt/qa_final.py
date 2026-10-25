"""The assembled question-answering engine: the handler classes plus the ordered routing table (first match wins)."""
import re
from . import analytics as A, qa_core as Q
from .qa import QABase, SAFETY_RX, SECRET_RX, LOGSAYS_RX, prep
from .qa_a import MixA, UNIT
from .qa_b import MixB
from .qa_c import MixC
from .qa_d import MixD
from .qa_e import MixE
from .qa_f import MixF
from .qa_g import MixG
from .qa_h import MixH
from .qa_i import MixI
from .qa_core import mean


class QA(QABase, MixA, MixB, MixC, MixD, MixE, MixF, MixG, MixH, MixI):
    def h_missing_soft(self, s, raw): return self.h_missing(s, raw)

    def h_brakes_info(self, s, raw):
        r = self.c.execute("SELECT pad FROM vday ORDER BY ts DESC LIMIT 1").fetchone() if self._has("vday") else None
        n = self.c.execute("SELECT COUNT(*) FROM events WHERE type='HARSH_BRAKE'").fetchone()[0]
        return ("I don't have a parts list, so I can't count the brakes. What the data shows: " + (f"brake pads at about {r[0]:.0f}%, " if r else "") + f"{n} hard-braking events recorded. Like most electric cars this one slows mainly with the motor (regeneration) and has friction brakes at the wheels for hard stops - typical, not read from this car.", "SERVICE")

    def h_bestcombo(self, s, raw):
        ab = Q.last_abnormal(self.c, self.now)
        if not ab: return "No abnormal reading is on record to explain."
        e = (ab["ts"], "ABNORMAL_READING", "warn", f"{ab['label']} {ab['value']:.1f} {ab['unit']} (normal {ab['normal']})")
        d = self.diag_event(e)
        if not d: return "There is not enough data around that reading."
        self.ctx.update(diag=d, topic="event", event_row=e)
        out = [f"Last detected anomaly: {self.fts(ab['ts'])}, {ab['label']} {ab['value']:.1f} {ab['unit']} against a normal {ab['normal']}.", "Signals around it (the hour before), as deviation from the car's own normal:"]
        b = d["base"]; pre = d["pre"]
        for m, lab, bs in (("flow", "coolant flow", b["flow"]), ("batt", "pack temperature", b["batt"]), ("power", "battery power", b["power"]), ("amb", "outside temperature", b["amb"])):
            if bs: out.append(f"  - {lab}: {pre[m]['mean']:.1f} {UNIT[m]} on average vs normal {bs['mean']:.1f} (sd {bs['sd']:.1f}) -> z = {(pre[m]['mean'] - bs['mean']) / bs['sd']:+.1f}")
        out.append("Best explanation from the combination (ranked):")
        for k, a in enumerate(d["alts"][:3], 1): out.append(f"  {k}. {a['name']} - {a['strength']}: {a['support']}.")
        out.append(f"Confidence: {d['confidence']} - {d['conf_why']}.")
        return "\n".join(out)

    def h_lag(self, s, raw):
        """Which came first around an event: the rise in load or the rise in pack temperature? Engine load is not recorded (EV) - the absolute battery power is used as the load."""
        e, lab, note, ab = self.anchor("most serious event"); ts = e[0] if e else (ab["ts"] if ab else None)
        if ts is None: return note or "No event to analyse."
        sg = Q.event_sig(self.c, ts, 2400, 300)
        if len(sg) < 100: return "There is not enough 1 Hz data around that event."
        res = sg.res; k = max(1, int(30 / res)); pw = [abs(v) for v in sg.cols["power"]]; pw = [mean(pw[max(0, i - k):i + k + 1]) for i in range(len(pw))]; bt = sg.cols["batt"]; n0 = max(5, int(300 / res))
        base_l, peak_l = mean(pw[:n0]), max(pw); base_t = mean(bt[:n0])
        who = self.evrow(e) if e else lab
        if peak_l - base_l < 3.0: return f"Engine load isn't recorded (this is an EV); I used battery power as the load. Around {who} the power never rose meaningfully (typical swing under 3 kW) in the 40 minutes before and 5 after, so there is no load increase to compare with the temperature."
        thr = base_l + 0.5 * (peak_l - base_l); i_l = next((i for i, v in enumerate(pw) if v > thr), None)
        i_t = next((i for i in range(len(bt) - 3) if min(bt[i:i + 3]) > base_t + 0.3), None)
        if i_l is None or i_t is None: return f"Engine load isn't recorded (this is an EV); I used battery power as the load. Around {who} I could not find both a clear load increase and a clear temperature rise in the window."
        dts = sg.ts[i_t] - sg.ts[i_l]
        if i_t <= 2: verdict = f"The pack temperature was already rising at the start of the window, so it began rising BEFORE the load increase at {self.fts(sg.ts[i_l])} (load went above {thr:.0f} kW)."
        elif dts > 30: verdict = f"The pack temperature rose AFTER the load: load went above {thr:.0f} kW at {self.fts(sg.ts[i_l])}, the temperature began to climb (0.3 C above its {base_t:.1f} C starting level) at {self.fts(sg.ts[i_t])}, {Q.dur(dts)} later."
        elif dts < -30: verdict = f"The pack temperature rose BEFORE the load increase: it began to climb at {self.fts(sg.ts[i_t])}, {Q.dur(-dts)} before the load went above {thr:.0f} kW ({self.fts(sg.ts[i_l])}) - so load did not start the heating."
        else: verdict = f"Both happened together (within 30 s): load above {thr:.0f} kW at {self.fts(sg.ts[i_l])}, temperature starting to climb at {self.fts(sg.ts[i_t])}."
        return f"Engine load isn't recorded (this is an EV); I used battery power (absolute value, charging counts as load) as the load. Around {who} (40 minutes before to 5 after, {sg.tier}): " + verdict + " Timing alone shows order, not cause."

    def h_cam_event(self, s, raw):
        ab = Q.last_abnormal(self.c, self.now); ts = ab["ts"] if ab else None
        cams = self.c.execute("SELECT ts, type, detail FROM events WHERE type IN ('CAMERA_BLOCKED','CAMERA_DEGRADED','CAMERA_RESTORED') ORDER BY ts").fetchall()
        if not cams: return "No camera event has been recorded for this vehicle."
        if ts is None: return "No anomaly is on record to associate a camera event with."
        near = min(cams, key=lambda c: abs(c[0] - ts)); dt_ = near[0] - ts
        return (f"The last detected anomaly was {ab['label']} {ab['value']:.1f} {ab['unit']} at {self.fts(ts)}. The nearest camera event is {near[1]} at {self.fts(near[0])} ({Q.dur(abs(dt_))} {'after' if dt_ > 0 else 'before'}): {Q.safe_text(near[2])} - that is "
                f"{'close enough to be related' if abs(dt_) < 600 else 'too far away to be linked to it'}. It is the front camera - the only camera in the data; I keep a 64x48 thumbnail at camera faults, not the footage.")


EVKIND = (r"overheat\w*|over.?temp\w*|diagnostic|\bdtc\b|fault codes?|trouble codes?|adas|disengage\w*|brak\w*.{0,12}event|brake.related|\bwarnings?\b|\bfaults?\b|camera event|harsh|hard.(brak|accel|stop)\w*|\bbraked\b|charging (session|event)|tyre (warning|event)")
ROUTES = [(n, re.compile(rx)) for n, rx in [
    ("safety", SAFETY_RX.pattern),
    ("poison", r"is the battery healthy|diagnostic message claims|claims that the vehicle is operating normally|500 km/h|credible|contradicts \d+% of the other|contradicts 99"),
    ("inject", SECRET_RX.pattern + "|" + LOGSAYS_RX.pattern),
    ("soh_rate", r"(replace|replacing|replacement).{0,25}battery|battery.{0,25}(replace|replacing|replacement|wearing out|wear out|worn out|needs? replac)|how (quickly|fast) is (the|my) battery (wearing|ageing|aging|degrading|degradation|dying)|how long will (the|my) battery (last|live)|battery (life|lifespan|lifetime)"),
    ("explain_simple", r"(explain|describe|tell me about|what does|what is).{0,25}(last|latest|recent|that|the) (warning|alert|fault|error|light).{0,30}(simply|plain|easy|short|briefly|mean)|(simply|plain english|in simple terms|eli5).{0,30}(warning|fault|alert)|what does (the|that|this) (warning|alert|light|fault) mean"),
    ("pump_ok", r"(is|are|how).{0,15}(coolant|cooling|pump).{0,25}(ok|okay|fine|working|good|healthy|status|doing)|cooling (system )?(status|health)|coolant (pump|flow) (status|ok)|(any|is there|are there).{0,12}(problem|issue|fault|trouble|wrong).{0,15}(coolant|cooling|pump)"),
    ("v12", r"\b12 ?v(olt)?\b.{0,20}(battery|health|ok|good|status|fine)|auxiliary battery|\b12 ?v battery"),
    ("battery_now", r"^(how.?s|how is|how are|what.?s|what is|show me|tell me about)( the| my)?( car.?s| vehicle.?s)? battery( doing| status| condition| like| level| state)?[?. ]*$|^battery( status| condition| state)[?. ]*$|how (healthy|full|charged).{0,15}(is )?(the |my )?battery|(battery|pack).{0,12}(health|level|status).{0,12}(and|&).{0,12}(charge|level|health|full)|how (healthy|full).{0,12}battery"),
    ("best_day", r"(which|what) (day|week).{0,25}(most|least|best|worst|more|less|lowest|highest).{0,15}(efficient|economical|energy|wasteful)|(most|least|best|worst) (efficient|economical|wasteful) (day|week)|greenest|most (eco|economical|thrifty) (day|trip)"),
    ("eff_day", r"(how )?(efficient|economical|thrifty).{0,25}(yesterday|today|last week|this week|last night|this morning)|efficiency (yesterday|today)|how did i do (yesterday|today|this week|last week)|consumption (yesterday|today)|how much energy did i use (yesterday|today)|wh ?(per|/) ?(km|mile).{0,25}(yesterday|today|last week|this week)"),
    ("last_charge", r"(when|how long ago).{0,20}(did i |was it |have i |was the )?(last )?(charge|charged|plug)|last (charge|charging)( session)?\b|when was the last charge|did i charge (yesterday|today|last night)|how long since (i|the car|it) (last )?(charged|was charged|plugged)|how long ago (was|did).{0,12}charg"),
    ("battery_equiv", r"\b\d[\d,]*(?:\.\d+)?\s*(?:milliamp(?:ere)?[- ]?hours?|mah)\b.{0,80}\b(?:battery|batteries|charge|charges|charging)\b|\b(?:battery|batteries|charge|charges|charging)\b.{0,80}\b\d[\d,]*(?:\.\d+)?\s*(?:milliamp(?:ere)?[- ]?hours?|mah)\b"),
    ("charge_cost", r"(cost|price|paid|pay|bill|expensive|cheap|money|spend on|spent on).{0,30}(charg|electric|energy|kwh|driv\w+ (cost|per)|per (km|mile))|(charg|electric).{0,20}(cost|spend|spent|bill)|how much (did|does|do|would) (it|i|charging).{0,15}cost|cost per (km|mile)"),
    ("charge_time", r"how long.{0,40}(charge|charging|fill|full|to \d+ ?%)|time to (charge|full)|(charge|charging).{0,25}(to|until) \d+ ?%|how (fast|quickly).{0,15}charge|when will.{0,15}(be )?(full|charged)|(how many )?(minutes|hours|mins?)\s+(to|until|till)\s+(charge|full|\d+)|how long.{0,12}(until|till|to) (it.?s )?(full|charged|\d+ ?%?)|when (will|would) (it|the battery|the car|the pack) (be|reach|hit|get to|come to)( at| to)? ?\d+|(if|when) i plug in.{0,25}(when|how long)"),
    ("speeding", r"^(?!.*(heat|cause|temperature|effect|accounted|regress))(?:(driving|drive|going|am i|do i drive).{0,12}too (fast|quick)|am i speeding|do i speed|over the speed limit|exceed.{0,10}speed limit|(over|above|exceed|break|beat)\w* (the )?(speed )?limit|speed limit|\btoo (fast|quick)\b|how (much|often|many times).{0,25}(above|over|faster than|exceed\w*|beyond) \d{2,3}|(percentage|share|%) of.{0,20}(above|over) \d{2,3})"),
    ("improve_range", r"\b(improve|increase|extend|boost|maximi[sz]e|get more|stretch)\b.{0,20}(range|efficiency|mileage|miles per|km per)|(range|efficiency).{0,15}\btips\b|\beco.?driv|how (can|do) i (drive|use) (less|more efficient)|drive more efficiently|\bsave (energy|battery)|(squeeze|stretch|get|gain|wring) (more )?(kilomet\w+|km|miles|range|distance) (out of|from) (the |my )?(battery|charge)|eco driving (advice|tips)|give me eco"),
    ("hot_battery", r"^(was|has|did|is) (the |my )?(battery|pack)( ever)?( get| got| been)? (too )?(hot|overheat\w*)\b(?!.*(why|caus|because|else|when))|^(has|did|was) (the |my )?(battery|pack) ever.{0,15}(hot|overheat)|^how hot (did|does|has|was) (the |my )?(battery|pack)"),
    ("unusual", r"^(?!.*(pack|battery|charg|coolant|tyre|tire|brake|camera|motor|before|after|range|speed|temperature|feels|sluggish|whole|history|\.)).{0,70}(anything|something|what).{0,20}(unusual|strange|odd|weird|abnormal|wrong).{0,25}(today|yesterday|this morning|tonight|last night|this week|on my drive)|^did anything (happen|go wrong)( today| yesterday| this week)?[?. ]*$"),
    ("wrong", r"strongest evidence that your entire analysis is wrong|evidence that your (entire )?analysis is wrong"),
    ("nuclear3", r"re-?run the investigation|before each incident occurred|remove all future information"),
    ("nuclear1", r"independent engineering investigation"),
    ("c1", r"reconstruct the 2 minutes before"), ("c2", r"normali[sz]e for trip duration"), ("c3", r"most severe anomaly in the entire"), ("c4", r"primarily associated with temperature, speed"),
    ("c5", r"followed by a diagnostic warning within (five|5) minutes"),
    ("precision", r"exactly \d+ minutes?|to the nearest second|exact (driving )?range|exact can message|next ten sensor|next \d+ sensor|exact (mechanical )?component that will fail|\d{1,2}:\d{2}(:\d{2})? tomorrow|speed.{0,40}tomorrow"),
    ("onto", r"unify all references|dependency graph|hierarchy of all|every signal that (directly|indirectly)|same physical incident|distinct failure episodes|group the diagnostic events|same underlying failure|definitely the same measurement|evidence proves that these two"),
    ("trace", r"data source you consulted|timestamps used in your conclusion|records were discarded|assumptions did you make|what part (of your answer )?is (directly )?(observed|calculated|inferred|speculat)|evidence would falsify|strongest argument that your answer is wrong"),
    ("predict_before", r"using only (information|data|evidence) available before|predict the fault without|predict whether the warning will occur"),
    ("avail_at", r"available to a real.?time|real.?time system"),
    ("leak_feat", r"future leakage|constitute future"),
    ("conf_change", r"confidence change|future observations are removed"),
    ("used_records", r"accidentally used future|which records were used|used future evidence"),
    ("cutoff", r"using only data up to|only the previous \w+ minutes|up to \d+ seconds before the event|do it again using only|recalculate your diagnosis"),
    ("sensor_faulty", r"sensor itself is faulty|battery actually overheated|temperature sensor.{0,30}faulty"),
    ("inconsistent", r"physically inconsistent|inconsistent with the rest|appear inconsistent|inconsistent records"),
    ("drift", r"sensor drift|calibration reference|most likely biased|quantify the bias"),
    ("glitch", r"sensor glitch|true event"), ("reconstruct", r"reconstruct the missing|missing telemetry|value of a missing signal"),
    ("corrupt5", r"5% of the sensor|were corrupted"), ("same_phen", r"measuring the same underlying|same underlying phenomenon"), ("sampling", r"sampling rate|sufficient to capture"),
    ("proto_decode", r"decode this can|decode 0x|physical value of the signal"),
    ("proto_map", r"which can message(?!.{0,30}(behind|caus|responsible))|corresponds to the vehicle|byte order|scaling factor|which ecu|can, can.?fd|some/?ip or another|dbc definition|raw can id|physical subsystem|different can messages|diagnostic session"),
    ("root_cause", r"root cause|early symptom"), ("direction", r"motor.?current|caused the .{0,40} or whether"), ("effect_ctrl", r"causal effect|controlling for ambient|holding .{0,25}constant"),
    ("confounder", r"confound"), ("attr3", r"driver, the environment|caused by the driver|distinguish whether the anomaly"), ("intervention", r"what intervention|which intervention|would have prevented"),
    ("causal_graph", r"causal graph|causal diagram"),
    ("cf_speed", r"\d+ ?(km/h|kph|km) (slower|faster)"), ("cf_temp", r"\d+ ?(\u00b0|degrees)? ?c? instead of \d+"), ("cf_soc", r"\d+ ?% soc instead|started at \d+ ?%"),
    ("cf_cooling", r"\d+ ?% more efficient"), ("cf_adas", r"would the adas|adas disengagement still"), ("cf_accel", r"counterfactual.{0,40}temperature|not accelerated aggressively"),
    ("cf_generic", r"flat road|braked \d+ seconds? earlier|next \d+ minutes of telemetry|continued driving|half the acceleration|simulate the same trip|degradation curve|charging history|prove .{0,40}would not have|counterfactual"),
    ("hidden", r"cargo|payload|road gradient|gradient at|tyre pressures? during|how many passengers|passengers were|exact gps location|air conditioner|was the a/?c|headwind|battery chemistry|pack configuration|brake pads?.{0,30}(before|condition)|individual (battery )?cell"),
    ("fleet", r"all vehicles|other vehicles|vehicles of the same model|across the fleet|\bfleet\b|nearest.?neighbou?r|percentile rank|another vehicle|vehicle models|same model|similar failure trajectory|compare vehicles"),
    ("boot", r"bootstrap"),
    ("earliest_change", r"earliest statistically|crossed into abnormal|permanently shifted|change point where"),
    ("soh_rate", r"degradation rate|remaining useful life|linear, exponential|(normal )?(aging|ageing)"), ("periodic", r"recurring temporal|every few days|\bperiodic"), ("worsening", r"progressively worse|same anomaly"),
    ("multitest", r"multiple hypothesis|multiple.?testing|bonferroni|false discovery"), ("top_sig", r"highest statistical significance|most significant anomaly"),
    ("trend_unc", r"uncertainty of (your|the) .{0,30}trend"), ("corr2", r"pearson|spearman|correlation remain"), ("hwy_test", r"hypothesis test|consumes more energy than city|motorway driving consumes more"),
    ("stat_ci", r"confidence interval for|statistically significant\?|state the test|effect size"),
    ("standstill", r"(temperature|heat).{0,40}(increase|rise|rose|climb|go up).{0,40}(standing still|stationary|idle|not moving)"),
    ("eff_trend", r"(am i|are we|do i|is my car).{0,12}(using|use|consuming|burning|drawing).{0,10}(more|less).{0,12}(energy|power|electricity|battery).{0,20}(than )?(usual|before|normal|last week|lately)|is my (consumption|energy use|efficiency) (up|higher|lower|worse|better|getting|changing)|has my (consumption|energy use|efficiency) (gone|changed|got)|has my (energy )?(consumption|energy use|efficiency|usage) (gone|got|become|changed|been)|(consumption|energy use|usage) (gone|going) up|has my (energy )?(consumption|energy use|efficiency|usage) (increased|risen|decreased|dropped|improved|worsened|gone down|gone up|got worse|got better|changed)|is my (consumption|energy use|usage) (increasing|rising|dropping|decreasing)"),
    ("trip_count", r"how many (trips|journeys|drives)( did i| have i| were there)?"),
    ("regen", r"how much.{0,30}regen|regen\w* (recover|got|saved|earned|gave)|energy (recovered|recuperated)|recuperat"),
    ("motor_now", r"how (hot|warm) is (the |my )?(drive )?motor|motor (temp\w*|heat)( now| right now)?[?. ]*$|is (the |my )?(drive )?motor (hot|overheating|ok|okay|fine)|^(how is|how.?s|status of) (the |my |our )?(engine|motor)[?. ]*$"),
    ("signal_why", r"why (is|are|does|do|did|has|have)\b.{0,25}\b(power|current|flow|pump|motor|battery|pack|speed|rpm|soc|state of charge|temperature|coolant|climate|hvac|outside)\b.{0,30}\b(negative|minus|zero|nothing|high|low|rising|climbing|dropping|falling|going up|going down|changing|jump\w*|spik\w*|hot|cold|not (moving|flowing|showing|reading)|below zero)|what does (a )?(negative|minus) .{0,20}(mean|power|kw|current)|(negative|minus) (sign|value|number|power|kw|current)|why does the (battery|pack|motor).{0,20}(heat|warm|get hot|go up)|what makes the (pack|battery|motor|coolant|power|temperature).{0,25}(rise|go up|heat|warm|drop|fall|go down|change)"),
    ("drive_safely", r"(how|tips|advice).{0,25}(drive|driving).{0,20}(safe|safely|safer|carefully)|(safe|safer|careful) driving|(driving|driver) safety|how (can|do) i (be )?(a )?safer driver|safe driving (tips|advice|habits)|(tips|advice).{0,10}\bsafe|behind the wheel|staying safe|stay safe|safe habits|calm driv|(safe|safely) on the road|remember to be safe|avoid (a )?(crash|crashes|accident|accidents|collision|collisions)|prevent (a )?(crash|crashes|accidents?)"),
    ("compare_days", r"compare (today|yesterday|the day before|(mon|tues|wednes|thurs|fri|satur|sun)day).{0,25}(with|to|and|against|vs|versus) (today|yesterday|the day before|(mon|tues|wednes|thurs|fri|satur|sun)day)|(today|yesterday)\s+(vs|versus|compared (to|with)|against)\s+(today|yesterday|the day before)|how (does|did) (today|yesterday) compare"),
    ("overview", r"^(?!.*health)(.*)((summary|summari[sz]e|overview|recap|rundown).{0,20}(week|month|days?|drives?|driving|trips?|car|vehicle)|how was (my|the) (week|month|weekend)|how.?s (my|the) week|^(car|vehicle) (status|summary|overview)[?. ]*$|how is (the|my) (car|vehicle)( doing)?[?. ]*$|how.?s the (car|vehicle)( doing)?[?. ]*$)"),
    ("reach", r"^(?!.*(replac|wear|worn|degrad|ageing|aging|lifespan|battery life))(?:do i need to (charge|plug)|should i (charge|plug)|need a charge|(can|will) i (make it|get home|get there|reach)|(will|can) (it|the car) (make it|reach|get me)|how far can i (go|drive)|how (much|many).{0,10}(battery|charge|range|km|kilomet|miles).{0,10}(left|remaining|do i have)|battery left|(enough|sufficient) (charge|battery|range)|will i get home|(will|can|does|is|would).{0,15}(battery|charge|range).{0,25}(get me|reach|make it|last|enough|take me|do the trip)|how much (further|farther|more).{0,10}(can|could) i (drive|go)|how (many|much) (more )?(km|kilomet\w+|miles).{0,10}(can i|could i|do i have|remain))"),
    ("advice", r"\b(safe|ok|okay|fine|alright|advis\w*|wise|recommend\w*|sensible|risky|dangerous|unsafe|good idea|worth|roadworthy|fit)\b.{0,40}\b(go|going|drive|driving|travel|travelling|traveling|trip|journey|out|road|airport|highway|motorway|use the car|take the car|leave)|(should|can|may|could|would|shall) (i|we)\b.{0,25}\b(go|drive|travel|take|use|keep|continue|head|set off)\b.{0,40}\b(car|vehicle|trip|journey|out|today|now|airport|highway|condition|warning|long|far|work|city|like this)|(am i|are we) (good|ready|okay|ok|fine|safe|clear|set) to (go|drive|leave|travel|set off)|any reason not to (drive|go)|safe for me to go|(go|going) out.{0,15}(safe|or not)|roadworthy|(keep|continue) driving|can i (still|keep) (drive|driving)|should i (be )?(worried|concerned|stop)|is it (dangerous|serious|urgent|a (big )?(problem|deal))|how (bad|serious) is (it|this)|^what should i do[?. ]*$|^what do i do[?. ]*$|what (should|can) i do about|safe to (keep )?driv|(can|could|may) (the )?(car|vehicle) be (driven|used|taken|trusted)|(is|was) (the )?(car|vehicle) (up to|ready for|fit for|good for|ok for|okay for) (it|that|a|the|this)|up to (it|a long drive|the trip)|should i (leave|keep) (the )?(car|vehicle) (at home|parked|in the garage)|allowed to (just )?keep driving|condition to (be )?(driven|drive|go|travel|use)|(good|fine|ok|okay|fit|ready|safe|reliable|up to it|capable).{0,6}enough.{0,20}(for|to) (that|it|the trip|a trip|a long|driving|the drive)|(planning|plan|want|going|need|have|about) to (drive|go|travel|take|head).{0,70}(is|will|can|would|should|could).{0,30}(car|vehicle).{0,25}(ok|okay|fine|good|ready|fit|safe|manage|handle|make it|enough|up to)"),
    ("longest_trip", r"longest (trip|drive|journey)(?!.*(under|stayed|idle|parked|stationary))|(biggest|farthest|furthest) (trip|drive|journey)|trip.{0,10}took the longest"),
    ("last_trip", r"^(what|how|tell me|show me|give me).{0,20}(last|latest|previous|most recent) (trip|drive|journey)[?. ]*$"),
    ("charging_now", r"(is|am) (the )?(car|battery|it|i).{0,12}(charging|plugged|connected to a charger)|am i plugged in|charging right now|^is it charging"),
    ("drive_modes", r"(?:how many|number of|what|which|list).{0,20}(?:driver[- ]?selectable |selectable )?(?:drive|driving) modes?\b|(?:drive|driving) modes?.{0,20}(?:does|has|available|supported|fitted|configured)"),
    ("improve_score", r"(?:improve|increase|raise|boost|better).{0,20}(?:driving |driver |eco )?score\b|(?:driving |driver |eco )?score.{0,20}(?:improve|increase|raise|boost|better)"),
    ("city_highway", r"(?:difference|compare|comparison|versus|vs\.?|between).{0,35}\bcity\b.{0,25}\b(?:highway|motorway)\b|\bcity\b.{0,25}\b(?:versus|vs\.?|and|compared (?:to|with))\b.{0,20}\b(?:highway|motorway)\b"),
    ("style", r"(am i|do i|is my).{0,12}(brak|accelerat).{0,12}(too )?(hard|harsh|aggressiv|fast)|how aggressive|aggressive driv|driving score|driver score|score.{0,10}driving|rate my driving|how smooth|how good.{0,10}driv|eco score|how.?s my driving|how do i drive\b|how am i driving|how well do i drive|do i drive (well|smoothly|badly|aggressively)"),
    ("style_trend", r"(is|has) my driving (getting|got|become|been).{0,12}(worse|better|smoother|rougher)|is my driving (improving|worse|better)"),
    ("dtc_explain", r"(what|mean|explain|tell me about).{0,15}\b[pbcu][0-9a-f]{4}\b|what does (that|this|the) (code|fault|error|dtc)|what (is|are) (that|this|the|a) (code|dtc|fault code|trouble code)|meaning of.{0,10}code|explain.{0,10}(the )?(code|dtc)|(explain|decode|interpret|translate).{0,20}(diagnostic |trouble |fault |error )?(code|dtc)"),
    ("battery_age", r"how old is (the|my) battery|battery.{0,10}\bage\b|how much (battery )?capacity (have i|did i|has it) lost|capacity (loss|lost)|^will (my|the) battery last[?. ]*$"),
    ("top_speed", r"how fast did i (go|travel)\b(?!.*average)|how fast can (i|we|the (?:car|vehicle)|this (?:car|vehicle)|it) (go|travel)|fastest i (went|drove)|the fastest i.?ve (gone|been|driven|done)|fastest (i have|i.ve) (gone|been|driven)|how fast have i (been|gone)|the fastest (speed )?i.?ve (driven|gone|been|done|reached)|fastest speed (i|the car) (has|have|did)"),
    ("places_most", r"(which|what) places.{0,15}(most|often|visit|go)|where do i (go|drive|charge) (most|often|usually)|where did i charge|most (visited|common) (place|destination)|favou?rite (place|destination)|where do i (usually|normally|often|mostly|typically|generally) (drive|go|travel)|where (does|did) the (car|vehicle) (usually|normally|typically|mostly) go|my (usual|regular|common) destinations"),
    ("commute", r"\bcommute\b|usual (route|trip)|regular (route|trip)|how long is my (usual|regular|normal|typical|daily|everyday)|how long does my (usual|regular|daily) (drive|trip|route)"),
    ("leave_time", r"what time did i (leave|depart|set off)|when did i (leave|depart|set off)( home| work| the house)?|when did i last (leave|go out of) (home|the house)|last (left|time i left) home|when did i last (go|get|head|step) out|last time i (went|got|headed|stepped) out"),
    ("compare_months", r"(compare|vs|versus|against|than).{0,25}(this|last|previous) month|month.{0,6}(on|over|vs|versus).{0,6}month|(this|last) month.{0,20}(compare|vs|versus|than)"),
    ("briefing", r"^(hi |hello |hey )?good (morning|afternoon|evening)[.! ]*$|\b(brief me|briefing|pre.?(drive|trip) check|before i (drive|set off|go|leave)|anything i (should|need to) know|how (are|is) things|start of (the )?day)\b"),
    ("health_summary", r"summari[sz]e .*(health|events)|vehicle health based"),
    ("unknowable", r"firmware|software version|which can (message|frame).{0,30}(behind|caus|responsible)|\bcan id\b|arbitration id|manufacturer'?s? sensor|which manufacturer|(point out|identify|name|which).{0,25}(broken|failed|failing|faulty|defective|bad) (component|part|sensor)|(company|brand|maker|vendor|supplier|who made|who built|what company).{0,30}(sensor|part|component)|what will .{0,40}\b(be|do|happen|read)\b.{0,30}in (exactly |about )?(an?|\d+) (hour|minute|second|day)s?|(exact|which|what) (mechanical )?part (is |was |will |has |might |failed |broke )?(failing|faulty|broken|defective|damaged|worn|bad|wrong)|mechanical part|sensor manufacturer|manufacturer of|which component (is|was) (failing|faulty|broken)|\bpredict|what will happen|next (fault|failure|warning)|exact time of the next"),
    ("memory", r"\b(you said|earlier you said|you told me|you mentioned|you claimed|you reported)\b|are you (certain|sure)"),
    ("recheck", r"re-?check|double.?check|verify your previous|using the raw data"),
    ("counter", r"contradict|evidence against|disprove|argues against|speaks against"),
    ("missing", r"cabin temp|interior temp|can.?fd|all (the )?(can )?signals|list (all )?(the )?signals|rear.?(left|right)|(left|right).{0,6}(rear|side)|cabin.{0,20}(temp|warm|hot|cold|cool)|(warm|hot|cold|cool)\w*.{0,20}\bcabin\b|side camera|other camera|which camera (captured|recorded|took|saw|detected|caught|produced)|camera see|radar"),
    ("abn_cond", r"abnormal (battery |pack )?temp\w*.{0,50}(associated|related|linked|cluster\w*|certain|particular|conditions?)|(hot|abnormal) .{0,30}driving conditions"),
    ("cluster", r"correlate .{0,40}(gps|position|location)|concentrated|same (route )?location|cluster\w* (around|at|in|on) (the )?(same )?(route|place|location|spot|stretch)|(gps|position|location|where the (car|vehicle) was).{0,40}(fault|warning|event|code)|(fault|warning|event|code)s?.{0,40}(gps|position|location|stretch of road|where the (car|vehicle) was)|same stretch"),
    ("day", r"what (happened|occurred|went on|was going on) (on|during|in) |how (much|far) (did|have) (i|we) (driven?|travel\w*|gone|go) (on|during) |how was (the )?(day|vehicle|car) on |what (happened|occurred|went on) (yesterday|today|last night|this morning)|how was (yesterday|today) for the (vehicle|car)"),
    ("cam_event", r"camera event"),
    ("proximity", r"^(?=.*\b(diagnostic|dtc|fault|trouble|codes?|logged)\b).*(within \d+ ?(s\b|sec|seconds|min)\w* of (the |a )?(last |latest )?(abnormal|anomal|warning|fault)|\d+ ?(s\b|sec|seconds|min)\w* (either side|around|before or after|on either side)|within (half a minute|a minute|a few seconds))"),
    ("rpm_lowspeed", r"rpm (was|is|were) high|high rpm.{0,40}low.{0,10}speed|(high|big|heavy|lots of|a lot of|hard).{0,12}(motor )?(load|power|torque|working).{0,40}(low|slow|crawl\w*|creep\w*)"),
    ("multi", r"changed significantly within|within \d+ ?(s\b|sec|seconds)\b.*(changed|change)"),
    ("lag", r"before or after .{0,30}(increase in )?(engine )?load|(rise|rose|increase).{0,25}(before or after|after or before).{0,30}load|(temperature|heat|overheating).{0,40}(go up|rise|rose|increase).{0,25}(before|after).{0,30}(load|power|demand)"),
    ("idle_before", r"(before|preceding|led up to|leading up to|came before|happened before|prior to) (the )?longest (idle|period|stop|pause)"),
    ("rpm_after", r"(after|following|once|when|as soon as).{0,60}(exceed\w*|above|over|crossed|passed|reached|past|hit|beyond).{0,20}\d[\d,]* ?rpm"),
    ("duration", r"(average|mean) speed.{0,40}\brpm\b|(how long|for how long|how many (minutes|seconds|hours)|calculate how many|duration).*(rpm|km/h|idl)|calculate.*(average speed|mean speed).*rpm|(median|average) rpm.*accelerat|percentage of (the )?(driving )?time.*(above|over|below)|(minutes|time).*\bidl"),
    ("longest", r"longest (continuous |single )?(period|stretch|time|idle|stop|pause)|longest idle"),
    ("extreme", r"\brpm\b.{0,25}(highest|max|peak|top)|(highest|max\w*|peak|top).{0,12}\brpm\b|at the (highest|lowest|maximum|peak)[- ]\w+ point|(highest|lowest|maximum|minimum|max|min).{0,40}(recorded|ever)\b|\b(recorded|ever)\b.{0,20}(highest|lowest)"),
    ("point", r"(?:\bat|@|around|about) ?\d{1,2}:\d{2}|\b\d{1,2}:\d{2}:\d{2}|\bat \d{1,2} ?(am|pm)\b"),
    ("brakes_info", r"how many (brakes|brake)|what (kind|type) of brakes|brake (type|system|hardware)"),
    ("trip", r"(average|mean) (vehicle )?speed (during|of|on|for) (the |this )?last trip|last trip.*(average|speed)|average speed.{0,15}\b(last|latest) trip"),
    ("avgtemp_mode", r"(average|mean) (battery|pack) temp\w* during (motorway|city)"),
    ("temprise", r"(maximum|max|largest|greatest|biggest).{0,25}(temperature|temp) (rise|increase|gain)|temperature rise over any"),
    ("tripdiff", r"difference in energy.{0,30}between|(highest|best).{0,12}(and|vs).{0,12}(lowest|worst).{0,12}efficiency trips"),
    ("tripwarn", r"percent(age)? of (the )?(trips|drives)|share of (the )?(trips|drives)|% of (the )?trips|proportion of (the )?(trips|drives)|fraction of (the )?(trips|drives)|how many (trips|drives) (had|contained|with)"),
    ("why_trip", r"why (was|is|does) .{0,30}(this|the last|that|my last) (trip|drive|journey).{0,30}(efficien|energy|consum|worse|poor|bad|high|hungry)|(this|last|latest) (trip|drive).{0,30}(inefficien|used so much|used more)|why did .{0,40}(efficiency|consumption|energy).{0,25}(drop|fall|rise|worse|increase).{0,25}\b(this|the last|that) trip|(this|the last) trip.{0,40}(efficiency|consumption).{0,25}(drop|worse|why)"),
    ("why_event", r"what (caused|made|triggered) .{0,30}overheat\w*|why .{0,30}overheat\w*|why do you think .{0,40}(overheat|event|warning|fault|anomal)|why did .{0,25}(overheat|the warning|the fault)|why (does|did) .{0,40}(overheat|warning).{0,20}(occur|happen)"),
    ("bestcombo", r"combination of signals|best explains"),
    ("periods", r"exact time periods|time periods where|which periods|exact periods|when exactly did the change"),
    ("signals_during", r"what other (vehicle )?signals (changed|moved) during|other signals changed during|when (fuel )?efficiency dropped.{0,20}(conditions|present)|(going on|happening|present|different).{0,25}(when|around the time|while|at the time).{0,30}efficien"),
    ("evidence", r"what evidence|evidence supports|supporting evidence|what supports (that|your)|how do you know"),
    ("top3", r"top (three|3)|three (possible|most likely) explanations"),
    ("obs_inf", r"directly observed|observed and which|observed.{0,30}inferred|(measur\w*|recorded|read|facts?).{0,40}(guess\w*|infer\w*|assum\w*|work\w* out)"),
    ("prove", r"\bprove\b|\bcausation\b|\bcaused the\b"),
    ("alt", r"alternative explanation|another explanation|other explanation|could there be another|different explanation|alternative"),
    ("conclusion", r"your conclusion.{0,30}(supporting|confidence)|conclusion, the supporting"),
    ("missing_info", r"what (data|information).{0,15}(missing|would you need|do you need)|missing.{0,40}(distinguish|reliable|conclusion)|data are you missing"),
    ("conf", r"how confident|how sure|how certain|confidence level"),
    ("calc", r"exact timestamps and calculations|show (me )?the calculation|how did you calculate|show your (working|calculation)|calculations used"),
    ("cooccur", r"what other signals changed (at the same time|when|while)|when battery temperature increased unusually|what else (moved|changed|happened|was going on)"),
    ("corr", r"relationship between .{0,30}(speed|temperature)|correlation between|is there a (relationship|link|correlation)|temperature.{0,30}(related|linked|connected|associated|depend\w*).{0,15}speed|speed.{0,30}(affect|influence|relate)\w*.{0,20}temperature|(faster|speed\w*).{0,20}(hotter|hot)"),
    ("pairs", r"two (apparently )?unrelated|changed at nearly the same time"),
    ("overheat_cond", r"overheat\w* events? .{0,90}(accelerat|cruis|idl)"),
    ("conditions", r"which driving condition|conditions? (produce|cause|lead to|give)s? .{0,15}(highest|most|more).{0,15}(consumption|energy)|highest energy consumption"),
    ("changes", r"three biggest changes|biggest changes in (the )?(vehicle )?behaviou?r|(main|biggest|largest|top|key).{0,15}(differences|changes).{0,40}(behav|this week|the week)"),
    ("compare_periods", r"compare (this|the) week|this week.{0,25}last week|first (recorded )?week.{0,60}(final|most recent|last|recent|latest)|compare the first|statistically"),
    ("degrade", r"gradual degradation|degraded the most|slow(ly)? degrad|signal.{0,20}degrad|\bdegradation\b|slowly (getting|got|become|became) worse|without a fault|no single fault"),
    ("why_range2", r"(worse|poor|bad|less) (mileage|efficiency|range)|why .{0,20}(feel|seem).{0,25}(mileage|efficiency|range)"),
    ("trend", r"(going|went|creeping|climbing|drifting) (up|down)|over the (past |last )?(weeks|days|month)|running (hotter|warmer|cooler)|\blately\b.{0,30}(hotter|worse|higher)|trending|\btrends?\b|getting (worse|better|hotter|warmer|higher|lower)|become more|becoming more|volatil|over time|has (the )?(average |typical |mean )?(\w+ )?(temp\w*|rpm)( rate| level)? (changed|drifted|moved|shifted)|(gotten|got) worse|is my efficiency (rising|falling|dropping|increasing)"),
    ("worstday", r"worst (24.hour|day|period)"),
    ("badday", r"(was|were) (yesterday|today).{0,25}(bad|good)|bad day"),
    ("recurring", r"recurring (anomal|issue|problem|pattern)"),
    ("repeat", r"same fault|fault.{0,15}(repeatedly|again)|repeatedly"),
    ("firstlast", r"first occurrence.{0,30}(most recent|last)|between the first and (the )?(most recent|last)"),
    ("precursors", r"recurring sequences|sequences?.{0,30}before (the )?(failures?|faults?)"),
    ("earliest", r"earliest (indication|sign)|first sign|first indication"),
    ("similar", r"similar fault|have in common|compare all similar"),
    ("history", r"(analy[sz]e|review) the entire (vehicle )?(history|dataset)|most significant abnormal"),
    ("serious", r"most serious event|before the most serious"),
    ("engineer", r"service engineer"),
    ("investigate", r"investigate first|what should i (look at|check|investigate)|where should i start"),
    ("ctx", r"(before|after|leading up|led (up )?to|prior to|preced\w*|immediately|between the .{0,25}(before|after)|signals changed before).{0,80}(overheat\w*|over.?temp\w*|warning|adas|diagnostic|fault|abnormal|anomal|event|\bidle\b)|(overheat\w*|warning|adas|fault).{0,50}(sequence|leading up)|sequence of events|did the temperature rise|accelerating or decelerating"),
    ("speedwin", r"(how fast|what speed|average speed|how quickly).{0,30}\b(yesterday|today|last night|this morning)\b"),
    ("events", r"(" + EVKIND + r")"),
    ("checkup", r"feels? sluggish|did anything look weird|anything unusual|something changed|behaving normally|anything suspicious|look weird|seems off|feels off|acting (up|strange)"),
    ("missing_soft", r"throttle|pedal position|accelerator|steering angle|engine load"),
]]

# the question text is canonicalised with "car" -> "vehicle" (qa.prep), so the everyday routes must accept both spellings
_EVERYDAY = {"briefing", "explain_simple", "pump_ok", "v12", "battery_now", "best_day", "eff_day", "last_charge", "charge_cost", "charge_time", "speeding", "improve_range", "hot_battery", "unusual", "eff_trend", "soh_rate", "motor_now", "overview", "reach", "advice",
             "longest_trip", "last_trip", "charging_now", "style", "style_trend", "dtc_explain", "battery_age", "top_speed", "places_most", "commute", "leave_time", "compare_months", "compare_days", "trip_count", "regen", "signal_why", "drive_safely"}
ROUTES = [(n, re.compile(re.sub(r"(?<![a-z])car(?![a-z])", "(?:car|vehicle)", rx.pattern)) if n in _EVERYDAY else rx) for n, rx in ROUTES]

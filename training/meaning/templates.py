"""Surface-form patterns for the compositional part of the training data.

A pattern is a sentence with slots; its LABEL is fixed by the (act, op, cond, vs, ent) of the group it belongs to and the concept it is filled with.  Nothing here decides what a user
means at run time: these strings only generate training sentences.  Slots: {c} concept noun phrase (from vgpt/meaning/lexicon.json + the glossary), {t} a period phrase (or empty),
{k} a condition suffix, {k2} the second condition, {t1}/{t2} two periods, {at} an instant, {thr} a threshold, {n} a number, {u} an unmodelled thing.
Families (the list index inside a group) are what the validation split holds out, so a pattern seen in training never appears in validation."""

PERIODS = ["last week", "yesterday", "this week", "this month", "last month", "today", "over the last 3 days", "in the past 24 hours", "over the last 14 days", "during the last two weeks",
           "in the last hour", "since monday", "over the past 7 days", "this morning", "last night", "in the last 30 days", "for the past 10 days", "the whole time", "since i got the car", "last weekend", ""]
AT_POOL = ["at 3 pm yesterday", "two hours ago", "at 8 this morning", "yesterday at noon", "at 18:30 on tuesday", "last night at 11", "30 minutes ago", "3 days ago at 10 am", "at midnight",
           "this morning at 7:15", "at 5 pm today", "10 minutes back", "at 9 yesterday evening", "on monday at 14:00", "an hour ago", "at lunchtime", "at 2 am", "yesterday at 6 pm"]
PERIOD_PAIRS = [("this week", "last week"), ("this month", "last month"), ("today", "yesterday"), ("last week", "the week before"), ("the last 7 days", "the 7 days before that"),
                ("this month", "the previous month"), ("the past 24 hours", "the 24 hours before"), ("now", "a month ago")]
COND_PHRASES = {
    "highway": ["on the highway", "on the motorway", "when driving on the freeway", "during highway driving", "at highway speeds on the expressway"],
    "city": ["in the city", "in city traffic", "when driving around town", "during urban driving", "in stop and go traffic"],
    "charging": ["while charging", "when plugged in", "during charging", "while on the charger"],
    "parked": ["while parked", "when the car is parked", "when it's standing", "while the car is switched off"],
    "driving": ["while driving", "when the car is moving", "during drives", "on the move"],
    "fast_charging": ["while fast charging", "during dc fast charging", "when rapid charging"],
    "raining": ["in the rain", "when it was raining", "during rainy weather", "on wet days"],
    "hard_braking": ["during hard braking", "when braking hard", "while slamming the brakes"],
    "hard_acceleration": ["during hard acceleration", "when accelerating hard", "while flooring it"],
    "regenerating": ["while regenerating", "during regen", "when the car is regenerating"],
    "night": ["at night", "after dark", "during the night", "at nighttime"],
    "cold": ["in cold weather", "when it's cold", "on cold days", "in winter"],
    "hot_weather": ["in hot weather", "when it's hot outside", "on hot days", "in summer"],
    "high_speed": ["at high speed", "when going fast", "above highway speeds"],
    "idling": ["while idling", "when stationary", "when standing still", "at a standstill"],
}
COND_NOUN = {"highway": ["highway driving", "motorway driving", "freeway use"], "city": ["city driving", "urban driving", "town driving"], "charging": ["charging", "charge sessions"],
             "parked": ["parked time", "when parked"], "driving": ["driving", "moving"], "fast_charging": ["fast charging", "dc charging"], "raining": ["rain", "wet weather"],
             "night": ["night driving", "night time"], "cold": ["cold weather", "winter"], "hot_weather": ["hot weather", "summer"], "idling": ["idling", "standing still"],
             "hard_braking": ["hard braking"], "hard_acceleration": ["hard acceleration"], "regenerating": ["regen"], "high_speed": ["high speed"]}
STATE_PHRASE = {"parked": ["parked", "standing still with the car off"], "charging": ["charging", "plugged in"], "highway": ["on the highway", "on the motorway"], "city": ["in the city", "in town"],
                "driving": ["driving", "on the move"], "idling": ["idling", "stationary"]}
THRESH = {"km/h": ["above 100 km/h", "over 80 km/h", "below 30 km/h", "faster than 120 km/h"], "C": ["above 40 degrees", "over 35 C", "below 5 degrees", "hotter than 45 C"],
          "%": ["below 20 percent", "above 90 percent", "under 30%", "over 80 percent"], "kW": ["above 50 kW", "over 100 kW", "below 5 kW"], "kPa": ["below 220 kpa", "under 200 kPa", "above 260 kPa"],
          "V": ["below 12 volts", "above 13 V"], "L/min": ["below 5 L/min", "under 3 litres per minute"], "rpm": ["above 5000 rpm", "over 8000 rpm"], "Nm": ["above 200 nm", "over 150 Nm"],
          "Wh/km": ["above 200 Wh/km", "under 130 Wh/km"], "m/s2": ["above 3 m/s2", "over 2 m/s2"], "g": ["above 0.4 g", "over 0.3 g"], "W": ["above 2000 W", "over 1500 W"], "deg": ["beyond 90 degrees", "over 45 degrees"]}
NUMBERS = ["20", "30", "40", "50", "60", "80", "100", "120", "150", "200", "250", "300"]
UNLISTED = ["washer fluid level", "wiper fluid", "engine oil level", "oil pressure", "wheel alignment", "shock absorbers", "exhaust", "clutch", "sunroof", "seat heater", "air filter", "headlights",
            "windscreen wipers", "cabin co2", "pollen in the cabin", "tyre brand", "paint colour", "airbags", "door lock state", "steering fluid", "coolant reservoir level", "boot space", "trunk",
            "horn", "mirror heating", "rear window defroster", "fuel cap", "gearbox oil", "spark plugs", "radio volume", "number of keys", "window position", "seat position", "tow bar"]

# (act, op, cond-mode, [patterns]).  cond-mode: "-" none, "k" a condition (suffix {k}), "kv" two conditions, "pv" two periods, "thr" threshold, "ent:trip|charge|day" an entity.
# {c} is always the concept noun phrase.  Each pattern list is a set of "families": the index is the family id used for the validation split.
NUM = ("signal", "derived", "declared")
GROUPS = [
    ("now", "none", "-", ["what is the {c} right now", "{c} now", "current {c}", "what's my {c} at the moment", "tell me the {c}", "show me the {c}", "give me the live {c}", "check the {c} for me",
                           "what's the {c} currently", "can you tell me the current {c}", "i want to know the {c}", "read out the {c}", "whats the {c}", "any idea what the {c} is", "{c} reading please",
                           "how much is the {c} now", "the {c} please", "quickly what's the {c}", "hey what is the {c}", "report the {c}"], NUM),
    ("at", "none", "-", ["what was the {c} {at}", "{c} {at}", "how much was the {c} {at}", "tell me the {c} {at}", "what was my {c} {at}", "{c} reading {at}", "give me the {c} from {at}",
                          "i want the {c} {at}", "do you have the {c} {at}", "look up the {c} {at}"], ("signal",)),
    ("stat", "mean", "-", ["what was my average {c} {t}", "average {c} {t}", "mean {c} {t}", "what's the typical {c} {t}", "{c} on average {t}", "avg {c} {t}", "tell me the average {c} {t}",
                            "how much was the {c} on average {t}", "give me the mean {c} over {t}", "what did the {c} average {t}", "calculate the average {c} {t}", "overall {c} {t} on average"], NUM),
    ("stat", "mean", "k", ["average {c} {k} {t}", "what is my mean {c} {k} {t}", "{c} {k} on average {t}", "how high is the {c} on average {k} {t}", "typical {c} {k} {t}"], NUM),
    ("stat", "median", "-", ["median {c} {t}", "what was the median {c} {t}", "the middle {c} value {t}", "typical median {c} over {t}"], NUM),
    ("stat", "sum", "-", ["total {c} {t}", "how much {c} {t} in total", "what is the total {c} {t}", "add up the {c} {t}", "sum of {c} {t}", "overall {c} {t}", "how much {c} did i have {t}"],
     ("energy_used", "distance_driven", "regen_energy")),
    ("stat", "spread", "-", ["how much does the {c} vary {t}", "how variable is the {c} {t}", "variation in {c} {t}", "how steady is the {c} {t}", "how consistent is the {c} {t}"], NUM),
    ("stat", "share", "k", ["what share of the time was the car {s} {t}", "what percentage of {t} was i {s}", "how much of the time was it {s} {t}", "percent of time {s} {t}"], ("drive_mode",)),
    ("stat", "duration", "k", ["how long was the car {s} {t}", "total time {s} {t}", "how many hours was it {s} {t}", "time spent {s} {t}", "for how long was i {s} {t}"], ("drive_mode",)),
    ("stat", "duration", "thr", ["how long was the {c} {thr} {t}", "how many minutes was the {c} {thr} {t}", "total time the {c} was {thr} {t}", "for how long was {c} {thr} {t}"], NUM),
    ("stat", "share", "thr", ["what percent of the time was the {c} {thr} {t}", "how often was the {c} {thr} {t}", "share of time with the {c} {thr} {t}", "how much of the time was the {c} {thr} {t}"], NUM),
    ("extreme", "max", "-", ["highest {c} {t}", "what was the maximum {c} {t}", "peak {c} {t}", "when was the {c} at its highest {t}", "how high did the {c} get {t}", "top {c} {t}", "max {c} {t}",
                              "what's the most {c} reached {t}", "greatest {c} {t}", "the biggest {c} {t}"], NUM),
    ("extreme", "min", "-", ["lowest {c} {t}", "what was the minimum {c} {t}", "when was the {c} lowest {t}", "how low did the {c} go {t}", "minimum {c} {t}", "smallest {c} {t}", "the least {c} {t}"], NUM),
    ("extreme", "max", "ent:day", ["which day had the highest {c}", "on which day was the {c} at its peak", "day with the most {c}", "which day was the {c} highest {t}"], ("energy_used", "distance_driven", "wh_per_km", "regen_energy")),
    ("extreme", "min", "ent:day", ["which day had the lowest {c}", "day with the least {c}", "which day was the {c} lowest {t}"], ("energy_used", "distance_driven", "wh_per_km", "regen_energy")),
    ("extreme", "max", "ent:trip", ["which trip had the highest {c}", "my trip with the most {c}", "which drive had the biggest {c} {t}"], ("energy_used", "distance_driven", "wh_per_km", "regen_energy")),
    ("trend", "change", "-", ["how has the {c} changed {t}", "{c} trend {t}", "show me the trend in {c} {t}", "has the {c} been changing {t}", "what's happening with my {c} {t}", "how did the {c} move {t}",
                               "is the {c} stable {t}", "{c} over time {t}"], NUM),
    ("trend", "drop", "-", ["has the {c} been dropping {t}", "is the {c} decreasing {t}", "when did the {c} drop the most {t}", "largest fall in {c} {t}", "did the {c} go down {t}", "is {c} getting lower {t}",
                             "biggest decrease in {c} {t}"], NUM),
    ("trend", "rise", "-", ["has the {c} been rising {t}", "is the {c} increasing {t}", "when did the {c} jump the most {t}", "largest rise in {c} {t}", "did the {c} go up {t}", "is {c} getting higher {t}",
                             "biggest increase in {c} {t}"], NUM),
    ("compare", "mean", "kv", ["compare {c} {k} and {k2}", "{c} {k} versus {k2}", "how does {c} differ {k} and {k2}", "is {c} higher {k} than {k2}", "difference in {c} between {k} and {k2}",
                                "{c}: {k} against {k2}", "which is higher, {c} {k} or {k2}"], NUM),
    ("compare", "mean", "pv", ["compare {c} {t1} with {t2}", "{c} {t1} versus {t2}", "how does {t1}'s {c} compare to {t2}", "is the {c} better {t1} than {t2}", "{c} {t1} against {t2}", "difference in {c} between {t1} and {t2}"], NUM),
    ("events", "list", "-", ["show all {c} {t}", "list the {c} {t}", "what {c} happened {t}", "give me every {c} {t}", "show me the {c} {t}", "{c} {t} list", "display the {c} from {t}"], ("event",)),
    ("events", "count", "-", ["how many {c} {t}", "number of {c} {t}", "count the {c} {t}", "how many times did {c} happen {t}", "total {c} {t}", "count of {c} {t}"], ("event",)),
    ("events", "exists", "-", ["was there any {c} {t}", "did any {c} happen {t}", "any {c} {t}", "has there been a {c} {t}", "did the car have a {c} {t}", "is there a {c} {t}", "were there {c} {t}"], ("event",)),
    ("events", "last", "-", ["when was the last {c}", "when did {c} last happen", "most recent {c}", "latest {c}", "when was the previous {c}", "what was the last {c}"], ("event",)),
    ("events", "first", "-", ["when was the first {c}", "earliest {c} on record", "the very first {c}", "when did the {c} first occur"], ("event",)),
    ("context", "none", "-", ["what happened before the {c}", "what led up to the {c}", "what was going on around the {c}", "show the signals around the {c}", "what preceded the {c}",
                               "conditions just before the {c}", "walk me through what happened before the last {c}", "context of the {c}", "what was the car doing before the {c}"], ("event",)),
    ("why", "none", "-", ["why did the {c} happen", "why was there a {c}", "what caused the {c}", "reason for the {c}", "what's behind the {c}", "explain the {c}", "what triggered the {c}", "why the {c}"], ("event",)),
    ("why", "drop", "-", ["why has the {c} dropped {t}", "why is the {c} low", "what made the {c} go down {t}", "why did my {c} fall {t}", "why is the {c} getting worse {t}"], NUM),
    ("why", "rise", "-", ["why is my {c} so high", "why has the {c} gone up {t}", "what is pushing the {c} up {t}", "why is the {c} increasing {t}", "why did my {c} jump {t}"], NUM),
    ("spec", "none", "-", ["what is the {c}", "tell me the {c}", "what's our {c}", "how much is the {c}", "{c} of this car", "what is the {c} of the vehicle", "do you know the {c}", "i need the {c}",
                           "give me the {c}", "can you tell me the {c} of my car"], ("spec",)),
    ("define", "none", "-", ["what is {c}", "what does {c} mean", "explain {c}", "define {c}", "meaning of {c}", "what's {c} in simple terms", "can you explain what {c} is", "tell me about {c}",
                              "what do you mean by {c}", "{c} explained"], ("signal", "derived", "event", "spec", "declared")),
    ("howto", "none", "-", ["how do i calculate {c}", "how is {c} computed", "what's the formula for {c}", "how to work out {c}", "how can i convert {c}", "how do i read the {c}", "how to find {c}"],
     ("signal", "derived", "spec")),
    ("advice", "none", "-", ["how can i improve my {c}", "tips for {c}", "what should i do about the {c}", "should i be worried about the {c}", "how do i keep the {c} healthy", "any advice on {c}", "how to take care of {c}"],
     ("vehicle_health", "battery_condition", "tyre_condition", "brake_condition", "driving_style", "maintenance", "range_estimate", "wh_per_km", "motor_condition", "cooling_condition")),
    ("health", "none", "-", ["how is the {c}", "is the {c} ok", "how's my {c} doing", "check the {c}", "is there anything wrong with the {c}", "{c} status", "are the {c} fine", "give me a health check of the {c}",
                              "any problems with the {c}"], ("battery_condition", "motor_condition", "cooling_condition", "tyre_condition", "brake_condition", "aux_battery_condition", "vehicle_health",
                                                              "battery_soh", "brake_pad_life", "driving_style")),
    ("estimate", "none", "-", ["how much {c} will i need", "estimate the {c}", "what is the {c} going to be", "work out the {c}", "calculate my {c}", "predict the {c}"],
     ("range_estimate", "charge_time", "charge_cost", "maintenance", "energy_used")),
    ("now", "unlisted", "-", ["what is the {u}", "how is the {u}", "current {u}", "check the {u}", "{u} now", "tell me the {u}"], ()),
]

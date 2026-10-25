# full_v11 against the earlier adapters

Written 2026-10-20 from `results/meaning/autopilot_state.json` (frozen sets, model alone) and `results/meaning/e2e_world_*.json` (the whole assistant on the 105-day world).

## 1. The model alone (exact six-word frame, no arbitration, no dialogue rules)

| set | questions | v6 (before today) | v8 | **v11** | v11 minus v6 |
|---|---|---|---|---|---|
| platform_v1 | 90 | 21.1 | 93.3 | **92.2** | +71.1 |
| platform_v2 | 80 | 25.0 | 88.7 | **95.0** | +70.0 |
| adv_v1 | 112 | 58.0 | 86.6 | **94.6** | +36.6 |
| adv_v2 (sealed) | 112 | 60.7 | 87.5 | **92.9** | +32.2 |
| adv_v3 (sealed) | 80 | 57.5 | 86.3 | **95.0** | +37.5 |
| novel_v1 | 287 | 93.4 | 93.4 | **95.8** | +2.4 |
| messy_v1 | 70 | 92.9 | 91.4 | **94.3** | +1.4 |
| user_v1 | 67 | 92.5 | 91.0 | **92.5** | +0.0 |
| cs5 | 62 | 80.6 | 88.7 | **85.5** | +4.9 |
| sealed_v2 (sealed) | 149 | 94.0 | 94.0 | **92.6** | -1.4 |
| sealed_v3 (sealed) | 149 | 93.3 | 95.3 | **94.6** | +1.3 |
| hard_v1 | 220 | 78.6 | 82.7 | **82.7** | +4.1 |
| hard_v2 | 220 | 88.2 | 87.7 | **86.8** | -1.4 |
| hard_v3 | 220 | 87.7 | 90.9 | **91.4** | +3.7 |
| hard_v4 | 220 | 79.5 | 79.5 | **77.3** | -2.2 |
| hard_v5 | 100 | 94.0 | 96.0 | **95.0** | +1.0 |

* The five sets about the new vocabulary and hard wording (platform, adv): v6 44.5 % -> v11 **93.9 %** (mean of the five sets).
* The eleven older sets: v6 88.6 % -> v11 89.9 % (mean); none is more than 2.2 points lower, hard_v4 (contested gold) is the one that fell.
* adv_v2 and adv_v3 are SEALED: never read for failures, never trained on.  v6 scored 61 and 58 on them, v11 93 and 95.

## 2. The whole assistant on the 105-day world (what the person actually gets)

For each question: the answer the assistant gives, against the answer a perfect reader of the question would have got from the same data at the same moment (strict text equality, so it under-counts equivalent answers).

| set | v6 + old rules (the product this morning) | v8 + old rules | **v11, rules off (the product now)** |
|---|---|---|---|
| platform_v1 | 30.0 | 87.8 | 92.2 |
| platform_v2 | 36.2 | 88.8 | 95.0 |
| adv_v1 | 64.3 | 85.7 | 94.6 |
| adv_v2 | 66.1 | 83.9 | 92.9 |
| adv_v3 | 67.5 | 86.2 | 95.0 |
| novel_v1 | 92.0 | 92.0 | 96.9 |
| messy_v1 | 94.3 | 92.9 | 94.3 |
| user_v1 | 91.0 | 89.6 | 92.5 |
| sealed_v2 | 94.6 | 95.3 | 95.3 |
| sealed_v3 | 89.9 | 91.3 | 94.6 |
| hard_v1 | 75.0 | 79.5 | 83.6 |
| hard_v2 | 88.6 | 87.7 | 90.0 |
| hard_v3 | 84.5 | 88.2 | 91.8 |
| hard_v4 | 79.1 | 79.1 | 78.6 |
| hard_v5 | 90.0 | 91.0 | 95.0 |
| **all** | **79.6** (n=2176) | **87.5** (n=2176) | **91.3** (n=2176) |

| adapter | answer latency p50 | p95 |
|---|---|---|
| v6 | 990 ms | 1383 ms |
| v8 | 986 ms | 1406 ms |
| v11_final (measured while other work was running on the machine) | 827 ms | 1363 ms |
| **v11, on an idle machine (249 questions)** | **745 ms** | **984 ms** |

v11 answers 322 questions correctly that v6 got wrong; v6 answered 68 correctly that v11 gets wrong.

### Examples (same question, same data)

**did the car just black out on me in the middle of a drive**  (adv_v1)
* v6: There were 26 ADAS disengagements over the whole recorded history. / Latest: 3 Oct, 11:42 system limit; 6 Oct, 08:24 driver override; 10 Oct, 12:31 camera blocked / Based on 
* v11: There was 1 unexpected shutdown over the whole recorded history. / Latest: 13 Sep, 14:02 unexpected shutdown (12 V supply loss); last valid data 120 ms before / Based on 1 re

**has the computer in the dash ever frozen up and restarted itself**  (adv_v1)
* v6: There were 26 ADAS disengagements over the whole recorded history. / Latest: 3 Oct, 11:42 system limit; 6 Oct, 08:24 driver override; 10 Oct, 12:31 camera blocked / Based on 
* v11: There were 12 ECU resets over the whole recorded history. / Latest: 1 Oct, 19:43 ADAS domain controller reset (watchdog timeout); 3 Oct, 11:22 ADAS domain controller reset 

**did the whole car go dark on me while i was on the road**  (adv_v2)
* v6: There were 26 ADAS disengagements over the whole recorded history. / Latest: 3 Oct, 11:42 system limit; 6 Oct, 08:24 driver override; 10 Oct, 12:31 camera blocked / Based on 
* v11: There was 1 unexpected shutdown over the whole recorded history. / Latest: 13 Sep, 14:02 unexpected shutdown (12 V supply loss); last valid data 120 ms before / Based on 1 re

**has one of the little computers in the car ever hung and rebooted**  (adv_v2)
* v6: There were 26 ADAS disengagements over the whole recorded history. / Latest: 3 Oct, 11:42 system limit; 6 Oct, 08:24 driver override; 10 Oct, 12:31 camera blocked / Based on 
* v11: There were 12 ECU resets over the whole recorded history. / Latest: 1 Oct, 19:43 ADAS domain controller reset (watchdog timeout); 3 Oct, 11:22 ADAS domain controller reset 

**kal ka sabse zyada speed kitna tha**  (adv_v3)
* v6: Vehicle speed is 0 km/h right now. / Based on 1 sample (1 Hz) from canonical live snapshot. / Evidence: measured, 1-second samples.
* v11: The highest vehicle speed over the last 7 days was 110 km/h, on 14 Oct, 12:46 (of one-minute averages). / At that moment: vehicle speed 110 km/h. / Based on 10,081 samples (p

**an overview of this week's hard braking would be appreciated**  (adv_v3)
* v6: Over the last 28 days, the car recorded 1875.1 km across 64 trips, with 94 hard-braking, 28 hard-acceleration and 1 hard-cornering events. These telemetry indicators desc
* v11: 22 hard-braking events this week; the latest was on 16 Oct, 20:15: 0.32 g at 38 km/h. /  - 16 Oct, 20:15: 0.32 g at 38 km/h /  - 15 Oct, 18:24: 0.33 g at 48 km/h /  - 15 Oct, 1

**feels like the battery is cooking**  (hard_v1)
* v6: Your battery: /  - Charge: 89%, about 272 km of range at your recent 180 Wh/km. /  - The battery health is 91% of its original capacity. /  - The pack temperature is 22 °C now,
* v11: Battery temperature is 22.0 °C right now. / Based on 1 sample (1 Hz) from canonical live snapshot. / Evidence: measured, 1-second samples.

**did the dashboard light up red recently**  (hard_v1)
* v6: That is not something this vehicle records, so I cannot answer it from its data. Nothing was substituted for it.
* v11: There were 254 warnings over the whole recorded history. / Latest: 15 Oct, 17:40 power limited to 70 % (motor temperature); 15 Oct, 17:41 P0A94 DC/DC converter performance 

**how much am i turning the wheel**  (hard_v2)
* v6: I understand you're asking for wheel slip. I can't calculate it from the signals currently recorded. I would need front-left wheel speed plus front-right wheel speed plus
* v11: Steering angle is 0 ° right now. / Based on 6 samples (1 Hz) from lkv_33_1s.v. / Evidence: measured, 1-second samples.

**how many volts does the little battery have**  (hard_v2)
* v6: Battery voltage is 388.4 V right now. / Based on 6 samples (1 Hz) from lkv_8_1s.v. / Evidence: measured, 1-second samples.
* v11: 12 V battery voltage is 11.7 V right now. / Based on 6 samples (1 Hz) from lkv_4_1s.v. / Evidence: measured, 1-second samples.

**kw motor nw**  (hard_v3)
* v6: Motor temperature is 23.5 °C right now. / Based on 1 sample (1 Hz) from canonical live snapshot. / Evidence: measured, 1-second samples.
* v11: Motor (mechanical) power is 0 kW right now. / Based on 6 samples (1 Hz). / Method: torque x rpm / 9549. / Evidence: computed from measurements, 1-second samples.

**where r we**  (hard_v3)
* v6: I am not sure which signal or period you mean. Tell me the quantity and the time span and I will compute it.
* v11: The car is parked at Home in the London area (GPS 51.5130 N, 0.0437 W, 19:34).

**give me a quick heads up on the car before i leave**  (hard_v4)
* v6: The most recent recorded problem is SOURCE_SILENT at 16 Oct, 17:10: Body CAN 250 kbit/s: no frames for 37 s while the vehicle is awake. Telemetry cannot guarantee that a 
* v11: Good evening. Drive with care. /  - Battery 89%, about 272 km of range at your recent 180 Wh/km. /  - Last trip: 11.7 km on 16 Oct, 183 Wh/km. /  - Needs attention - 12 V batte

**is the pack alright after the heat yesterday**  (hard_v4)
* v6: Climate-control power is 0 W right now. / Based on 1 sample (1 Hz) from canonical live snapshot. / Evidence: measured, 1-second samples.
* v11: Your battery: /  - Charge: 89%, about 272 km of range at your recent 180 Wh/km. /  - The battery health is 91% of its original capacity. /  - The pack temperature is 22 °C now,

### Where v11 is worse than v6

68 questions that v6 answered as the perfect reader would have, and v11 does not (by set: hard_v4 17, hard_v1 12, hard_v2 8, novel_v1 6, hard_v3 5, adv_v2 4, sealed_v3 4, user_v1 3, sealed_v2 3, messy_v1 2, hard_v5 2, platform_v2 1, adv_v3 1).  Strict text equality counts a harmless extra line as a difference, so some of these are not real losses; the first one per set, unfiltered:

**has the pack ever got hotter than it should**  (adv_v2)
* v6: There were 21 battery over-temperature warnings over the whole recorded history. / Latest: 19 Sep, 18:15 Battery temperature above 40 °C; 3 Oct, 13:07 Battery temperature a
* v11: The highest battery temperature over the whole recorded history was 52.9 °C, on 11 Jul, 17:52 (of one-minute averages). / Based on 150,936 samples (per-minute) from sig_1m.

**what is the range at the moment**  (adv_v3)
* v6: The current estimated driving range is about 272 km at 89% charge, based on the recent 180 Wh/km. It is an estimate, not a guaranteed distance.
* v11: Estimated range is about 272 km (battery 89.0%, at your recent 180 Wh/km). / Estimate: based on your recent consumption of 180 Wh/km / Estimate: estimated by SoC x capacity /

**how long since the thing last beeped at me**  (hard_v1)
* v6: The last warning was on 16 Oct, 17:10: Body CAN 250 kbit/s: no frames for 37 s while the vehicle is awake. / 254 warnings in total the whole recorded history. / Based on 254 
* v11: The last hard-braking event was on 16 Oct, 20:15: 0.32 g at 38 km/h. / 390 hard-braking events in total the whole recorded history. / Based on 390 recorded events. / Evidence: 

**how long have we driven in total**  (hard_v2)
* v6: Odometer is 52,557 km right now. / Based on 6 samples (1 Hz) from lkv_29_1s.v. / Evidence: measured, 1-second samples.
* v11: The car was driving for 13 h 53 min (13.9 hours) the last 7 days. Each recorded one-minute sample counts as 60 s; 833 of 10,081 recorded samples qualified.

**before the overheat warn**  (hard_v3)
* v6: In the 10 minutes before the battery over-temperature warning on 3 Oct, 18:41 (the interval 3 Oct, 18:31 to 3 Oct, 18:41; the event itself is the anchor and is not part o
* v11: For the battery over-temperature warning on 3 Oct, 18:41, the most likely contributor is reduced coolant flow (strong evidence); that is what the data supports, not a pro

**the red light came on again what is it about**  (hard_v4)
* v6: For the warning on 16 Oct, 17:10, the most likely contributor is the event's recorded cause (strong evidence); that is what the data supports, not a proven cause. /  - the 
* v11: That is not something this vehicle records, so I cannot answer it from its data. Nothing was substituted for it.

**the build up to the last trouble code**  (hard_v5)
* v6: In the 10 minutes before the diagnostic trouble code on 15 Oct, 17:41 (the interval 15 Oct, 17:31 to 15 Oct, 17:41; the event itself is the anchor and is not part of thes
* v11: 1 diagnostic trouble code over the last 7 days; the latest was on 15 Oct, 17:41: P0A94 DC/DC converter performance (status 0x09: testFailed, confirmedDTC). /  - 15 Oct, 17:

**max tyre temp**  (messy_v1)
* v6: The highest tyre temperature over the last 7 days was 43.4 °C, on 12 Oct, 17:11 (of one-minute averages). / Based on 10,081 samples (per-minute) from lkv_38_1m.v. / Data qual
* v11: Tyre temperature is 22.0 °C right now. / Based on 6 samples (1 Hz) from lkv_38_1s.v. / Evidence: measured, 1-second samples.

**how often was the battery above 40 degrees last week**  (novel_v1)
* v6: 0.0% of the driving time the last 7 days was spent with battery temperature above 40 °C (0 of 833 recorded one-minute samples, about 0 minutes).
* v11: Battery temperature went above 40 °C on 0 separate occasions the last 7 days; the longest lasted 0 min. An occasion is a stretch of consecutive recorded samples (no gap o

**is hard braking commoner in town than on the highway**  (platform_v2)
* v6: Hard-braking event in city driving: 144 in 98.8 h (1.46 per hour); on the motorway / highway: 49 in 20.7 h (2.37 per hour) the last 60 days. Exact binomial p = 0.004 (a c
* v11: I read this as contrast two periods or two driving conditions (a quantity, or how often an event happened): hard-braking event (change, city), but I cannot answer that ki

**has my pack been warming up over time**  (sealed_v2)
* v6: I found no clear sustained change in battery temperature over the last 14 days; the biggest single move was on 12 Oct. / Largest day-to-day move in the unfavourable directi
* v11: I found no clear sustained change in battery temperature rise rate over the last 14 days; the biggest single move was on 5 Oct. / Largest day-to-day move: 5 Oct, 0.0 to 0.0

**how do i extend the battery life**  (sealed_v3)
* v6: Your battery: /  - Charge: 89%, about 272 km of range at your recent 180 Wh/km. /  - The battery health is 91% of its original capacity. /  - The pack temperature is 22 °C now,
* v11: Battery health is 91 % right now. / Based on 42,936 samples (per-minute) from vday.soh. / Data quality: battery soh was missing for 1 stretch(es), 265 min in total. / Evidence:

**acceleration between speeds 20 to 30 km per hour**  (user_v1)
* v6: Acceleration from 20 to 30 km/h: 6.78 s on average, 0.76 m/s2, across 332 valid run(s) in the last 7 days. Crossings are linearly interpolated; gaps, invalid states and n
* v11: I understood this as summary over a period: average, total, count, share of time, spread (for an event: how many, or how long it lasted in total): vehicle speed (share, t


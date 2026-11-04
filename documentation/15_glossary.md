# 15. Glossary

| Term | Meaning in this project |
|---|---|
| **ADAS** | advanced driver assistance system (lane assist, adaptive cruise, emergency braking) |
| **Adapter / LoRA** | a small set of extra trained weights added to the base model; stored as a separate file (35 MB); `frame2r12` is the shipped one |
| **Act** | the first word of a frame: the kind of answer wanted (now, stat, trend, why, ...) |
| **Ask-back** | the assistant asking "did you mean 1 or 2"; off by default, except for an exactly ambiguous shorthand |
| **Archive** | the lossless store of old one-second rows (`archive/*.vga`) |
| **AAOS** | Android Automotive OS; run here as a Cuttlefish guest on the Pi |
| **Audit hook** | a Python feature used to refuse every non-CAN socket and name lookup in the whole process |
| **Bank** | a text file of hand-written training questions with labels (`natural_*.txt`) |
| **BAM / RTS-CTS** | J1939 transport protocol modes for messages longer than 8 bytes |
| **Boundary** | the line between what the model decides and what code decides (`docs/BOUNDARY.md`); also the safety boundary between the collector and assistant domains |
| **CAN / CAN-FD** | Controller Area Network and its flexible-data-rate version, the main in-vehicle bus |
| **Calibration** | mapping the model's raw score to the share of questions actually read right at that score |
| **Canon** | the pre-parser that repairs spelling and shorthand (`vgpt/meaning/canon.py`) |
| **Car (program)** | the separate simulated vehicle that writes to the pipe; never imported by Vantage |
| **Cause engine** | the code that explains why a quantity changed or an event happened (`vgpt/cause/`) |
| **Concept** | the thing a question is about (a signal, event, derived quantity, spec, topic, or `unlisted`) |
| **Cond / vs / ent / c2 / ref** | frame slots: driving condition, second operand of a comparison, selected unit (trip, day...), second concept, back-reference |
| **Cuttlefish** | Google's virtual Android device; used for Android Automotive on the Pi |
| **Daily tier** | per-day means, max and min of every column, kept forever |
| **DBC** | a CAN database file describing messages and signals |
| **Decoder file / profile pack** | what describes one vehicle: bindings, DBCs, SOME/IP catalogue, DTC table, retention |
| **DoIP, UDS, ISO-TP, OBD-II** | diagnostics over IP; unified diagnostic services; the transport that carries them on CAN; the emissions diagnostics standard |
| **DPF / AdBlue** | diesel particulate filter; the urea fluid for diesel emissions |
| **DTC** | diagnostic trouble code |
| **Domain (collector / assistant)** | the two code groups in the safety design |
| **Emitter** | the code that walks a car's profile backwards to write raw bus traffic |
| **Evidence** | the object that carries an answer's claim, numbers, units, samples, tier, quality and gaps |
| **Exact frame** | all eight words equal the gold frame; "strict" = no equivalent frame accepted, "reviewed" = a person agreed equivalents |
| **Fingerprint** | scoring installed decoder files against the identifiers a link really carries |
| **Frame** | the model's output: eight words from the ontology (act concept op cond vs ent c2 ref) |
| **Frozen set / sealed set** | an evaluation file never edited (hashed) / a set written once, kept out of development and never trained on |
| **Gate** | a rule fixed before training for whether a new adapter replaces the old one |
| **Glossary** | one-line word meanings added to the prompt for rare words (`glossary.jsonl`) |
| **GGUF** | the llama.cpp model file format |
| **HPC** | the vehicle's high-performance computer (the upper end of the scaling range) |
| **Hybrid (honest)** | the product is learned meaning plus code; "model-only" was withdrawn |
| **Instant load** | loading a whole finished history into a car's store in one step, for testing |
| **J1939** | the CAN-based protocol of heavy vehicles |
| **Lake** | the table store for every signal beyond the nine core ones (one row per second per signal) |
| **Layers** | the switchable rule stages between frame and answer (`vgpt/meaning/layers.py`) |
| **Ledger** | trips and charging sessions derived once from one-second rows |
| **LIN, FlexRay, CANopen** | a low-cost body bus; a chassis bus; an industrial CAN protocol also in EV battery systems |
| **Literals** | typed values parsed from text (dates, times, numbers, units): the only text-pattern code in the meaning path |
| **Mapper** | the engine that decides which quantity a signal is, refusing when unsure |
| **Meaning frame** | see Frame |
| **Minute tier** | one-minute rows: duration-weighted mean, min and max, kept five years |
| **Ontology** | the data that says what a question can be about and what a well-formed frame looks like |
| **ONNX Runtime** | the runtime used for the int8 model on the Pi |
| **Op** | the operator word of a frame (mean, max, count, rise, ...) |
| **Pipe** | the one-way local channel from the Car to Vantage (named pipe or FIFO) |
| **pmin** | the lowest probability of any token the mask allowed: the raw confidence |
| **Pre-parser** | `canon` and `askme`: visible rule stages in front of the model |
| **Profile router** | decodes raw traffic for one vehicle using its profile pack |
| **Quantity** | a physical thing a car reports (speed, state of charge) |
| **Receiver** | Vantage's end of the pipe: sniff, fingerprint, decode, store |
| **RuleBudget** | a test that counts text-pattern calls per module and fails if one grows |
| **Runtime** | the one writer of the database: decode, checks, events, ledger |
| **Sniff** | classify a block of bytes from its structure alone |
| **SOME/IP** | scalable service-oriented middleware over IP in automotive Ethernet |
| **SoC / SoH** | state of charge / state of health of the battery |
| **Spec** | a fixed fact about the vehicle (seat count, capacity, tyre specification) |
| **Speculative decoding** | verifying several guessed tokens in one model pass, accepting only those the model would itself choose; lossless |
| **Strict / reviewed** | see Exact frame |
| **Tier** | a storage level (seconds, minutes, daily, records, archive); also a deployment profile (`edge2g`, `pi5`, `vm8`, `hpc`) |
| **Token mask** | at each step, only tokens that continue a valid ontology word are allowed |
| **Typed query** | the validated description of what to compute (`Query` in `query_ir.py`) |
| **Unlisted** | a concept meaning "a thing no signal in this car exists for" |
| **VHAL** | the Android Vehicle Hardware Abstraction Layer |
| **VIN** | vehicle identification number, used to tell two cars of one model apart |
| **Vantage** | the assistant program (and the project's name since 7 October) |
| **VehicleGPT** | the original project name and the case study's title for this task |
| **Wire record** | the pipe's unit: start marker, length, CRC-32, microsecond time, channel, bytes |
| **World** | a generated multi-month or multi-year history of one car, with planted faults and a ground-truth file |

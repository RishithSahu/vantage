# The cars of the test bench

45 car models from 34 makers, generated from the decoder files (`python tools_cars_catalogue.py`). Real-vehicle models use the open CAN databases of opendbc (MIT licence); the models of a *synthetic maker* (Aster, Kestrel, Meridian, Rhino, Nova, Volta, Prestige) are written for the test bench and say so. Every car's traffic is recognised from the bytes alone (`tests/test_pipe.py`).

Protocols and buses across the cars: logs (45), NMEA GNSS (45), CAN (44), UDS/OBD (44), camera (30), CAN-FD (4), SOME/IP (3), Android VHAL (3), LIN (2), DoIP (1), FlexRay (1), CANopen (1).

| Maker | Model | Year | Powertrain | Buses and protocols | Quantities | Pack id |
|---|---|---|---|---|---|---|
| Acura | RDX (2018) | 2018 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 37 | `acura_rdx_2018` |
| Alfa Romeo | Giulia (Giorgio platform) | 2018 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 36 | `alfa_romeo_giulia` |
| Aster Motors | Volt S | 2026 | bev | CAN, SOME/IP, Android VHAL, UDS/OBD, logs, NMEA GNSS, camera, CAN-FD | 43 | `aster_ev` |
| BMW | 3 Series (E90 320i) | 2009 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 29 | `bmw_3_series_e9x` |
| BYD | Atto 3 | 2023 | bev | CAN, UDS/OBD, logs, NMEA GNSS, camera | 29 | `byd_atto_3` |
| Cadillac | CT6 Plug-in (2017) | 2017 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 36 | `cadillac_ct6_phev` |
| Chevrolet | Bolt EV | 2021 | bev | CAN, UDS/OBD, logs, NMEA GNSS, camera | 19 | `chevrolet_bolt_ev` |
| Chrysler | Pacifica Hybrid (2017) | 2017 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 34 | `chrysler_pacifica_phev` |
| Ford | Mondeo 2.0 (CGEA 1.2, 2011) | 2011 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 31 | `ford_mondeo_2011` |
| GWM | Haval H6 PHEV (CAN-FD) | 2024 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera, CAN-FD | 31 | `gwm_haval_h6_phev` |
| Honda | Civic EX (2022) | 2022 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 39 | `honda_civic_2022` |
| Honda | Clarity Plug-in Hybrid (2018) | 2018 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 38 | `honda_clarity_phev` |
| Honda | Odyssey EX-L (2018) | 2018 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 38 | `honda_odyssey_2018` |
| Hongqi | HS5 | 2021 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 25 | `hongqi_hs5` |
| Hyundai | Elantra 2.0 (C-CAN) | 2016 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 40 | `hyundai_elantra_ccan` |
| Hyundai | Santa Fe (2007) | 2007 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 30 | `hyundai_santafe_2007` |
| Hyundai | i30 (2014) | 2014 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 34 | `hyundai_i30_2014` |
| Kestrel | C3 hatchback | 2024 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 32 | `kestrel_petrol` |
| Kia / Hyundai | Generic Kia / Hyundai CAN (2019) | 2019 | diesel | CAN, UDS/OBD, logs, NMEA GNSS, camera | 40 | `kia_hyundai_generic` |
| Luxgen | S5 (2015) | 2015 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 30 | `luxgen_s5_2015` |
| Mazda | CX-5 (2017) | 2017 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 37 | `mazda_cx5_2017` |
| Mazda | RX-8 (rotary) | 2008 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 30 | `mazda_rx8` |
| Mercedes-Benz | E 350 (W212) | 2010 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 33 | `mercedes_e350_w212` |
| Meridian | E5 SUV | 2026 | bev | CAN, SOME/IP, Android VHAL, UDS/OBD, logs, NMEA GNSS, camera, CAN-FD | 39 | `meridian_ev` |
| MG | ZS EV (HSC2 platform) | 2022 | bev | CAN, UDS/OBD, logs, NMEA GNSS, camera | 24 | `mg_zs_ev` |
| Nissan | Leaf (2018) | 2018 | bev | CAN, UDS/OBD, logs, NMEA GNSS, camera | 32 | `nissan_leaf_2018` |
| Nissan | X-Trail (2017) | 2017 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 35 | `nissan_xtrail_2017` |
| Nissan | Xterra 4.0 | 2011 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 27 | `nissan_xterra_2011` |
| Nova | N7 Ethernet | 2026 | bev | CAN, SOME/IP, Android VHAL, logs, NMEA GNSS, camera, LIN, CAN-FD, DoIP | 41 | `nova_ev` |
| Opel | Omega (2001) | 2001 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 30 | `opel_omega_2001` |
| Prestige | GT V8 FlexRay | 2026 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, LIN, FlexRay | 42 | `prestige_gt` |
| Ram | 1500 (DT, 2019) | 2019 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 35 | `ram_1500_dt` |
| Rhino | T4 light truck | 2025 | diesel | CAN, UDS/OBD, logs, NMEA GNSS | 31 | `rhino_diesel` |
| Rivian | R1T (primary actuator bus) | 2022 | bev | CAN, UDS/OBD, logs, NMEA GNSS, camera | 31 | `rivian_r1t` |
| Subaru | Crosstrek Hybrid (2020) | 2020 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 36 | `subaru_crosstrek_hybrid` |
| Subaru | Outback (2015) | 2015 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 36 | `subaru_outback_2015` |
| Tesla | Model 3 Long Range | 2020 | bev | CAN, UDS/OBD, logs, NMEA GNSS, camera | 31 | `tesla_model_3` |
| Toyota | Prius (XW30 hybrid) | 2010 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 36 | `toyota_prius_hybrid` |
| Toyota | RAV4 (TNGA, 2019) | 2019 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 39 | `toyota_rav4_tnga` |
| Toyota | iQ (2009) | 2009 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 30 | `toyota_iq_2009` |
| Unknown OEM (dummy) | PX-1 with a proprietary CAN | 2022 | petrol | CAN, UDS/OBD, logs, NMEA GNSS | 16 | `proprietary_demo` |
| Volkswagen | Golf (MQB evo) | 2020 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 34 | `vw_golf_mqbevo` |
| Volta | CityBus 12 m | 2026 | bev | UDS/OBD, logs, NMEA GNSS, CANopen | 23 | `volta_bus` |
| Volvo | V40 (2017) | 2017 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 33 | `volvo_v40_2017` |
| Volvo | V60 T5 | 2015 | petrol | CAN, UDS/OBD, logs, NMEA GNSS, camera | 32 | `volvo_v60_2015` |

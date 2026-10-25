import os, sys, glob, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.vehicle import onboard
D = os.path.join(ROOT, "tests", "fixtures", "opendbc")
cands = ["honda_civic_ex_2022_can_generated.dbc","honda_clarity_hybrid_2018_can_generated.dbc","honda_accord_2018_can_generated.dbc","honda_crv_ex_2017_can_generated.dbc","honda_odyssey_exl_2018_generated.dbc","acura_rdx_2018_can_generated.dbc","acura_ilx_2016_nidec.dbc",
 "ford_fusion_2018_pt.dbc","ford_lincoln_base_pt.dbc","ford_cgea1_2_ptcan_2011.dbc","chrysler_pacifica_2017_hybrid_generated.dbc","chrysler_ram_dt_generated.dbc","chrysler_cusw.dbc","subaru_outback_2015_generated.dbc","subaru_forester_2017_generated.dbc",
 "subaru_global_2017_generated.dbc","subaru_global_2020_hybrid_generated.dbc","mazda_3_2019.dbc","mazda_2017.dbc","mazda_rx8.dbc","vw_mqb.dbc","vw_mqb_2010.dbc","vw_mqbevo.dbc","vw_golf_mk4.dbc","psa_aee2010_r3.dbc","opel_omega_2001.dbc",
 "nissan_leaf_2018_generated.dbc","nissan_x_trail_2017_generated.dbc","hyundai_canfd.dbc","hyundai_i30_2014.dbc","hyundai_santafe_2007.dbc","hyundai_kia_generic.dbc","toyota_tnga_k_pt_generated.dbc","toyota_new_mc_pt_generated.dbc","toyota_nodsu_pt_generated.dbc",
 "toyota_iQ_2009_can.dbc","toyota_2017_ref_pt.dbc","volvo_v40_2017_pt.dbc","fca_giorgio.dbc","luxgen_s5_2015.dbc","rivian_primary_actuator.dbc","cadillac_ct6_powertrain.dbc","gm_global_a_powertrain_generated.dbc","tesla_powertrain.dbc","tesla_model3_vehicle.dbc","comma_body.dbc"]
for f in cands:
    p = os.path.join(D, f)
    if not os.path.exists(p): print(f, "missing"); continue
    try:
        spec, rep = onboard.draft("x", [(p, "can0", "canfd" in f)])
    except Exception as e: print(f, "ERR", str(e)[:60]); continue
    cs = collections.Counter(b["concept"] for b in spec["signals"] if not b["concept"].startswith("_"))
    print("%-48s mapped %3d review %3d unmapped %4d | concepts %2d: %s" % (f, len(rep["mapped"]), len(rep["review"]), len(rep["unmapped"]), len(cs), ",".join(sorted(cs))[:110]))

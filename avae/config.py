VIS_BET = False
VIS_LOS = False
VIS_ACC = False
VIS_REC = False
VIS_EMB = False
VIS_INT = False
VIS_DIS = False
VIS_POS = False
VIS_HIS = False

FREQ_ACC = 10
FREQ_REC = 10
FREQ_EMB = 10
FREQ_INT = 10
FREQ_DIS = 10
FREQ_POS = 10

FREQ_EVAL = 10
FREQ_STA = 10

DEFAULT_RUN_CONFIGS = {
    "limit": None,
    "split": 20,
    "depth": 3,
    "channels": 64,
    "latent_dims": 8,
    "pose_dims": 1,
    "no_val_drop": True,
    "epochs": 20,
    "batch": 128,
    "learning": 0.001,
    "gpu": True,
    "beta": 1.0,
    "gamma": 2,
    "loss_fn": "MSE",
    "beta_min": 0.0,
    "beta_cycle": 4,
    "beta_ratio": 0.5,
    "kl_weight_method": "flat",
    "freq_eval": FREQ_EVAL,
    "freq_sta": FREQ_STA,
    "freq_emb": FREQ_EMB,
    "freq_rec": FREQ_REC,
    "freq_int": FREQ_INT,
    "freq_dis": FREQ_DIS,
    "freq_pos": FREQ_POS,
    "freq_acc": FREQ_ACC,
    "freq_all": False,
    "eval": False,
    "dynamic": False,
    "vis_emb": VIS_EMB,
    "vis_rec": VIS_REC,
    "vis_los": VIS_LOS,
    "vis_bet": VIS_BET,
    "vis_int": VIS_INT,
    "vis_dis": VIS_DIS,
    "vis_pos": VIS_POS,
    "vis_acc": VIS_ACC,
    "vis_his": VIS_HIS,
    "vis_all": False,
    "model": "a",
}

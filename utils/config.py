import yaml
from easydict import EasyDict as edict

config = edict()

config.GPUS = 0, 1
config.WORKERS = 16
config.LOG_DIR = ""
config.TENSORBOARD_DIR = "./tensorboard"
config.MODEL_DIR = ""
config.RESULT_DIR = ""
config.VERBOSE = False
config.TAG = ""

# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# DATASET related params
config.DATASET = edict()
config.DATASET.NAME = ""
config.DATASET.TRAIN = ""
config.DATASET.VAL = ""
config.DATASET.TEST = ""
config.DATASET.ANNOTATION_PATH = ""
config.DATASET.ANCHOR_PATH = ""
config.DATASET.VIDEO_CACHE = ""
config.DATASET.BLOCK_SIZE = ""
config.DATASET.NO_VAL = False

# LOSS related params
config.LOSS = edict()
config.LOSS.NA = 5
config.LOSS.ANCHOR_T = 4.0
config.LOSS.ALPHA = 1
config.LOSS.BETA = 5
config.LOSS.SCALE = 0.1
config.LOSS.GAMMA = 0.3

# TRAIN related params
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 20
config.TRAIN.MAX_EPOCH = 20
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.STEP_ACCUMULATE = 1
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False
config.TRAIN.FP16 = False
config.TRAIN.FINE_TUNE = False

# TEST related params
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 0.4
config.TEST.INTERVAL = 1
config.TEST.BATCH_SIZE = 1

# MODEL related params
config.MODEL = edict()
config.MODEL.NAME = ""
config.MODEL.CHECKPOINT = ""  # The checkpoint for the best performance
config.MODEL.TEXT_ATTN_LAYER = 3
config.MODEL.CRO_ATTN_LAYER = 2
config.MODEL.CON_ALPHA = 0.2
config.MODEL.CON_BETA = 0.3
config.MODEL.MARGIN = 0.5


def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if isinstance(v, dict):
                _update_dict(cfg[k], v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

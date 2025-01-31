from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.data_path = ""  # Directory where datasets are stored
_C.descriptors_fname = ""
_C.num_class = 10
_C.class_names = []
_C.model = ""
_C.backbone = ""
_C.resolution = 224
_C.stride = 16

_C.seed = None
_C.deterministic = False
_C.gpuid = None
_C.num_workers = 8
_C.prec = "fp16"  # fp16, fp32, amp

_C.stage = 0
_C.frozen = None
_C.pretrained = True
_C.epochs = 10
_C.batch_size = 256


_C.noise_mode = ""
_C.noise_ratio = 0.0
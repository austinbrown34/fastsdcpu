from os import environ, cpu_count

cpu_cores = cpu_count()
cpus = cpu_cores // 2 if cpu_cores else 0
APP_VERSION = "v1.0.0 beta 120"
LCM_DEFAULT_MODEL = "stabilityai/sd-turbo"
LCM_DEFAULT_MODEL_OPENVINO = "rupeshs/sd-turbo-openvino"
APP_NAME = "FastSD CPU"
APP_SETTINGS_FILE = "settings.yaml"
RESULTS_DIRECTORY = environ.get("RESULTS_DIR", "results")
CONFIG_DIRECTORY = environ.get("CONFIGS_DIR", "configs")
DEVICE = environ.get("DEVICE", "cpu")
SD_MODELS_FILE = "stable-diffusion-models.txt"
LCM_LORA_MODELS_FILE = "lcm-lora-models.txt"
OPENVINO_LCM_MODELS_FILE = "openvino-lcm-models.txt"
TAESD_MODEL = "madebyollin/taesd"
TAESDXL_MODEL = "madebyollin/taesdxl"
TAESD_MODEL_OPENVINO = "deinferno/taesd-openvino"
LCM_MODELS_FILE = "lcm-models.txt"
TAESDXL_MODEL_OPENVINO = "rupeshs/taesdxl-openvino"
LORA_DIRECTORY = environ.get("LORA_DIR", "lora_models")
CONTROLNET_DIRECTORY = environ.get("CONTROLNET_DIR", "controlnet_models")
MODELS_DIRECTORY = environ.get("MODELS_DIR", "models")
GGUF_THREADS = environ.get("GGUF_THREADS", cpus)
TAEF1_MODEL_OPENVINO = "rupeshs/taef1-openvino"
S3_BUCKET_NAME = environ.get("S3_BUCKET_NAME", None)
S3_FOLDER_PREFIX = environ.get("S3_FOLDER_PREFIX", None)
S3_REGION = environ.get("S3_REGION", None)
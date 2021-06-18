import logging.config
import os


# ========================= logger ================================
log_conf = './logger.conf'
logging.config.fileConfig(log_conf)
logger = logging.getLogger('GraphEvolveDroid')


# ============================ files ==============================
script_root = os.path.dirname(__file__) # root_path.
data_dir = ""

# 结果保存的地址
model_path = os.path.join(script_root, "save_models")
embed_path = os.path.join(script_root, "visualation/embeddings")
report_file = os.path.join(script_root,"save_models","reports.csv")

cuda_device_id = 0
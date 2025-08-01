import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_env(var_name):
    return os.getenv(var_name)









 
import yaml
from easydict import EasyDict

def load_config(fname):
    with open(fname,'r') as f:
        try:
            return EasyDict(yaml.safe_load(f))
        except yaml.YAMLError as exc:
            print(exc)


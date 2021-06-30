import yaml

def load_config(fname):
    with open(fname,'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


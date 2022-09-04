import yaml
import os

class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.yaml_read = f.read()
        self.config_dict = yaml.load(self.yaml_read, Loader=yaml.FullLoader)

    def __getattr__(self, name):
        if self.config_dict.get(name) is not None:
            return self.config_dict[name]
        else:
            return None

    def print(self):
        print("-------------------------------------------------")
        print("Model configurations: ")
        print(self.yaml_read)
        print("-------------------------------------------------")
        print(" ")

if __name__ == '__main__':
    config = Config('../config/config.yml')
    config.print()
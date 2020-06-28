import os
from argparse import ArgumentParser
from configparser import ConfigParser
import torch


class Config(ConfigParser):
    def __init__(self, config_file):
        raw_config = ConfigParser()
        raw_config.read(config_file)
        self.cast_values(raw_config)

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                val = None
                if type(value) is str and value.startswith("[") and value.endswith("]"):
                    val = eval(value)
                    setattr(self, key, val)
                    continue
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(raw_config[section], attr)(key)
                        break
                    except:
                        val = value
                setattr(self, key, val)


def parse_config():
    parser = ArgumentParser(
        description="Text CNN")
    parser.add_argument('--config', dest='config', default='CONFIG')
    # action='store_true') # for debug
    parser.add_argument('--train', dest="train",action="store_true", default=False)
    # action='store_true') # for debug
    parser.add_argument('--test', dest="test", default=True)
    parser.add_argument('-v', '--verbose', default=False)

    args = parser.parse_args()
    print("args: ",args)        #不同的地方1，它输出了一下
    config = Config(args.config)

    config.train = args.train
    config.test = args.test
    config.verbose = args.verbose
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    print(torch.cuda.is_available())#看看cuda是否可用
    return config

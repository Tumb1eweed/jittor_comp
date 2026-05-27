import os
import numpy as np
import random
import time
import logging
import logging.handlers


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)


def str_list(argstr):
    return list(argstr.split(','))


def get_log_dir_name_tblogger(name=''):
    log_dir_name = name + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    return log_dir_name


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_hyperparams(writer, log_dir, args):
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    with open(os.path.join(log_dir, 'hparams.csv'), 'w') as csvf:
        csvf.write('key,value\n')
        for k, v in vars_args.items():
            csvf.write('%s,%s\n' % (k, v))

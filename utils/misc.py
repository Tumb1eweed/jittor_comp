import numpy as np
import random
import time


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)


def str_list(argstr):
    return list(argstr.split(','))


def get_log_dir_name_tblogger(name=''):
    log_dir_name = name + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    return log_dir_name

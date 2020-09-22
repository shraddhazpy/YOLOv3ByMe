import tensorflow as tf
import numpy as np
import argparse

import utils,create_models




if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='No of classes to be entered')
    parser.add_argument('n_classes',type=int)
    args= parser.parse_args()




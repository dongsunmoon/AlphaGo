import numpy as np
import re
import os
import cPickle as pickle

DATA_DIR = 'dataset/dataset'
PREFIX = 'data-'
POSTFIX = '.sgf'

# All Kgs data is too large to upload onto main memory.
# This module is to serve Kgs files as network inputs.
class KgsInput:
    def __init__(self):
        self.path = ''
        self.fileCounts = 0
    def count_data_files(self):
        for root, dirs, files in os.walk(DATA_DIR):
            for fname in files:
                print fname
                if fname.endswith('.pkl'):
                    fullpath = os.path.join(root, fname)
                    self.fileCounts += 1
        print "Total files: %s" % (self.fileCounts)

train = KgsInput()
test = KgsInput()
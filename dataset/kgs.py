import numpy as np
import re
import os
import cPickle as pickle


# All Kgs data is too large to upload onto main memory.
# This module is to serve Kgs files as network inputs.
class KgsInput:
    def __init__(self):
        self.path = ''
    def tmp_1(self):
        print "-----test-----"


train = KgsInput()
test = KgsInput()
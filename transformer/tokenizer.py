import torch
import torch.nn as nn
import math
import numpy as np


class Tokenizer():
    def __init__(self):
        pass

    def tokenize(self, text):
        text = text.lower()
        text = text.replace('?', '')
        text = text.replace("'", '')
        return text.split()    


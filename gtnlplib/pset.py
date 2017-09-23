import gtnlplib.parsing as parsing
import gtnlplib.data_tools as data_tools
import gtnlplib.constants as consts
import gtnlplib.evaluation as evaluation
import gtnlplib.utils as utils
import gtnlplib.feat_extractors as feat_extractors
import gtnlplib.neural_net as neural_net

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag

from collections import defaultdict

# Read in the dataset
dataset = data_tools.Dataset(consts.TRAIN_FILE, consts.DEV_FILE, consts.TEST_FILE)

# Assign each word a unique index, including the two special tokens
word_to_ix = { word: i for i, word in enumerate(dataset.vocab) }

# Some constants to keep around
LSTM_NUM_LAYERS = 1
TEST_EMBEDDING_DIM = 5
WORD_EMBEDDING_DIM = 64
STACK_EMBEDDING_DIM = 100
NUM_FEATURES = 3

# Hyperparameters
ETA_0 = 0.01
DROPOUT = 0.0

def make_dummy_parser_state(sentence):
    dummy_embeds = [ w + "-EMBEDDING" for w in sentence ] + [consts.END_OF_INPUT_TOK + "-EMBEDDING"]
    return parsing.ParserState(sentence + [consts.END_OF_INPUT_TOK], dummy_embeds, utils.DummyCombiner())


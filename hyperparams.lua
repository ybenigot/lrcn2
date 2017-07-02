

local M = {}

-- hyperparameters
SIZE=224 -- size of resnet input images 
CNN_OUTPUT_SIZE = 1000 -- sie of resnet output
MAX_SENTENCE_LENGTH = 20
BATCH_SIZE = 192
LEARNING_RATE = 0.1
SAMPLE_SAVE_COUNT = 10000

LR_DECAY_EPOCH_COUNT = 10
MAX_EPOCH = 200

return M

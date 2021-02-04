import transformers
import tokenizers
import os
import torch

TRAIN_DATA = 'train.csv'
TEST_DATA = 'test.csv'

WORKING_DIR = '../'
INPUT_DIR = '../input'
MODELS_DIR = '../models'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
LR = 3e-5
EPOCHS = 5
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "../models/model.bin"
TRAINING_FILE = "../input/train.csv"
OPTIMIZER = transformers.AdamW
SCHEDULER = transformers.get_linear_schedule_with_warmup
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
BERT_MODEL = transformers.BertModel.from_pretrained(BERT_PATH, return_dict=False)
# TOKENIZER = tokenizers.BertWordPieceTokenizer(
#     os.path.join(BERT_PATH, 'vocab.txt'),
#     do_lower_case=True
# )
import os
from transformers import AutoTokenizer
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME

DEVICE = "cpu"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WARMUP_PROPORTION = 0.1
EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 1

LABELS = ['0', '1']
NUM_LABELs = len(LABELS)
SEED = 42

DATA_DIR = "./data/vn_dataset_version"
DATASET_FILENAME = 'vn_all_triple.csv'
DATASET_FILE_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

TRAIN_FILENAME = 'train_with_corr.csv'
TRAIN_FILE_PATH = os.path.join(DATA_DIR, TRAIN_FILENAME)
TEST_FILENAME = 'test_with_corr.csv'
TEST_FILE_PATH = os.path.join(DATA_DIR, TEST_FILENAME)
VAL_FILENAME = 'val_with_corr.csv'
VAL_FILE_PATH = os.path.join(DATA_DIR, VAL_FILENAME)

OUTPUT_DIR = './output_model'
MODEL_PATH = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
CONFIG_PATH = os.path.join(OUTPUT_DIR, CONFIG_NAME)

BASE_MODEL_PATH = "./phobert-base"
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, do_lower_case=True)
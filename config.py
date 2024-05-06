DATA_DIR = 'assets/dataset/wave_buoy_data_halifax.csv'

EMBED_DIM = 6
MAX_HORIZON = 24
HORIZON_LIST = list(range(1, MAX_HORIZON + 1))
TARGET = 'VCAR'
THRESHOLD_PERCENTILE = 0.99
CV_N_FOLDS = 5
TRAIN_SIZE = 0.6
TEST_SIZE = 0.2

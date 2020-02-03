import os


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

RAW_DATA = os.path.join(ROOT_PATH, "data", "raw")
INTERIM_DATA = os.path.join(ROOT_PATH, "data", "interim")
OUTPUT_DATA = os.path.join(ROOT_PATH, "data", "output")

# File names
TEST_DATA = os.path.join(RAW_DATA, "test.csv")
TRAIN_DATA = os.path.join(RAW_DATA, "train.csv")

NORMALIZED_TEST = os.path.join(INTERIM_DATA, "normalized_test.csv")
NORMALIZED_TRAIN = os.path.join(INTERIM_DATA, "normalized_train.csv")
EMBEDDING_MATRIX = os.path.join(INTERIM_DATA, "embedding_matrix.csv")

MODELS_PATH = create_folder(os.path.join(ROOT_PATH, "src", "models", "saved_models"))

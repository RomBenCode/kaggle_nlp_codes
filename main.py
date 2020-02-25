from src.preprocess import preprocess
from src.models.embedding_only import train_embedding_only_model
# TODO: logging
# TODO: tests


def main(bool_dict):
    if bool_dict["preprocess"]:
        preprocess()

    if bool_dict["train_embedding_model"]:
        train_embedding_only_model()


if __name__ == "__main__":
    bool_dict = {
        "preprocess": True,
        "train_embedding_model": True
    }

    main(bool_dict)

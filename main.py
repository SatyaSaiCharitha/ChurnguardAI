from src.data_preprocessing import (
    load_data,
    clean_data,
    encode_target,
    encode_features
)
from src.model_training import train_model
from src.config import DATA_PATH


def main():
    # Load
    df = load_data(DATA_PATH)

    # Clean
    df = clean_data(df)

    # Encode target
    df = encode_target(df)

    # Encode features
    df = encode_features(df)

    # Train model
    model = train_model(df)


if __name__ == "__main__":
    main()
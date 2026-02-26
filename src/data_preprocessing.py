import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID (not useful for modeling)
    df.drop("customerID", axis=1, inplace=True)

    return df


def encode_target(df):
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def encode_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df
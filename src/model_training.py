from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd
import matplotlib.pyplot as plt



def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Get feature importance
    importances = model.feature_importances_

    # Create dataframe
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    })

    # Sort values
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Print top 10
    top10 = importance_df.head(10)

    plt.figure()
    plt.barh(top10["Feature"], top10["Importance"])
    plt.xlabel("Importance Score")
    plt.title("Top 10 Churn Driving Features")
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()
    plt.show()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.35).astype(int)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # Save model
    joblib.dump(model, "models/churn_model.pkl")

    # Save feature column order
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    print("Model and feature columns saved successfully")

    return model
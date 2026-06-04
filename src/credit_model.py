import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)



def main():
    # Resolve data path relative to project root
    data_path = Path(__file__).resolve().parents[1] / "Data" / "credit_data.csv"

    if not data_path.exists():
        print(f"Data file not found at {data_path}. Creating a small sample dataset.")
        # Ensure parent directory exists
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample = '''Age,Income,Education,Marital_Status,Credit_Score
25,50000,Bachelors,Single,1
45,80000,Masters,Married,1
35,62000,Bachelors,Married,1
23,32000,Highschool,Single,0
52,110000,PhD,Married,1
40,76000,Bachelors,Divorced,0
29,54000,Bachelors,Single,1
31,58000,Masters,Married,1
38,67000,Bachelors,Married,1
27,48000,Highschool,Single,0
50,90000,Masters,Married,1
33,60000,Bachelors,Single,1
26,45000,Highschool,Single,0
48,95000,PhD,Married,1
41,82000,Masters,Married,1
30,57000,Bachelors,Single,0
36,65000,Bachelors,Married,1
28,52000,Highschool,Single,0
47,88000,Masters,Married,1
55,120000,PhD,Married,1
'''
        data_path.write_text(sample)

    try:
        df = pd.read_csv(data_path)
    except pd.errors.EmptyDataError:
        print(f"Data file {data_path} is empty.")
        sys.exit(1)

    if df.empty:
        print(f"Data file {data_path} contains no rows.")
        sys.exit(1)

    print("Data preview:")
    print(df.head())

    # Remove missing values
    df = df.dropna()

    # Convert categorical columns
    label_encoder = LabelEncoder()

    for col in df.select_dtypes(include='object').columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Target column
    target = "Credit_Score"

    if target not in df.columns:
        print(f"Target column '{target}' not found in data.")
        sys.exit(1)

    X = df.drop(target, axis=1)
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # ROC-AUC (if available) and Confusion Matrix plot
    try:
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, prob)
                print("ROC-AUC:", auc)
            except Exception:
                print("Could not compute ROC-AUC (possibly single-class y_test).")
    except Exception:
        print("Model does not support predict_proba or ROC-AUC computation failed.")

    try:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title("Confusion Matrix")
        plt.show()
    except Exception as e:
        print("Could not plot confusion matrix:", e)


if __name__ == '__main__':
    main()
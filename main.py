import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df, target):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")

    return accuracy, precision, recall, f1


def cross_validate_model(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    print(f"Cross-validated F1 Score (Macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return cv_scores


if __name__ == "__main__":
    data_path = "data/liver_cirrhosis.csv"
    target = 'Stage'

    df = load_data(data_path)

    X, y = preprocess_data(df, target)

    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    cross_validate_model(model, X, y)

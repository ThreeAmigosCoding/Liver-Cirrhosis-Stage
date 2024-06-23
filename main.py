import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
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


def plot_correlation_matrix(df):
    correlation_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={'shrink': .8})
    plt.title('Matrica korelacije')
    plt.show()


def plot_distributions(df):
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df.hist(bins=30, figsize=(20, 15))
    plt.tight_layout()
    plt.show()


def plot_boxplots(df):
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(20, 15))
    for i, column in enumerate(numeric_df.columns, 1):
        plt.subplot(4, 5, i)
        sns.boxplot(y=numeric_df[column])
    plt.tight_layout()
    plt.show()


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def apply_pca(X_train, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")

    return accuracy, precision_macro, recall_macro, f1_macro, f1_micro


def cross_validate_model(model, X, y, cv=5):
    cv_scores_macro = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    cv_scores_micro = cross_val_score(model, X, y, cv=cv, scoring='f1_micro')
    print(f"Cross-validated F1 Score (Macro): {cv_scores_macro.mean():.4f} ± {cv_scores_macro.std():.4f}")
    print(f"Cross-validated F1 Score (Micro): {cv_scores_micro.mean():.4f} ± {cv_scores_micro.std():.4f}")

    return cv_scores_macro, cv_scores_micro


if __name__ == "__main__":
    data_path = "data/liver_cirrhosis.csv"
    target = 'Stage'

    df = load_data(data_path)
    print(df['Stage'].value_counts())

    X, y = preprocess_data(df, target)

    plot_correlation_matrix(df)

    plot_boxplots(df)

    plot_distributions(df)

    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)

    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    cross_validate_model(model, X, y)

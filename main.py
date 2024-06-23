import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
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

    # df = remove_outliers_iqr(df)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def remove_outliers_iqr(df):
    columns_to_check = ['Cholesterol', 'Tryglicerides', 'Platelets', 'Albumin', 'Copper', 'Prothrombin', 'Alk_Phos',
                        'SGOT', 'Bilirubin']
    for column in columns_to_check:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        filter = (df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))
        df = df.loc[filter]
    return df


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
    precision_macro = precision_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='micro')
    f1_micro = f1_score(y_test, y_pred, average='micro')

    print(f"Accuracy: {accuracy:.8f}")
    print(f"Precision: {precision_macro:.8f}")
    print(f"Recall: {recall_macro:.8f}")
    print(f"Micro F1 Score: {f1_micro:.8f}")

    return accuracy, precision_macro, recall_macro, f1_micro


if __name__ == "__main__":
    data_path = "data/liver_cirrhosis.csv"
    target = 'Stage'

    df = load_data(data_path)
    print(df['Stage'].value_counts())

    X, y = preprocess_data(df, target)

    # region Explorative analysis
    plot_correlation_matrix(df)

    plot_distributions(df)

    plot_boxplots(df)
    # endregion

    # region Training and evaluation
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)

    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_micro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_test, y_test)
    # endregion

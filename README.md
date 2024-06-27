# Liver Cirrhosis Stage

Machine learning script for liver cirrhosis stage prediction. Includes detailed preprocessing and 4 different models trained with various parameter values. You can find more about this project from [project report](https://github.com/ThreeAmigosCoding/Liver-Cirrhosis-Stage/blob/main/Liver%20cirrhosis%20stage%20prediction.pdf).

## Authors
- [Miloš Čuturić](https://github.com/cuturic01)
- [Luka Đorđević](https://github.com/lukaDjordjevic01)
- [Marko Janošević](https://github.com/janosevicsm)

#
## Prerequisites
- Python 3.x installed on your machine.
- Virtual environment (`venv`) to manage project dependencies.

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ThreeAmigosCoding/Liver-Cirrhosis-Stage
    cd Liver-Cirrhosis-Stage
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m virtualenv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Script

1. **Ensure you are in the project directory and the virtual environment is activated**.

2. **Run the main script**:
    ```bash
    python main.py
    ```
3. **If everything is correct after the model training you should get the information about the performance of all 4 models and the best parameters for each model. Output example**:
    ```bash
    Evaluating RandomForest...
    Best parameters for RandomForest: {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 200}
    Accuracy: 0.95520000
    Precision: 0.95520000
    Recall: 0.95520000
    Micro F1 Score: 0.95520000
    ```


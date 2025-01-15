import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pytest

def load_data():
    file_path = "data/stock_data.csv"
    stock_df = pd.read_csv(file_path)
    stock_df['sp500_up'] = (stock_df['sp500'] > stock_df['prev_day']).astype(int)
    return stock_df

# Evaluate feature combinations, we want to know which performs best
def evaluate_features(stock_df, features):
    X = stock_df[features]
    y = stock_df['sp500_up']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Pytest method to find and print the best feature combination
def test_best_feature_combination():
    stock_df = load_data()
    all_features = ['joblessness', 'vix', 'epu', 'us3m', 'ads', 'prev_day', 'sp500_volume', 'djia_volume', 'GPRD']
    best_score = 0
    best_features = []

    for i in range(1, len(all_features) + 1):
        for combo in itertools.combinations(all_features, i):
            score = evaluate_features(stock_df, list(combo))
            if score > best_score:
                best_score = score
                best_features = combo
    
    print(f"Best feature combination: {best_features} with Accuracy: {best_score}")

# Pytest method to perform Grid Search for hyperparameter tuning
# Warning! This does take some time to run. You can reduce the param_grid to make it faster
def test_hyperparameter_tuning():
    stock_df = load_data()
    best_features = ['joblessness', 'vix', 'epu', 'us3m', 'prev_day']
    X = stock_df[best_features]
    y = stock_df['sp500_up']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Reduce these params to reduce test runtime
    param_grid = {
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [100, 300, 500],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

    model = XGBClassifier(eval_metric='logloss')
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='accuracy', cv=tscv, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print("Best Hyperparameters:")
    print(grid_search.best_params_)
    print(f"Accuracy with best hyperparameters: {best_accuracy}")

# Main method runs XGBoost with the best feature set
def main():
    stock_df = load_data()
    best_features = ['joblessness', 'vix', 'epu', 'us3m', 'prev_day']
    X = stock_df[best_features]
    y = stock_df['sp500_up']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Train the model with optimized hyperparameters
    model = XGBClassifier(
        eval_metric='logloss',
        colsample_bytree=1.0,
        gamma=0,
        learning_rate=0.3,
        max_depth=7,
        n_estimators=500,
        reg_alpha=0,
        reg_lambda=2,
        subsample=0.8
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with features {best_features}: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()

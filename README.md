# ml-sp500-predictor

A simple attempt to apply machine learning techniques to historical S&P500 data to predict market movements.

## The dataset
Taken from https://www.kaggle.com/datasets/shiveshprakash/34-year-daily-stock-data, 34 years of stock market data.

## The approach
I started out with a linear regression model but switched to XGBoost. XGBoost should generally be more suitable for stock market predictions
(as the data is highly non-linear).

A basic setup yeilded an accuracy of around .55. A few changes got this up to .595

### Optimising features
The dataset has 13 columns, so plenty of features to choose from. I wanted to see which combination of features work best, so wrote a quick test
to run through the combinations.

```
pytest app.py
```

```Best feature combination: ('joblessness', 'vix', 'epu', 'us3m', 'prev_day') with Accuracy: 0.5952311718522827```

The above features are therefore used by the main method. I also found that a `test_size` of 0.4 performed best.

I also tried to apply a class weighting to help the model more accurately predict the minority case (down days), but didnt' make
a significant difference.

### Hyperparameter tuning

With default values, we get an accuracy of 0.595. I wrote a `test_hyperparameter_tuning()` to evaluate the model with a range of hyperparameters,
using `GridSearchCV`. The best parameter set brought up the accuracy to 0.606.

## Running the model

Ensure dependencies are installed:

```
pip install pandas
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install seaborn
pip install pytest

brew install libomp
```

```
python app.py
```

## TODO - Next steps

We can probably improve accuracy by applying further techniques:

- Feature engineering (e.g. moving averages / trends)
- Stack models! (e.g add Random forest or Logistic regression)

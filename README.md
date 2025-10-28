# Evaluating Machine Learning Models Against the Black-Scholes Formula for Option Pricing

### Overveiw

The Black-Scholes model is the baseline for option prices, however it is built on assumptions that can lead to inaccuracies when predicting real market values. This project compares this traditional method with modern Machine Learning models (Neural Network and XGBoost) to evaluate how well they capture real-world option prices. By training the models on data from Yahoo Finance, the aim is to investigate whether machine learning models can outperform the Black-Scholes model for predicting option prices.

### Method

Data is collected and organised using Pandas DataFrames, with feature scaling applied before training. The three models were then implemented and compared against the real market prices:

The Black-Scholes formula varies slightly depending on whether it calculates 'call' or 'put' options, and provides predictions based on input features such as; strike price, underlying price, volatility, time until expiry, and risk-free interest rate. The Neural Network architecture consists of 3 layers with ReLU activation functions. The Adam optimizer was used for efficient gradient descent. The Mean Squared Error loss function was used to minimise the error between predicted and real values, and the data was also scaled to prevent features with larger values dominating. The XGBoost model utilised the same input features as the Neural Network, with a small learning rate and large ensemble of trees - tuned to prevent overfitting. Each model's predictions were plotted against the real market prices, allowing each model's fit to be visualised clearly. 

Each model was also analysed statistically using three common metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared Value (R2), the results stored and later plotted against eachother to compare models.

### Results

Both the Neural Network and the XGBoost achieve approximately the same R2 score of 0.9813. This is a 3.66% difference with the Black-Scholes model. Since R2 values tend to be above 90% for models, an increase of 3.66% is a noticeable difference, highlighting the accuracy of machine learning methods without taking away from the BS model. 

Both NN and XGBoost exhibited similar Root Mean Squared Error and Mean Average Error values, with XGBoost outperforming slightly. Both machine learning methods had an approximate 38.1% improvement in RMSE compared to the Black-Scholes model. The Neural Network and XGBoost model also displayed a 42.31% an 55.00% improvement in MAE results.

These results show both ML models outperform Black-Scholes across all companies. Discussed in detail in the notebook.

### Libraries and Tools 

Python
Yahoo Finance - Data collection.
NumPy, Pandas, Matplotlib - Data processing and visualisation.
Scikit-Learn, XGBoost, Tensorflow (Keras) - Model implementation.


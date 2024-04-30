# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Temperature.csv')
data['date'] = pd.to_datetime(data['date'])

plt.plot(data['date'], data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.show()

def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['temp'])

plot_acf(data['temp'])
plt.show()
plot_pacf(data['temp'])
plt.show()

sarima_model = SARIMAX(data['temp'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['temp'][:train_size], data['temp'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```

### OUTPUT:
![image](https://github.com/Vivekreddy8360/TSA_EXP10/assets/94525701/9639a5e2-638d-4595-98db-57fc82fee031)
![image](https://github.com/Vivekreddy8360/TSA_EXP10/assets/94525701/607d8e22-a45d-4978-8c40-3a9b76f7d08b)
![image](https://github.com/Vivekreddy8360/TSA_EXP10/assets/94525701/3a1bf105-cb24-43d6-8bfd-8352de32f457)
![image](https://github.com/Vivekreddy8360/TSA_EXP10/assets/94525701/9e0fb666-2576-4c2f-b58a-b4dd036ae14d)


### RESULT:
Thus the program run successfully based on the SARIMA model.

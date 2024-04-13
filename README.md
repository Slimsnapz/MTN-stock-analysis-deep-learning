# MTN-stock-analysis-deep-learning
Project
#### -- Project Status: [Completed]

## Project Description
### Overview
it only makes sense to pick up a project related to the topic. I decided to explore stock price prediction ideas since it is perhaps the simplest form of a time series project out there. I thought it would be best to analyze a local public company close to home, namely MTN Nigeria. MTN Nigeria is a company that primarily offers cellular network access and ICT solutions. It's services portfolio include mobile voice, text services, Internet services, such as video calling, data services and Internet browsing, mobile Internet and mobile WiFi services.. In this project, I will be using MTN stock data from 2020 to 2024 to apply time series analysis techniques as well as creating a LSTM model to learn and forecast the data. The MTN Stock Analysis project aims to leverage machine learning techniques, particularly deep learning with Long Short-Term Memory (LSTM) networks, to analyze and forecast the stock prices of MTN Nigeria, a leading telecommunications provider in Nigeria. This project integrates various methodologies, including time series data preprocessing, visualization, and LSTM modeling, to derive insights and predictions about MTN Nigeria's stock performance.

### Objectives
Analyze historical stock data to identify patterns and trends.
Develop and train LSTM models for stock price forecasting.
Evaluate model performance and reliability for investment decision-making.

### Methods Used
#### Time Series Data Preprocessing
* Data cleaning and transformation to ensure quality and consistency.
* Feature engineering to extract relevant information for modeling.
#### Time Series Data Visualization
* Exploratory data analysis to visualize trends, seasonality, and anomalies.
* Interactive plots to provide intuitive insights into the stock's historical performance.
#### Long Short-Term Memory (LSTM)
* Development of LSTM models to capture temporal dependencies in the data.
* Training and optimization of the LSTM networks for accurate forecasting.
 

### Technologies
* Python: Core programming language for data analysis and machine learning.
* scikit-learn (sklearn): Machine learning library for data preprocessing and model evaluation.
* matplotlib & seaborn: Visualization libraries for data exploration and result presentation.
* pandas: Data manipulation and analysis tool.
* numpy: Mathematical computations and array operations.
* statsmodels: Statistical models and tests for time series analysis.

### Data
The MTN stocks data was downloaded from [Nigeria investing website](https://ng.investing.com/equities/mtn-nigeria-com-historical-data) and 1057 Data points were obtained comprising of stock details from 1 january 2020 to 2024. 


## Preprocessing
The data from investment Nigeria was good. However, the data column are of diffrent data types which can prove problematic during analysis. I decided to convert the data in each column to the same data type. The stocks data also comes in IDR, which has quite the unconventional use of commas and decimal points. The function below will parse or convert IDR values to recognizable integers that can easily be processed. 
```python
#creating a function that removes the % sign from the figures
def idr_parser(cur_str):
  try:
    cur_str = str(cur_str)
  except:
    pass
  cur_str = re.sub('[%]', '', cur_str)
  return cur_str
```
## Visualization
For the data visualization, I constructed distribution plots of each feature and also explored the full historical data in the 4-year span. Moving averages plots (on various window lengths) and seasonal decomposition plots are also constructed since it is time-series data after all. All of the plots are displayed in the [notebook](https://github.com/Slimsnapz/MTN-stock-analysis-deep-learning/blob/main/MTN_STOCK_tensorflow_LSTM.ipynb).

## LSTMs
Recurrent Neural Networks are undoubtedly the most suitable type of neural network to handle time-series data due to their ability to retain a memory of past inputs. LSTMs are essentially just a better version of vanilla RNNs due to the cell states that allows information to be removed or added at will. Essentially, LSTMs are far better at retaining memory and are insensitive to gap length, which is quite important considering the many data points we will input. Stacked LSTMs, on the other hand, increase the depth by adding multiple hidden LSTM layers in the architecture, which allows for better learning capabilities and accuracy. After experimenting with both vanilla LSTMs and stacked LSTMs on the stocks data, stacked LSTMs were (more often than not) able to marginally produce better results. The code snippet below shows the details of the stacked LSTM architecture, as well as the optimizer and loss function used in the training pipeline.
```python
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(step,1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy',tf.keras.metrics.MeanSquaredError(),
             tf.keras.metrics.RootMeanSquaredError()],
    
)
model.summary()
```
The stacked LSTM was then trained for 10 epochs 
## Result
After training was completed, the model was used to predict the test set and produced a Root Mean Squared error of approximately 86, which is quite an outstandingly low figure considering our data ranges between the 2000 and 4000 area. 

The predictions of the model on both the training and test set plotted against the actual data are also displayed below.
![](Screenshot(638).png)

##Conclusion
The MTN Stock Analysis project serves as a comprehensive exploration of MTN Nigeria's stock data, employing advanced machine learning techniques to forecast future stock prices. By integrating time series data preprocessing, visualization, and LSTM modeling, this project aims to provide valuable insights and predictions to aid investors and stakeholders in making informed decisions.

For detailed implementation and code snippets, please refer to the project repository.

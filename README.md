# MTN-stock-analysis-deep-learning
Project
#### -- Project Status: [Completed]

## Project Description
After finishing Kaggle's amazing time series course, it only makes sense to pick up a project related to the topic. I decided to explore stock price prediction ideas since it is perhaps the simplest form of a time series project out there. However, instead of picking up the popular or hot company stocks (like AAPL, Tesla, etc.), I thought it would be best to analyze a local public company close to home, namely Telkom Indonesia. Telkom Indonesia is a multinational telecommunications conglomerate and is one of the most popular stocks listed on the Indonesia Stocks Exchange. In this project, I will be using Telkom's stock data from 2016 to 2021 to apply time series analysis techniques as well as creating a stacked LSTM model to learn and forecast the data.

### Methods Used
* Web Scraping
* Time Series Data Preprocessing
* Time Series Data Visualization 
* Stacked Long Short-Term Memory Neural Networks 

### Technologies
* Python
* BeautifulSoup4
* Pandas
* Matplotlib, Seaborn
* Tensorflow
* Scikit-learn

### Data
The Telkom stocks data was scraped using Beautiful Soup from the [Yahoo Finance website](https://finance.yahoo.com/quote/TLKM.JK/) and 1260 Data points were obtained comprising of stock details from 2 November 2016 to 2021. 


## Preprocessing
The data from the web scraping script already produced decently clean data. However, there are still several missing values from the `value` column originating from the Yahoo Finance site itself. I decided to simply rows, which has a null entry on `value` resulting in only a 4.7% loss of data. The stocks data also comes in IDR or Indonesian Rupiah, which has quite the unconventional use of commas and decimal points. The function below will parse or convert IDR values to recognizable integers that can easily be processed. 
```python
def idr_parser(cur_str):
    cur_str = re.sub("[,]", '', cur_str)
 
    if '.' in list(cur_str[-3:]):
        return cur_str[:-3]
    
    return cur_str
```
## Visualization
For the data visualization, I constructed distribution plots of each feature and also explored the full historical data in the 5-year span. Moving averages plots (on various window lengths) and seasonal decomposition plots are also constructed since it is time-series data after all. All of the plots are displayed in the [notebook](https://github.com/anantoj/telkom-stocks-analysis/blob/main/telkom_stock_notebook.ipynb).

## LSTMs
Recurrent Neural Networks are undoubtedly the most suitable type of neural network to handle time-series data due to their ability to retain a memory of past inputs. LSTMs are essentially just a better version of vanilla RNNs due to the cell states that allows information to be removed or added at will. Essentially, LSTMs are far better at retaining memory and are insensitive to gap length, which is quite important considering the many data points we will input. Stacked LSTMs, on the other hand, increase the depth by adding multiple hidden LSTM layers in the architecture, which allows for better learning capabilities and accuracy. After experimenting with both vanilla LSTMs and stacked LSTMs on the stocks data, stacked LSTMs were (more often than not) able to marginally produce better results. The code snippet below shows the details of the stacked LSTM architecture, as well as the optimizer and loss function used in the training pipeline.
```python
lstm = Sequential()
lstm.add(LSTM(128, return_sequences=True, input_shape=(step,1)))
lstm.add(LSTM(64, return_sequences=False))
lstm.add(Dense(25))
lstm.add(Dense(1))

lstm.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[tensorflow.keras.metrics.MeanSquaredError(),
             tensorflow.keras.metrics.RootMeanSquaredError()]
)
```
The stacked LSTM was then trained for 10 epochs 
## Result
After training was completed, the model was used to predict the test set and produced a Root Mean Squared error of approximately 70, which is quite an outstandingly low figure considering our data ranges between the 2000 and 4000 area. 

The predictions of the model on both the training and test set plotted against the actual data are also displayed below.
![png](result.png)

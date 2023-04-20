import yfinance as yf
from prophet import Prophet


def predict(ticker, days):
    yfin = yf.Ticker(ticker)
    print("Stock: ", yfin.info['name'])
    hist = yfin.history(period="max")
    hist = hist[['Close']]
    hist.reset_index(level=0, inplace=True)
    hist = hist.rename({'Date': 'ds', 'Close': 'y'}, axis='columns')
    print("Curent Data")
    print(hist.tail())
    m = Prophet(daily_seasonality=True)
    m.fit(hist)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    print("Predicted Data")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    figure1 = m.plot(forecast)
    figure2 = m.plot_components(forecast)


predict("ETH-USD", 365)

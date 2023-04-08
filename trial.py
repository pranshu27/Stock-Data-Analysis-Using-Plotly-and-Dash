import yfinance as yf
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px 
import datetime as dt
from dateutil.relativedelta import relativedelta

top50 = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'TSLA', 'JPM', 'JNJ', 'V', 'BRK-A', 'NVDA', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'INTC', 'VZ', 'CMCSA', 'KO', 'PEP', 'PFE', 'NFLX', 'T', 'ABT', 'CRM', 'CVX', 'MRK', 'WMT', 'CSCO', 'XOM', 'ABBV', 'CVS', 'ACN', 'ADBE', 'ORCL', 'BA', 'TMO', 'TGT', 'F', 'NKE', 'MDT', 'UPS', 'MCD', 'LOW', 'IBM', 'MMM', 'GE', 'AMGN']


# Define the app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Stock Comparison'),
    dcc.Tabs([
        dcc.Tab(label='Stocks\'/Attributes Comparison Playground', children=[
            html.Div([
                html.Label('Select stock symbol: '),
                dcc.Dropdown(id='input-box', options=[{'label': i, 'value': i} for i in top50], value='AAPL', multi = True),
                html.Br(),
                html.Label('Select stock attribute: '),
                dcc.Dropdown(id='stock-attribute', options=[
                    {'label': 'Open', 'value': 'Open'},
                    {'label': 'High', 'value': 'High'},
                    {'label': 'Low', 'value': 'Low'},
                    {'label': 'Close', 'value': 'Close'},
                    {'label': 'Adj Close', 'value': 'Adj Close'},
                ], value='Adj Close'),
                html.Br(),
                html.Label('Select time period: '),
                dcc.RadioItems(id='time-period', options=[
                    {'label': '1 Month', 'value': '1mo'},
                    {'label': '3 Months', 'value': '3mo'},
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '5 Years', 'value': '5y'}
                ], value='1mo'),
                html.Br(),
                dcc.Graph(id='stock-vs-sensex')
            ])
        ]),
        
        dcc.Tab(label='Average Sensex Hike/Dip', children=[
            html.Div([
                html.Label('Select Overall Time Period: '),
                dcc.RadioItems(id='overall-time-period', options=[
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '5 Years', 'value': '5y'},
                    {'label': '10 Years', 'value': '10y'},
                    {'label': '20 Years', 'value': '20y'}
                ], value='1mo'),
                html.Br(),
                html.Label('Select Rolling Average Time Period: '),
                
                dcc.RadioItems(
                    id='rolling-mean-time',
                    options=[
                        {'label': '30 days', 'value': 30},
                        {'label': '60 days', 'value': 60},
                        {'label': '90 days', 'value': 90},
                    ],
                    value=30
                ),
                html.Br(),
                dcc.Graph(id='average-sensex-hike')
            ])
        ]),
        dcc.Tab(label='Pair Plots', children=[
            html.Div([
                html.Label('Select stock symbol: '),
                dcc.Dropdown(id='input-box1', options=[{'label': i, 'value': i} for i in top50], value='AAPL'),
                html.Br(),
                dcc.Graph(id='pair-plots')
            ])
        ])
    ])
])

# Define the callback for average-sensex-hike graph
@app.callback(Output('average-sensex-hike', 'figure'),
              [Input('overall-time-period', 'value'),
               Input('rolling-mean-time', 'value')])


def update_graph(time_period, rolling_mean_time):
    end_date = dt.date.today()
    
    if time_period == '6mo':
        start_date = end_date - dt.timedelta(days=180)
    elif time_period == '1y':
        start_date = end_date - dt.timedelta(days=365)
    elif time_period == '5y':
        start_date = end_date - dt.timedelta(days=1825)
    elif time_period == '10y':
        start_date = end_date - dt.timedelta(days=3650)
    elif time_period == '20y':
        start_date = end_date - dt.timedelta(days=7300)
    else:
        start_date = end_date - dt.timedelta(days=365)
        
    sensex_data = yf.download("^BSESN", start=start_date, end=end_date)
    sensex_data['Daily Return'] = sensex_data['Adj Close'].pct_change()
    rolling_avg = sensex_data['Daily Return'].rolling(window=rolling_mean_time).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sensex_data.index, y=rolling_avg, mode='lines', name='Rolling Average'))
    # fig.update_layout(title='Average Sensex Hike', xaxis_title='Date', yaxis_title='Average Daily Return', )
    
    fig.update_layout(
    title='Average Sensex Hike',
    xaxis_title='Date',
    yaxis_title='Average Daily Return',
    hovermode='x unified',
    xaxis=dict(showspikes=True, spikemode='across', spikedash='dot'),
    yaxis=dict(showspikes=True, spikemode='across', spikedash='dot')
    )

    fig.add_shape(
        dict(
            type='line',
            x0=sensex_data.index[0],
            y0=0,
            x1=sensex_data.index[-1],
            y1=0,
            line=dict(
                color='black',
                width=1,
                dash='dash'
            )
        )
    )

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward")
                    
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    
    return fig





# Define the callback for stock vs. sensex graph
@app.callback(Output('stock-vs-sensex', 'figure'),
              [Input('input-box', 'value'),
               Input('stock-attribute', 'value'),
               Input('time-period', 'value')])

def update_stock_vs_sensex(ticker, attribute, time_period):
    tickers  = ticker
    # Get data for selected stock and Sensex
    end_date = dt.date.today()
    if time_period == '1mo':
        start_date = end_date - dt.timedelta(days=30)
    elif time_period == '3mo':
        start_date = end_date - dt.timedelta(days=90)
    elif time_period == '6mo':
        start_date = end_date - dt.timedelta(days=180)
    elif time_period == '1y':
        start_date = end_date - dt.timedelta(days=365)
    elif time_period == '5y':
        start_date = end_date - dt.timedelta(days=1825)
    else:
        start_date = end_date - dt.timedelta(days=365)
    
    data = []
    tmp = None
    if type(tickers) == str:
        stock_data = yf.download(tickers, start=start_date, end=end_date)
        trace = go.Scatter(x=stock_data.index, y=stock_data[attribute], name=tickers)
        tmp = tickers
        data.append(trace)
    else:
        for ticker in tickers:
            # print(ticker, type(ticker))
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            trace = go.Scatter(x=stock_data.index, y=stock_data[attribute], name=ticker)
            data.append(trace)
            tmp = ','.join(tickers)

    layout = go.Layout(
    title=f'{attribute} - Comparison of {tmp} over {time_period}',
    yaxis=dict(title='Price(₹)'),
    hovermode='x unified',
    xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    
                    
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return {'data': data, 'layout': layout}


# Define the callback for pair plots
@app.callback(Output('pair-plots', 'figure'),
              [Input('input-box1', 'value')])
def update_graph(stock_symbol):
    # Get data for stock
    stock_data = yf.download(stock_symbol, start='2022-01-01', end='2023-04-06')

    # Create pair plot
    fig = px.scatter_matrix(stock_data,
                            dimensions=['Open', 'High', 'Low', 'Close', 'Adj Close'],
                            color='Volume')

    # Update layout
    fig.update_layout(title=f'{stock_symbol} Pair Plots')

    # Return figure
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
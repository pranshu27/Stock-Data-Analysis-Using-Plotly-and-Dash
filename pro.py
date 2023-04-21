import yfinance as yf
import pandas as pd
import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import plotly.graph_objs as go
import plotly.express as px 
import datetime as dt
import plotly.figure_factory as ff
from pandas.plotting import scatter_matrix
import warnings
import yfinance as yf
import pandas as pd
import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import plotly.graph_objs as go
import plotly.express as px
import datetime as dt
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
import dash_bootstrap_components as dbc
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn import linear_model
from keras.models import Sequential, load_model
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import seaborn as sns

from scipy.stats import norm        
from dateutil.relativedelta import relativedelta
import dash_bootstrap_components as dbc

top50 = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'BRK-A', 'NVDA', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'INTC', 'VZ', 'CMCSA', 'KO', 'PEP', 'PFE', 'NFLX', 'T', 'ABT', 'CRM', 'CVX', 'MRK', 'WMT', 'CSCO', 'XOM', 'ABBV', 'CVS', 'ACN', 'ADBE', 'ORCL', 'BA', 'TMO', 'TGT', 'F', 'NKE', 'MDT', 'UPS', 'MCD', 'LOW', 'IBM', 'MMM', 'GE', 'AMGN']

min_date = '2010-01-01'
max_date = str(dt.date.today())
import datetime

# Convert min_date and max_date to Unix timestamps
min_timestamp = datetime.datetime.strptime(min_date, '%Y-%m-%d').timestamp()
max_timestamp = datetime.datetime.strptime(max_date, '%Y-%m-%d').timestamp()

# Define the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'custom.css'])


# Define the navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("HOME", href="javascript:location.reload(true)")),
        dbc.NavItem(dbc.NavLink("ABOUT", href="/about", active = 'exact'))
    ],
    brand="Know your stocks.. Become a Pro Investor !",
    brand_href="#",
    color="brown",
    dark=True,
)

# Define the list of time periods
periods = {
    '6m': '6mo',
    '1y': '1y',
    '2y': '2y',
    '5y': '5y',
    '10y': '10y',
    '20y': '20y'
}


# Define the layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Br(),
    dbc.Tabs([
        # dcc.Tab(label='Risk Analysis using LR', children=[
        #     html.Div([
        #         html.Br(),
        #         html.Label('Select stock symbol:', style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
        #         dcc.Dropdown(id='st', options=[{'label': i, 'value': i} for i in top50], value='AAPL'),
        #         html.Br(),
        #         html.Label('Select Time Period used for Prediction: ', style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
        #         dcc.RadioItems(id='overall', options=[
        #             {'label': '6 Months', 'value': '6mo'},
        #             {'label': '1 Year', 'value': '1y'},
        #             {'label': '5 Years', 'value': '5y'},
        #             {'label': '10 Years', 'value': '10y'},
        #             {'label': '20 Years', 'value': '20y'}
        #         ], value='1y'),
        #         html.Br(),
        #         html.Label('Select Prediction Time Period: ', style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                
        #         dcc.RadioItems(
        #             id='prediction-time',
        #             options=[
        #                 {'label': '30 days', 'value': '30'},
        #                 {'label': '60 days', 'value': '60'},
        #                 {'label': '90 days', 'value': '90'},
        #             ],
        #             value='30'
        #         ),
        #         html.Br(),
        #         dcc.Graph(id='risk-analysis-graph')
        #     ])
        # ]),


        dcc.Tab(label = 'Technical Analysis using Candlesticks Chart/Moving Averages', children=[
            html.Div([
    html.Label('Select stock symbol: ', style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
    dcc.Dropdown(id='symbol-dropdown', options=[{'label': s, 'value': s} for s in top50], value=top50[0]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Label('Select Time Period for Technical Analysis', style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
    dcc.Dropdown(id='period-dropdown', options=[{'label': k, 'value': v} for k, v in periods.items()], value='1y'),
    html.Br(),
    html.Br(),
    html.Label('Select Number of Days for Moving Average', style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
    dcc.RangeSlider(
        id='moving-average-slider',
        min=5,
        max=60,
        step=5,
        value=[20, 50],
        marks={i: f'{i} days' for i in range(5, 61, 5)}
    ),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='candlestick-chart'),
            width={'size': 10, 'offset': 1}
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='moving-averages-chart'),
            width={'size': 10, 'offset': 1}
        )
    ]),
])
        ]),
        

        dcc.Tab(label='Risk Analysis using Monte Carlo Simulation', children=[
            html.Div([
                
                html.Label('Select stock symbol: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.Dropdown(id='stk', options=[{'label': i, 'value': i} for i in top50], value='MSFT'),
                html.Br(),
                html.Label('Select number of simulations to run: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.RadioItems(id='sim', options=[
                    {'label': '5', 'value': '5'},
                    {'label': '50', 'value': '50'},
                    {'label': '100', 'value': '100'},
                    {'label': '500', 'value': '500'},
                    # {'label': '20 Years', 'value': '20y'}
                ], value='100'),
                html.Br(),
                html.Label('Select Prediction Time Period: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                
                dcc.RadioItems(
                    id='time',
                    options=[
                        {'label': '30 days', 'value': '30'},
                        {'label': '60 days', 'value': '60'},
                        {'label': '90 days', 'value': '90'},
                    ],
                    value='30'
                ),
                html.Br(),
                dcc.Graph(id='monte-carlo'),
                dcc.Graph(id='monte-carlo-dist')
            ])
        ]),

        # stk, sim, time

        dcc.Tab(label='Stocks\'/Attributes Comparison Playground', children=[
            html.Div([
                html.Label('Select stock symbol: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.Dropdown(id='stock-select', options=[{'label': i, 'value': i} for i in top50], value=['AMZN', 'MSFT'], multi = True),

                html.Br(),
                html.Label('Select stock attribute: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.Dropdown(id='stock-attribute', options=[
                    {'label': 'Open', 'value': 'Open'},
                    {'label': 'High', 'value': 'High'},
                    {'label': 'Low', 'value': 'Low'},
                    {'label': 'Close', 'value': 'Close'},
                    {'label': 'Adj Close', 'value': 'Adj Close'},
                ], value='Adj Close'),
                html.Br(),
                html.Label('Select time period: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
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
        
        dcc.Tab(label='Avg. Sensex Hike/Dip', children=[
            html.Div([
                html.Label('Select Overall Time Period: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.RadioItems(id='overall-time-period', options=[
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '5 Years', 'value': '5y'},
                    {'label': '10 Years', 'value': '10y'},
                    {'label': '20 Years', 'value': '20y'}
                ], value='6mo'),
                html.Br(),
                html.Label('Select Rolling Average Time Period: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                
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
        dcc.Tab(label='Trend Analysis', children=[
            html.Div([

                html.Label('Select stock symbol: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.Dropdown(id='stkk3', options=[
                             {'label': i, 'value': i} for i in top50], value='MSFT'),
                html.Br(),
                html.Label('Select Time Period: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.RadioItems(id='overall3', options=[
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '5 Years', 'value': '5y'},
                    {'label': '10 Years', 'value': '10y'},
                    {'label': '20 Years', 'value': '20y'}
                ], value='1y'),
                html.Br(),
                dcc.Graph(id='trend-analysis')
            ])
        ]),
        dcc.Tab(label='Pair Plots', children=[
            html.Div([
                html.Label('Select stock symbol: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.Dropdown(id='ticker-select', options=[{'label': i, 'value': i} for i in top50], value='MSFT'),
                html.Br(),
                dcc.Graph(id='pair-plots')
            ])
        ]),
        
        dcc.Tab(label='Long/Short Term Investment', children=[
            html.Div([
                html.Label('Select stock symbol: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.Dropdown(id='input-box', value = ["MSFT", "AMZN"], options=[{'label': i, 'value': i} for i in top50], multi = True),
                html.Br(),
                html.Label('Select Time Period used for Prediction: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.RadioItems(id='overall', options=[
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '5 Years', 'value': '5y'},
                    {'label': '10 Years', 'value': '10y'},
                    {'label': '20 Years', 'value': '20y'}
                ], value='1y'),
                html.Br(),
                dcc.Graph(id='Long-short')
            ])
        ]),
        
        
        dcc.Tab(label='Risk Analysis using LSTM', children=[
            html.Div([

                html.Label('Select stock symbol: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.Dropdown(id='st', options=[
                             {'label': i, 'value': i} for i in top50], value='MSFT'),
                html.Br(),
                html.Label('Select Time Period used for Prediction: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
                dcc.RadioItems(id='overallnew', options=[
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '5 Years', 'value': '5y'},
                    {'label': '10 Years', 'value': '10y'},
                    {'label': '20 Years', 'value': '20y'}
                ], value='1y'),
                html.Br(),
                html.Label('Select Prediction Time Period: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),

                dcc.RadioItems(
                    id='prediction-time',
                    options=[
                        {'label': '30 days', 'value': '30'},
                        {'label': '60 days', 'value': '60'},
                        {'label': '90 days', 'value': '90'},
                    ],
                    value='30'
                ),
                html.Br(),

                dcc.Graph(id='risk-analysis-graph')
            ])
        ]),
        dcc.Tab(label='Know where to invest!', children=[
        html.Div([
            html.Label('Select stock symbols:',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
            dcc.Dropdown(
                id='stocks-dropdown',
                options=[{'label': i, 'value': i} for i in top50],
                value=['MSFT','AMZN'],
                multi=True
            ),
            html.Br(),

            html.Label('Select the investment amount:',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
            dcc.Input(
                id='investment-amount',
                type='number',
                value=1000,
                style={'width': '300px'}
            ),
            html.Br(),

            html.Label('Select date range:',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
            dcc.RangeSlider(
                id='date-range-slider',
                min=min_timestamp,
                max=max_timestamp,
                value=[min_timestamp, max_timestamp],
                # marks = {
                #     int(date.timestamp()): {'label': date.strftime('%Y-%m-%d'), 'style': {'cursor': 'pointer'}, 'tooltip': date.strftime('%Y-%m-%d')}
                #     for date in pd.date_range(start=min_date, end=max_date)
                # },
                marks={
                    int(date.timestamp()): date.strftime('%Y-%m-%d')
                    for date in pd.date_range(start=min_date, end=max_date)
                },
                # tooltip={
                #     'always_visible': True,
                #     'placement': 'topLeft',
                #     'format': {'specifier': '%Y-%m-%d'}
                # },
                step=None
            ),
            html.Div(id='date-range-slider-output'),
            html.Br(),

            dcc.Graph(id='allocation', style={'height': '600px'})
        ], style={'padding': '20px'})
        ], selected_style={'font-weight': 'bold'}),
        
        
        dcc.Tab(label='Correlation Analysis of Multiple Companies', children=[
                    html.Div([
            html.Label('Select stock symbol: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
            dcc.Dropdown(id='st1', options=[{'label': i, 'value': i} for i in top50], value=['MSFT','AMZN'], multi = True),
            html.Br(),

            html.Label('Select the analysis period: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
            dcc.RadioItems(id='period', options=[
                {'label': '1 Month', 'value': '1m'},
                {'label': '6 Months', 'value': '6m'},
                {'label': '1 Year', 'value': '1y'},
                {'label': '5 Years', 'value': '5y'},
                {'label': '10 Years', 'value': '10y'}
            ], value='1y'),
            html.Br(),

            html.Label('Select data component: ',style={'font-weight': 'bold', 'font-size': '20px', 'color': '#007bff'}),
            dcc.Dropdown(id='data-component', options=[
                {'label': 'Adj Close', 'value': 'Adj Close'},
                {'label': 'Open', 'value': 'Open'},
                {'label': 'High', 'value': 'High'},
                {'label': 'Low', 'value': 'Low'},
                {'label': 'Close', 'value': 'Close'},
                {'label': 'Volume', 'value': 'Volume'}
            ], value='Adj Close'),
            html.Br(),
            
             html.Div(
                dcc.Graph(id='correlation-graph'),
                style={'display': 'flex', 'justify-content': 'center'} # Align graph to center
            )

            # dcc.Graph(id='correlation-graph', style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})

        ])
    
        ])
    ])
])



@app.callback(
    dash.dependencies.Output('date-range-slider-output', 'children'),
    [dash.dependencies.Input('date-range-slider', 'value')]
)
def update_output(value):
    return f'Selected dates: {pd.to_datetime(value[0], unit="s").strftime("%Y-%m-%d")} to {pd.to_datetime(value[1], unit="s").strftime("%Y-%m-%d")}'




# Define the callback for candlestick chart
@app.callback(
    Output('candlestick-chart', 'figure'),
    [Input('symbol-dropdown', 'value'),
     Input('period-dropdown', 'value')])
def update_candlestick_chart(symbol, period):
    # Load stock data
    df = yf.Ticker(symbol).history(period=period)
    
    # Create the candlestick chart trace
    trace = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
    
    # Create the figure layout
    layout = go.Layout(
        title=dict(text='Candlesticks Chart', font=dict(size=20)),
        xaxis_rangeslider_visible=False,
        yaxis=dict(title='Price'),
        xaxis=dict(title='Date'),
    )
    
    # Create the figure object
    fig = go.Figure(data=[trace], layout=layout)
    
    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
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


# Define the callback for moving averages chart
@app.callback(
    Output('moving-averages-chart', 'figure'),
    [Input('symbol-dropdown', 'value'),
     Input('period-dropdown', 'value'),
     Input('moving-average-slider', 'value')])
def update_moving_averages_chart(symbol, period, moving_average_days):
    # Load stock data
    df = yf.Ticker(symbol).history(period=period)
    
    # Calculate moving averages
    moving_averages = []
    for days in moving_average_days:
        moving_averages.append(df['Close'].rolling(window=days).mean())
    
    # Create the moving averages traces
    traces = []
    for i, days in enumerate(moving_average_days):
        trace = go.Scatter(x=df.index, y=moving_averages[i], mode='lines', name=f'{days}-day MA')
        traces.append(trace)
    
    # Create the figure layout
    layout = go.Layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Moving Averages'),
        title=dict(text='Moving Averages', font=dict(size=20)),
        showlegend=True,
        legend=dict(x=0, y=1),
    )
    
    # Create the figure object
    fig = go.Figure(data=traces, layout=layout)
    
    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
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
              [Input('stock-select', 'value'),
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
              [Input('ticker-select', 'value')])
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


#===============================================================================================================================================================



#    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #===============================================================================================================================================================
# Define function to calculate expected stock price using Monte Carlo simulation
def monte_carlo_simulation(start_price, days, mu, sigma, num_simulations):
    dt = 1 / 252  # time interval for simulation
    prices = np.zeros((days, num_simulations))  # initialize prices array
    prices[0] = start_price  # set the initial price

    for i in range(1, days):
        # calculate the daily returns
        daily_returns = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * norm.ppf(np.random.rand(num_simulations)))

        # calculate the new prices
        prices[i] = prices[i-1] * daily_returns

    return prices

@app.callback(
    [Output('monte-carlo', 'figure'), Output('monte-carlo-dist', 'figure')],
    [Input('stk', 'value'),
     Input('sim', 'value'),
     Input('time', 'value')]
)

def update_risk_analysis_graph_monte_carlo(stock, simulations, prediction_time):
    end_date = dt.date.today()

    if prediction_time == '30':
        days = 30
    elif prediction_time == '60':
        days = 60
    else:
        days = 90

    start_date = end_date - dt.timedelta(days= 730)
    data = yf.download(stock, start=start_date, end=end_date)

    # calculate the daily returns
    returns = np.log(1 + data['Close'].pct_change()).dropna()

    # calculate the expected return and volatility
    mu = returns.mean()
    sigma = returns.std()

    # get the last closing price
    start_price = data['Close'][-1]

    # perform Monte Carlo simulation
    if simulations == '5':
        num_simulations = 5
    elif simulations == '50':
        num_simulations = 50
    elif simulations == '500':
        num_simulations = 500
    else:
        num_simulations = 100

    prices = monte_carlo_simulation(start_price, days, mu, sigma, num_simulations)

    # plot the results
    fig = go.Figure()

    for i in range(num_simulations):
        fig.add_trace(go.Scatter(x=data.index, y=prices[:, i], mode='lines', name='Simulation ' + str(i+1)))

    fig.update_layout(title='Monte Carlo Simulation', xaxis_title='Date', yaxis_title='Stock Price')
    # return fig  

    #  plot the distribution of final results
    final_prices = prices[-1, :]
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=final_prices, nbinsx=20, name='Final Prices'))
    fig_dist.add_vline(x=np.percentile(final_prices, 1), line_dash='dash', line_color='red', name='1% Quantile')
    fig_dist.update_layout(title='Distribution of Final Prices', xaxis_title='Final Stock Price at 1% Quartile', yaxis_title='Frequency',  bargap=0.1)
    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=15, label="15d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                   
                    
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig, fig_dist




@app.callback(Output('correlation-graph', 'figure'),
              [Input('st1', 'value'),
               Input('period', 'value'),
               Input('data-component', 'value')])

def correlation(st1, time_period, data_component):
    end_date = dt.date.today()

    if time_period == '6m':
        start_date = end_date - dt.timedelta(days=180)
    elif time_period == '1m':
        start_date = end_date - dt.timedelta(days=30)
    elif time_period == '5y':
        start_date = end_date - dt.timedelta(days = 1825)
    elif time_period == '10y':
        start_date = end_date - dt.timedelta(days = 3650)
    else:
        start_date = end_date - dt.timedelta(days=365)

    stocks = []
    for i in st1:
        stocks.append(i)

    closing_df = yf.download(stocks, start=start_date, end=end_date)[data_component] # Use the selected data component

    # compute daily returns
    tech_rets = closing_df.pct_change()

    # the same scatter plot using plotly instead of cufflinks
    fig = ff.create_scatterplotmatrix(tech_rets, diag='histogram', size=5,
                                      height=740, width=880)

    for trace in fig['data']:
        trace['opacity'] = 0.7
        trace['marker'] = dict(color="indianred", line=dict(color='white',
                                                           width=0.7))

    return fig




# Trend Analysis
@app.callback(Output('trend-analysis', 'figure'),
              [Input('stkk3', 'value')],
              [Input('overall3', 'value')])
def update_trend_analysis(stock_name, time_period):
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

    sx_data = yf.download(stock_name, start=start_date, end=end_date)

    # Function defining trend
    def trend(x):
        if x > -0.5 and x <= 0.5:
            return 'Slight or No change'
        elif x > 0.5 and x <= 1:
            return 'Slight Positive'
        elif x > -1 and x <= -0.5:
            return 'Slight Negative'
        elif x > 1 and x <= 3:
            return 'Positive'
        elif x > -3 and x <= -1:
            return 'Negative'
        elif x > 3 and x <= 7:
            return 'Among top gainers'
        elif x > -7 and x <= -3:
            return 'Among top losers'
        elif x > 7:
            return 'Bull run'
        elif x <= -7:
            return 'Bear drop'

    # Compute the daily percentage change in the stock prices
    sx_data['Day_Perc_Change'] = sx_data['Close'].pct_change() * 100

    # Add a new column for the trend
    sx_data['Trend'] = sx_data['Day_Perc_Change'].apply(trend)

    # Group the data by trend and count the number of occurrences
    sx_pie = sx_data.groupby(
        'Trend')['Trend'].count().reset_index(name='count')

    # Plot the pie chart using Plotly
    fig = px.pie(sx_pie, values='count', names='Trend',
                 title='Trend Analysis',
                 hole=0.4, color='Trend', height=800, width=600)
    return fig


# # Define the callback function to update the scatterplot matrix
# @app.callback(
#     dash.dependencies.Output('scatterplot-matrix', 'children'),
#     [dash.dependencies.Input('stock-dropdown', 'value'),
#      dash.dependencies.Input('component-dropdown', 'value'),
#      dash.dependencies.Input('time-dropdown', 'value')])
# def update_scatterplot_matrix(stocks, component, time_period):
#     # Create an empty DataFrame to store the stock data
#     stock_data = pd.DataFrame()
    
#     # Loop through each selected stock and download the corresponding data using yfinance
#     for stock in stocks:
#         stock_data = pd.concat([stock_data, yf.download(stock, period=time_period, interval='1d')[component]], axis=1)
    
#     # Rename the columns to include the stock ticker and component name
#     stock_data.columns = [f'{stock}_{component}' for stock in stocks]
    
#     # Create a scatterplot matrix using pandas
#     fig = px.scatter_matrix(stock_data)
    
#     # Return the scatterplot matrix as a Plotly figure
#     return dcc.Graph(figure=fig)






@ app.callback(
    Output('risk-analysis-graph', 'figure'),
    [Input('st', 'value'),
     Input('overallnew', 'value'),
     Input('prediction-time', 'value')
    ]
    
)
def update_risk_analysis_graph(stock, tot_time, prediction_time):
    end_date = dt.date.today()

#  we predict from end_date to till after looking at start_time to till data
    if prediction_time == '30':
        till = end_date - dt.timedelta(days=30)
    elif prediction_time == '60':
        till = end_date - dt.timedelta(days=60)
    elif prediction_time == '90':
        till = end_date - dt.timedelta(days=90)
    else:
        till = end_date - dt.timedelta(days=30)

    df2 = yf.download(stock, start=till, end=end_date)

    end_date = dt.date.today()

    if tot_time == '6mo':
        start_date = till - dt.timedelta(days=180)
    elif tot_time == '1y':
        start_date = till - dt.timedelta(days=365)
    elif tot_time == '5y':
        start_date = till - dt.timedelta(days=1825)
    elif tot_time == '10y':
        start_date = till - dt.timedelta(days=3650)
    elif tot_time == '20y':
        start_date = till - dt.timedelta(days=7300)
    else:
        start_date = till - dt.timedelta(days=365)

    df = yf.download(stock, start=start_date, end=till)

    features = ['Open', 'High', 'Low', 'Volume']
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(df[features])
    feature_transform = pd.DataFrame(
        columns=features, data=feature_transform, index=df.index)
    # feature_transform.head() ===================================================================
    feature_transform_test = scaler.transform(df2[features])
    feature_transform_test = pd.DataFrame(
        columns=features, data=feature_transform_test, index=df2.index)

    # Splitting to Training set and Test set
    X_train = feature_transform
    y_train = df['Adj Close'].values.ravel()
    X_test = feature_transform_test
    y_test = df2['Adj Close'].values.ravel()

    # Process the data for LSTM
    trainX = np.array(X_train)
    testX = np.array(X_test)
    X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Building LSTM model
    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(
        1, trainX.shape[1]), activation='relu', return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')

    # Training LSTM model
    history = lstm.fit(X_train, y_train, epochs=100,
                        batch_size=8, verbose=1, shuffle=False)
    y_pred = lstm.predict(np.array(X_test))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df2.index, y=y_test,
                    mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df2.index, y=y_pred.flatten(),
                    mode='lines', name='Predicted'))

    fig.update_layout(title='Stock Prediction using LSTM',
                        xaxis_title='Date', yaxis_title='Price')
    
    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=15, label="15d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                   
                    
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig

        # df training df2 testing
# Define the callback for long short graph
@app.callback(Output('Long-short', 'figure'),
              [Input('input-box', 'value'),
               Input('overall', 'value')])

def update_stock_categories(ticker, time_period):
    tickers = ticker
    
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
    
    stock_data = yf.download(tickers, start=start_date, end=end_date)

    # Compute the daily percentage change in the stock prices
    df_daily_pct_change = stock_data['Adj Close'].pct_change()

    # Compute the coefficient of variation (CV) for each stock
    df_cv = df_daily_pct_change.std() / df_daily_pct_change.mean()
    
    # Categorize the stocks into three categories based on their CV values
    long_term_stocks = df_cv[df_cv >= np.percentile(df_cv, 75)].index.tolist()
    short_term_stocks = df_cv[df_cv <= np.percentile(df_cv, 25)].index.tolist()
    other_stocks = df_cv[(df_cv > np.percentile(df_cv, 25)) & (df_cv < np.percentile(df_cv, 75))].index.tolist()

    # Filter the CV dataframe to include only the three categories of stocks
    df_cv_filtered = df_cv[df_cv.index.isin(long_term_stocks + short_term_stocks + other_stocks)]

    # Create a bar chart of the CV values for each stock
    fig = go.Figure(data=[go.Bar(x=df_cv_filtered.index, y=df_cv_filtered)])
    fig.update_layout(title='Variability of Stocks', xaxis_title='Stocks', yaxis_title='CV')

    # Highlight the three categories of stocks using different colors
    fig.update_traces(marker_color=np.where(df_cv_filtered.index.isin(long_term_stocks), 'green',
                                             np.where(df_cv_filtered.index.isin(short_term_stocks), 'red', 'blue')))
    fig.update_layout(showlegend=True)

    # Add a legend to show which color is for which category
    fig.update_layout(legend=dict(title='', orientation='v', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='green'), name='Long-term'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='red'), name='Short-term'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='blue'), name='Other'))

    return fig




@app.callback(Output('allocation', 'figure'),
              [Input('stocks-dropdown', 'value'),
              Input('investment-amount', 'value'),
              Input('date-range-slider', 'value')])

def update_allocation(stocks, investment_amount, date_range):
    start_date = date_range[0]
    end_date = date_range[1]
    
    df = yf.download(stocks, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    daily_returns = df.pct_change()
    
    # Calculate mean and standard deviation of daily returns
    mean_daily_returns = daily_returns.mean()
    std_daily_returns = daily_returns.std()
    
    # Calculate Sharpe ratio
    sharpe_ratio = (mean_daily_returns / std_daily_returns) * np.sqrt(252)
    
    # Allocate funds based on Sharpe ratio
    weights = sharpe_ratio / np.sum(sharpe_ratio)
    allocations = investment_amount * weights
    
    # Create list of labels for companies with zero allocation
    zero_allocations = list(set(df.columns) - set(stocks))
    
    # Add zero-allocated companies to stocks and allocations lists
    stocks += zero_allocations
    allocations = np.append(allocations, np.zeros(len(zero_allocations)))
    
    fig = go.Figure(data=[go.Pie(labels=stocks, values=allocations, hole=.5)])

    # Update layout
    fig.update_layout(title='Money Allocation',
                    font=dict(size=12),
                    annotations=[dict(text='Total Amount', x=0.5, y=0.5, font_size=20, showarrow=False)],
                    showlegend=False)

    # Display chart
    return fig


if __name__ == '__main__':
    app.run_server(port=8051, debug=False)

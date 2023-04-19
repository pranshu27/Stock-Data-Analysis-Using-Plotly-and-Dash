import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import yfinance as yf
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

# Define the list of stock symbols
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

# Define the layout of the dashboard using Bootstrap components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("Interactive Candlestick Chart", className="text-center mb-4"), width=12)
    ),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(id='symbol-dropdown', options=[{'label': s, 'value': s} for s in symbols], value=symbols[0]),
            width={'size': 2, 'offset': 1}
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='graph', style={'backgroundColor': '#000000'}),
            width={'size': 10, 'offset': 1}
        )
    ]),
], fluid=True)

# Define the update function for the graph
@app.callback(
    Output('graph', 'figure'),
    [Input('symbol-dropdown', 'value')])
def update_figure(symbol):
    # Load stock data
    df = yf.Ticker(symbol).history(period="1y")
    
    # Create the candlestick chart trace
    trace = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
    
    # Create the figure layout
    layout = go.Layout(
        xaxis_rangeslider_visible=False,
        yaxis=dict(title='Price'),
        xaxis=dict(title='Date'),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000'
    )
    
    # Create the figure object
    fig = go.Figure(data=[trace], layout=layout)
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

seasonal_regression_tab = html.Div([
    html.H1('Seasonal Regression Analysis'),
    html.P('This tab uses regression analysis to identify the relationship between a stock\'s performance and various seasonal factors.'),
    dcc.Graph(
        id='seasonal-regression-graph'
    )
])

app.layout = html.Div([
    html.Nav([
        html.A('Stock Comparison', className='navbar-brand', href='#'),
        html.Div([
            html.Ul([
                html.Li(
                    html.A('Home', className='nav-link', href='#')
                ),
                html.Li(
                    html.A('Features', className='nav-link', href='#')
                ),
                html.Li(
                    html.A('Pricing', className='nav-link', href='#')
                ),
                html.Li(
                    html.A('About', className='nav-link', href='#')
                ),
                html.Li(
                    html.A('Seasonal Regression', className='nav-link active', href='#')
                ),
            ], className='navbar-nav'),
            html.Form([
                html.Input(
                    className='form-control mr-sm-2',
                    type='search',
                    placeholder='Search',
                    aria_label='Search'
                ),
                html.Button(
                    className='btn btn-outline-success my-2 my-sm-0',
                    type='submit',
                    children='Search'
                )
            ], className='form-inline my-2 my-lg-0')
        ], className='collapse navbar-collapse', id='navbarNav')
    ], className='navbar navbar-expand-lg navbar-dark bg-primary'),
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='Stock Comparison', children=[stock_comparison_tab]),
            dcc.Tab(label='Seasonal Regression', children=[seasonal_regression_tab])
        ])
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)

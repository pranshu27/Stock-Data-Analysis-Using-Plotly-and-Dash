import dash
import dash_bootstrap_components as dbc
import dash_html_components as html


# Define the layout for the About page
about_layout = html.Div([
    dbc.Container([
        html.H1('About Stock Market Comparison', className='text-center mt-5 mb-4'),
        html.P('Stock Market Comparison is a web application that allows you to compare the performance of multiple stocks over a given period of time. With interactive charts and graphs, you can easily see how stocks have performed and identify trends and patterns that may help you make better investment decisions.', className='lead text-center mb-5'),
        html.H3('Meet the Developers', className='text-center'),
        dbc.Row([
            dbc.Col([
                html.Img(src='/assets/images/avatar1.jpg', className='img-fluid rounded-circle mx-auto d-block mt-4 mb-3', style={'width': '150px'}),
                html.H4('John Smith', className='text-center'),
                html.P('John is a full-stack developer with experience in building web applications using Python and JavaScript.', className='text-center')
            ], md=4),
            dbc.Col([
                html.Img(src='/assets/images/avatar2.jpg', className='img-fluid rounded-circle mx-auto d-block mt-4 mb-3', style={'width': '150px'}),
                html.H4('Jane Doe', className='text-center'),
                html.P('Jane is a data analyst with expertise in financial markets and statistical modelling.', className='text-center')
            ], md=4),
            dbc.Col([
                html.Img(src='/assets/images/avatar3.jpg', className='img-fluid rounded-circle mx-auto d-block mt-4 mb-3', style={'width': '150px'}),
                html.H4('David Lee', className='text-center'),
                html.P('David is a UI/UX designer with experience in creating intuitive and user-friendly interfaces for web and mobile applications.', className='text-center')
            ], md=4)
        ]),
        html.Hr(),
        html.P('Stock data provided by Yahoo Finance API.', className='text-center mt-4')
    ], className='mt-5')
    , html.Div(id = 'page-content')
])


# Create a new Dash app object for the About page
about_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
about_app.layout = about_layout


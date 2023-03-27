from jupyter_dash import JupyterDash
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.express as px

def generate_timeline(df, column):
    fig = px.line(df, x=df.index, y=column)
    fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=20, b=20))
    return fig

def generate_boxplot(df, column):
    fig = px.box(df, x=column)
    fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def uniswap_report(data_fetcher):
    app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    tabs_children = []
    for pool_name, pool_data in data_fetcher.pools.items():
        pool_daily_data = pool_data['daily_data']
        pool_swap_data = pool_data['swap_data']
        
        daily_data_subtab_children = [
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Number of Transactions"),
                    dbc.CardBody(dcc.Graph(figure=generate_timeline(pool_daily_data, 'Tx Count')))
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fees USD"),
                    dbc.CardBody(dcc.Graph(figure=generate_timeline(pool_daily_data, 'Fees USD')))
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Trading Volume"),
                    dbc.CardBody(dcc.Graph(figure=generate_timeline(pool_daily_data, 'Trading Volume')))
                ])
            ], width=4)
        ]
        
        swap_data_subtab_children = [
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Price Distribution"),
                    dbc.CardBody(dcc.Graph(figure=generate_boxplot(pool_swap_data, 'price')))
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Tick Distribution"),
                    dbc.CardBody(dcc.Graph(figure=generate_timeline(pool_swap_data, 'price')))
                ])
            ], width=6)
        ]

        pool_tab = dbc.Tab([
            dbc.Tabs([
                dbc.Tab(dbc.Row(daily_data_subtab_children), label="Daily Data"),
                dbc.Tab(dbc.Row(swap_data_subtab_children), label="Swap Data")
            ])
        ], label=pool_name)

        tabs_children.append(pool_tab)

    app.layout = dbc.Container([
        dbc.Tabs(tabs_children)
    ])

    app.run_server(mode="inline")

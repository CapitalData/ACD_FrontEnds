# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])

curriculum = pd.read_csv("CurriculumEntity.csv")
df = pd.DataFrame(curriculum, columns = ['name', 'url', 'method', 'style', 'perspective', 'colorformat', 'scriptformat', 'topic', 'scopesequence'])


app.layout = html.Div([
    
    html.Div(
        className="container",
        children=[
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="col-lg",
                        children=[ 
                            html.Div(
                                children=[
                                    html.H1(
                                        children=[
                                            html.Br(),
                                            "Education Graph",
                                        ],
                                        style={"text-align": "left"},
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Div(
                        className="col-lg",
                        children=[
                            html.A(
                                                html.Img(
                                                    src="assets/ACD.png",
                                                    style={"float": "right", "height": "100px"},
                                                ),
                                                href="https://austincapitaldata.com",
                                            ),
                        ]
                    )
                ]
            )
        ]
    ),
    html.Div(
        className="container",
        children=[
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="col-lg",
                        children=[
                            html.Div(
                                className="card my-3",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("A graph-based approach to home school curriculum", className="card-title"),
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-lg",
                        children=[
                            html.Div(
                                className="card my-3",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("About", className="card-title"),
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
        ]
    ),
    html.Div(
        className="container",
        children=[
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="col-lg",
                        children=[
                            html.Div(
                                className="card mb-3",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.H5("How to use", className="card-title"),
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                ]
            )
        ]
    ),
    html.Div(
        className="container",
        children=[
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="col-sm",
                        children=[
                            html.Div(
                                className="card mb-3",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.Label(["Do you prefer mastery or spiral",
                                            dcc.Dropdown(
                                                id='methodform',
                                                options=[
                                                    {'label': 'Mastery', 'value': 'mastery'},
                                                    {'label': 'Spiral', 'value': 'spiral'}],
                                                    value='mastery',
                                            ),
                                            ]),
                                            html.Label(["Do you prefer conceptual or procedural curriculum?",
                                            dcc.Dropdown(
                                                id='styleform',
                                                options=[
                                                    #{'label': 'Conceptual', 'value': 'conceptual'},
                                                    #{'label': 'Procedural', 'value': 'procedural'}],
                                                    {'label': 'Nonsecular', 'value': 'nonsecular'},
                                                    {'label': 'Secular', 'value': 'secular'}],
                                                    value='nonsecular',
                                            ),]),
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="col-lg",
                        children=[
                        html.Div(
                                className="card mb-3",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            html.Span("This is an example of a box in white.")
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                ]
            )
        ]
    ),
    html.Div(
        className="container",
        children=[
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="col-lg",
                        children=[
                            html.Div(
                                className="card mb-3",
                                children=[
                                    html.Div(
                                        className="card-body",
                                        children=[
                                            dcc.Graph(
                                                id='example-graph',
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                ]
            )
        ]
    ),
    
])

@app.callback(
    Output('example-graph', 'figure'),
    Input('methodform', 'value'),
    Input('styleform', 'value')
    )
def update_figure(selected_method, selected_perspective): #, )
    #filtered_df = df[df.method == selected_method]
    #selected_perspective = 'nonsecular' #### so far this is requrired. i must not be getting this variable in
    filtered_df = df[(df.method == selected_method) & (df.perspective == selected_perspective)]

    fig = px.sunburst(
    filtered_df,
    path=["topic", 'method', 'style', 'perspective', 'colorformat', 'scriptformat', "name"],
    color="perspective",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    maxdepth=-1,)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
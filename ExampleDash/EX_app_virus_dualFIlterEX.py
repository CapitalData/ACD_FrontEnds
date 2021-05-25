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

import pandas as pd
from io import StringIO
import requests

#shutil.rmtree.avoids_symlink_attacks=True
from datetime import datetime, date, time
import csv

# image rendering 
import holoviews as hv
import dask.dataframe as dd
from holoviews.operation.datashader import (
    datashade, aggregate, dynspread,
    bundle_graph, split_dataframe,  regrid#, stack
)
from holoviews.element.graphs import layout_nodes
from datashader.layout import forceatlas2_layout, random_layout
hv.extension('bokeh')


verbose = False
workflow = 'pullfromlocal' #pullfromgit



###################################################
########### Methods ###############################
###################################################


def report_values(URI, KEYSPACE, query_in, ret_var, attribute):
    """return a list from a match query, reports on one or more attrubuted for the type listed"""

    t1=time.time()
    lookup = dict(zip(ret_var, attribute))

    with Grakn.core_client(URI) as client:
        with client.session(KEYSPACE, SessionType.DATA) as session:
            #with session.transaction(TransactionType.WRITE) as tx:
            with session.transaction(TransactionType.READ) as tx:
                iterator=tx.query().match(query_in)
                a=(list(iterator))
                #answers = [ans.get('virus-name') for ans in iterator]
                #result = [ answer.value() for answer in answers ]
                live_df = pd.DataFrame()
                
                # this pulles attributes out of the answer object
                for answer in a:
                    row={lookup[i]: [answer.get(i).get_value()] for i in ret_var}
                    live_df = live_df.append(pd.DataFrame(row))
                live_df.reset_index(drop=True, inplace=True)
   
    print(f"elapsed time {time.time()-t1}")            
    return live_df 

## HOW MANY VIRUSES ARE THERE IN THE DB
#there are 15 viruses in the database they appear to be relatives of covid.

def count_values(URI, KEYSPACE, type, limit):
    """ count_values(URI, KEYSPACE, virus, 10) """

    with Grakn.core_client(URI) as client:
        with client.session(KEYSPACE, SessionType.DATA) as session:
            #with session.transaction(TransactionType.WRITE) as tx:
            with session.transaction(TransactionType.READ) as tx:
                query_in=f"match $d isa {type}; limit {limit}; count;"
                ans=tx.query().match_aggregate(query_in)
                print(ans.get().as_int())

def get_edges(src, tgt, nodes=[]):
    """takes two series source and target and makes an edge for each row, 
    node name is translated to unique node index from node series, if no node list is passed then     the source and target take on the strings from the original file
    source and target must be in register and the same length"""
    rowct=len(src)
    edges = pd.DataFrame(np.zeros([rowct,2]), columns=['source', 'target'])
    for i in range(rowct):
        #print (f'{i} rows, adding {tgt[i]} and {src[i]}')
        #idx=pd.Index(nodes).get_loc(tgt[i])
        if len(nodes) == 0:
            edges.iloc[i,1]=tgt.iloc[i,0]
            edges.iloc[i,0]=src.iloc[i,0]
        else: 
            edges.iloc[i,1]=int(pd.Index(nodes).get_loc(tgt.iloc[i,0]))
            edges.iloc[i,0]=int(pd.Index(nodes).get_loc(src.iloc[i,0]))
    return edges

def split_edges(graph):
    paths = split_dataframe(graph.edgepaths.data[0])
    return graph.clone((graph.data, graph.nodes, paths))

def join_edges(graph):
    return graph.clone((graph.data, graph.nodes, [graph.edgepaths.array()]))


###################################################
########### Data handling #########################
###################################################

if workflow == 'pullfromgit':
## get the locally stored credentials


    cred_Dict = {}
    with open('keys/github_config.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            #skipping comments
            if rows[0][0]!='#': 
                cred_Dict[rows[0]]=rows[1]

    # this token for the raw data seems to expire? check that if things are not working
    token =  cred_Dict['token']
    ################ Get Data #############################

    #url_rv1a = 'https://github.com/CapitalData/biograkn-covid/raw/master/Dataset/Coronaviruses/GenomeIdentityClean.csv'
    url_rv1a = 'https://raw.githubusercontent.com/CapitalData/virusgraph/master/Dataset/Coronaviruses/GenomeIdentityClean.csv?token={token}'

    rv = requests.get(url_rv1a).text
    GenomeID = pd.read_csv(StringIO(rv), error_bad_lines=False)

    url_rv2 = 'https://github.com/CapitalData/biograkn-covid/raw/master/Dataset/Coronaviruses/Host proteins (potential drug targets).csv'

    rv_2 = requests.get(url_rv2).text
    VirusProtein = pd.read_csv(StringIO(rv_2), error_bad_lines=False)

elif workflow == 'pullfromlocal':
#**********************************************
# ***** Get the saved data. from local *****
## in this alternative workflow the data is pulled using Pull_data.ipynb from the graph then saved locally 

    VirAttrib = pd.read_csv('virus_attributes.csv', error_bad_lines=False) #not for graph?
    vir_prot_vir_rt= pd.read_csv('vir_prot_vir_rt.csv', error_bad_lines=False)
    #vir_prot_vir_rt

    if verbose == True:
        #print(GenomeID.head())
        #print(VirusProtein.head())
        print(VirAttrib.head())
        print(vir_prot_vir_rt.head())

# build the graph by nodes and edges
### concatenate and the nodes list with labels.
df1 = pd.DataFrame(pd.Series(vir_prot_vir_rt['virus_name1']))
df1.rename(columns={df1.columns[0]:'entity_name'}, inplace=True)

df2 = pd.DataFrame(pd.Series(vir_prot_vir_rt['uniprot_entry_name']))
df2.rename(columns={df2.columns[0]:'entity_name'}, inplace=True)

df3 = pd.DataFrame(pd.Series(vir_prot_vir_rt['virus_name2']))
df3.rename(columns={df3.columns[0]:'entity_name'}, inplace=True)

frames = [df1, df2, df3]
nodes_all=pd.Series(np.unique(pd.concat(frames)))

# with index number labels
edges_1= get_edges(src=df1, tgt=df2, nodes=nodes_all)
edges_2= get_edges(src=df2, tgt=df3, nodes=nodes_all)
edges_all=pd.concat([edges_1,edges_2])

## with string labels
edges_all_label=pd.concat([get_edges(src=df1, tgt=df2),get_edges(src=df2, tgt=df3)])

## add lables ###
df1['cat'] = 'virus'
df2['cat'] = 'protein'
df3['cat']= 'virus'
frames = [df1, df2, df3]

nodes_all_labels=pd.concat(frames).drop_duplicates()
nodes_all_labels.reset_index(inplace=True, drop=True)



########################################################
#############launch the application ####################
########################################################

app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
available_indicators = df['Indicator Name'].unique()
#############################


##############################
##### initial layout #########
##############################

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
                                            "Virus Graph",
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
    # html.Div(
    #     className="container",
    #     children=[
    #         html.Div(
    #             className="row",
    #             children=[
    #                 html.Div(
    #                     className="col-lg",
    #                     children=[
    #                         html.Div(
    #                             className="card my-3",
    #                             children=[
    #                                 html.Div(
    #                                     className="card-body",
    #                                     children=[
    #                                         html.H5("A graph-based approach to home school curriculum", className="card-title"),
    #                                     ]
    #                                 )
    #                             ]
    #                         )
    #                     ]
    #                 ),
    #                 html.Div(
    #                     className="col-lg",
    #                     children=[
    #                         html.Div(
    #                             className="card my-3",
    #                             children=[
    #                                 html.Div(
    #                                     className="card-body",
    #                                     children=[
    #                                         html.H5("About", className="card-title"),
    #                                     ]
    #                                 )
    #                             ]
    #                         )
    #                     ]
    #                 )
    #             ]
    #         ),
    #     ]
    # ),



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
                                            html.Label(["Filter by the host orgamism",
                                            dcc.Dropdown(
                                                id='methodform',
                                                options=[
                                                    {'label': 'all', 'value': 'All'},
                                                    {'label': 'human', 'value': 'Human'},
                                                    {'label': 'mouse', 'value': 'Mouse'},
                                                    {'label': 'avian', 'value': 'Avian'}],
                                                    value='Human',
                                            ),
                                            ]),
                                            html.Label(["Filter by the country of origin",
                                            dcc.Dropdown(
                                                id='styleform',
                                                options=[
                                                    {'label': 'All', 'value': 'All'},
                                                    {'label': 'USA', 'value': 'USA'},
                                                    {'label': 'China', 'value': 'China'}],
                                                    value='All',
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
#########
    
    html.Div([
        dcc.Graph(
            id='example-graph',
            #hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    
#######

    html.Div([
        dcc.Graph(
            id='example-graph2',
            #hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Fertility rate, total (births per woman)'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Life expectancy at birth, total (years)'
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),
    

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Slider(
        id='crossfilter-year--slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=df['Year'].max(),
        marks={str(year): str(year) for year in df['Year'].unique()},
        step=None
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'}) 

])

#########################
##### Callbacks #########
#########################

#Index(['GenBank ID ', 'Coronavirus ', 'Identity % ', 'Host ',
#       'Location discovered '],
#      dtype='object')
#Index(['Coronavirus ', 'Alternative virus name', 'Host Protein ', 'UniProt ID',
#       'Host Gene Entrez ID ', 'PubMed ID/STRING'],

@app.callback(
    Output('example-graph', 'figure'),
    Input('methodform', 'value'),
    Input('styleform', 'value')
    )
def update_figure(selected_method, selected_perspective): #, )
    df = VirAttrib
    #########
    # filter the dataframe and include potental All (no filter)
    if selected_perspective  =='All' and selected_method  =='All':
        filtered_df = df
    elif selected_perspective == 'All':
        filtered_df = df[(df.organism_name == selected_method)]
    elif selected_method == 'All':
        filtered_df = df[(df.country_name == selected_perspective)]
    else:
        filtered_df = df[(df.organism_name == selected_method) & (df.country_name == selected_perspective)]
    #########
    if verbose == True: 
        print(filtered_df.head())

    fig = px.sunburst(
    filtered_df,
    
    path=["virus_name", 'genbank_id', 'identity_percentage', 'organism_name', "country_name"],
    color="organism_name",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    maxdepth=-1,)

    return fig

##############
@app.callback(
    Output('example-graph2', 'figure'),
    Input('methodform', 'value'),
    Input('styleform', 'value')
    )
def update_figure(selected_method, selected_perspective): #, )
    df = VirAttrib
    #########
    # filter the dataframe and include potental All (no filter)
    if selected_perspective  =='All' and selected_method  =='All':
        filtered_df = df
    elif selected_perspective == 'All':
        filtered_df = df[(df.organism_name == selected_method)]
    elif selected_method == 'All':
        filtered_df = df[(df.country_name == selected_perspective)]
    else:
        filtered_df = df[(df.organism_name == selected_method) & (df.country_name == selected_perspective)]
    #########
    if verbose == True: 
        print(filtered_df.head())

    fig = px.sunburst(
    filtered_df,
    
    path=["virus_name", 'genbank_id', 'identity_percentage', 'organism_name', "country_name"],
    color="organism_name",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    maxdepth=-1,)

    return fig

# #magic for notebook?
# #%opts Graph Nodes [bgcolor='black' width=800 height=800 xaxis=None yaxis=None]
# # %%opts Nodes (size=5)
# graph = layout_nodes(hv.Graph(edges_all), layout=forceatlas2_layout)
# forceatlas = bundle_graph(graph, split=False)

# pad = dict(x=(-.5, 1.3), y=(-.5, 1.3))
# datashade(forceatlas, width=800, height=800) * forceatlas.nodes.redim.range(**pad)

###############

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')
     #,
    # dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
    # dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
    # dash.dependencies.Input('crossfilter-year--slider', 'value')
    ])

def update_graph(xaxis_column_name, yaxis_column_name): 
        #,
        #         xaxis_type, yaxis_type,
        #         year_value):
    year_value = 2002
    dff = df[df['Year'] == year_value]

    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
            )

    fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])
    #fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
    #fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig

def create_time_series(dff, axis_type, title):
    fig = px.scatter(dff, x='Year', y='Value')
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=title)
    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig

@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['Country Name'] == country_name]
    dff = dff[dff['Indicator Name'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)

@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['Indicator Name'] == yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name)

if __name__ == '__main__':
    app.run_server(debug=True)
# %%

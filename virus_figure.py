#virus_figure.py
# conda activate plotlyenv2

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

## We can use requests and get a byte stream of the CSVget virus data
## dropping lines with extra data

#############################################
## get the locally stored credentials
cred_Dict = {}
with open('keys/github_config.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        #skipping comments
        if rows[0][0]!='#': 
            cred_Dict[rows[0]]=rows[1]

token =  cred_Dict['token']
################ Get Data #############################

#url_rv1a = f'https://raw.githubusercontent.com/CapitalData/virusgraph/master/Dataset/Coronaviruses/GenomeIdentityClean.csv?token={token}'
#url_rv1a =  f'https://github.com/CapitalData/virusgraph/raw/master/notebooks/vir_prot_vir_rt.csv'
url_rv1a =  f'https://github.com/CapitalData/virusgraph/raw/master/notebooks/virus_attributes.csv'
rv = requests.get(url_rv1a).text
VirAttrib = pd.read_csv(StringIO(rv), error_bad_lines=False)

#url_rv2 = 'https://github.com/CapitalData/biograkn-covid/raw/master/Dataset/Coronaviruses/Host proteins (potential drug targets).csv'
url_rv2 =  f'https://github.com/CapitalData/virusgraph/raw/master/notebooks/vir_prot_vir_rt.csv'

rv_2 = requests.get(url_rv2).text
vir_prot_vir_rt = pd.read_csv(StringIO(rv_2), error_bad_lines=False)

app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
available_indicators = df['Indicator Name'].unique()

#############################

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

print('debug point')
Makegraph1.py


from grakn.client import *
import pandas as pd
import time
import collections
import numpy as np

import holoviews as hv
import dask.dataframe as dd
from holoviews.operation.datashader import (
    datashade, aggregate, dynspread,
    bundle_graph, split_dataframe,  regrid#, stack
)
from holoviews.element.graphs import layout_nodes
from datashader.layout import forceatlas2_layout, random_layout


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


# Get the saved data.
VirAttrib = pd.read_csv('virus_attributes.csv', error_bad_lines=False) #not for graph?
vir_prot_vir_rt= pd.read_csv('vir_prot_vir_rt.csv', error_bad_lines=False)


### parsing code 

# build the graph by nodes and edges
### concatenate and the nodes list with labels.
df1 = pd.DataFrame(pd.Series(vir_prot_vir_rt['virus-name1']))
df1.rename(columns={df1.columns[0]:'entity_name'}, inplace=True)

df2 = pd.DataFrame(pd.Series(vir_prot_vir_rt['uniprot-entry-name']))
df2.rename(columns={df2.columns[0]:'entity_name'}, inplace=True)

df3 = pd.DataFrame(pd.Series(vir_prot_vir_rt['virus-name2']))
df3.rename(columns={df3.columns[0]:'entity_name'}, inplace=True)

frames = [df1, df2, df3]
nodes_all=pd.Series(np.unique(pd.concat(frames)))

# with index number labels
edges_1= get_edges(src=df1, tgt=df2, nodes=nodes_all)
edges_2= get_edges(src=df2, tgt=df3, nodes=nodes_all)
edges_all=pd.concat([edges_1,edges_2])

## with string labels
edges_all_label=pd.concat([get_edges(src=df1, tgt=df2),get_edges(src=df2, tgt=df3)])
edges_all_label.columns=['source_nm', 'target_nm']

## add lables ###
df1['cat'] = 'virus'
df2['cat'] = 'protein'
df3['cat']= 'virus'
frames = [df1, df2, df3]

nodes_all_labels=pd.concat(frames).drop_duplicates()
nodes_all_labels.reset_index(inplace=True, drop=True)

print(nodes_all_labels.head(), '\n')

print(nodes_all.head(), '\n')

print(len(edges_all_label))
print(edges_all_label.head())


## Do the graph visualization
hv.extension('bokeh')
%opts Graph Nodes [bgcolor='black' width=800 height=800 xaxis=None yaxis=None]
%opts Nodes (size=5)

def split_edges(graph):
    paths = split_dataframe(graph.edgepaths.data[0])
    return graph.clone((graph.data, graph.nodes, paths))

def join_edges(graph):
    return graph.clone((graph.data, graph.nodes, [graph.edgepaths.array()]))


graph = layout_nodes(hv.Graph(edges_all), layout=forceatlas2_layout)
#nodes
forceatlas = bundle_graph(graph, split=False)

#pad = dict(x=(-.5, 1.3), y=(-.5, 1.3))
#datashade(forceatlas, width=800, height=800) * forceatlas.nodes.redim.range(**pad)

# annotate the indexes
forceatlas2=forceatlas
#forceatlas2._nodes.data.iloc[0,2] = 'test'
forceatlas2._nodes.data.set_index(np.zeros(len(forceatlas._nodes.data)),inplace=True)
forceatlas2._nodes.data.set_index(nodes_all,inplace=True)
forceatlas2._nodes.data.reset_index(inplace=True)

# add data names for the edges at top level
# print(len(forceatlas2.data), len(edges_all_label))
frames= [forceatlas2.data, edges_all_label]
forceatlas2.data = pd.concat(frames, axis=1)

#G = layout_nodes(G, layout=nx.layout.fruchterman_reingold_layout, kwargs={'weight': 'weight'})
#nodes_array = G.nodes.array()
#newNodes_array = []
#for node in nodes_array:
#    newNodes_array.append((node[0], node[1], node[2], random.choice(['class1', 'class2', 'class3'])))
#N = hv.Nodes(newNodes_array, vdims='class')

%opts Graph (edge_alpha=0 edge_hover_alpha=1 edge_nonselection_alpha=0 node_size=8 node_alpha=0.5) [inspection_policy='nodes']

split_atlas = split_edges(forceatlas)
udp_style = dict(edge_hover_line_color='red', node_fill_color='red', edge_selection_line_color='red')
tcp_style = dict(edge_hover_line_color='blue', node_fill_color='blue', edge_selection_line_color='blue')
udp = split_atlas.select(protocol='udp', weight=(1000, None)).opts(style=udp_style)
tcp = split_atlas.select(protocol='tcp', weight=(1000, None)).opts(style=tcp_style)
fig = datashade(forceatlas2, width=450, height=450, normalization='log', cmap=['grey', 'white']) * tcp * udp  

#points = hv.Points(df, ['x','y'])
#labels = hv.Labels(df, ['x','y'], 'label')

#datashade(points) * decimate(labels)
fig


hv.save(fig, 'out3.html')
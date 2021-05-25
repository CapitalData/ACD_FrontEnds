#!/usr/bin/env python
# coding: utf-8

# In[60]:


from grakn.client import *
import pandas as pd
import time
import collections
import numpy as np

# The purpose of this script is to create a report of which elements of a schema have data present.
KEYSPACE = 'biograkn_V2prod'
URI = "localhost:1729"


# In[61]:


def report_values(query_in, ret_var, attribute):
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


# In[65]:


# virus level data - attributes plus discovery-orign and organism
## get all the virus names in the notebook, add thier relatedness attribute0.
# -- funciton: get attribute list from type

attribute= ['virus-name', 'genbank-id', 'identity-percentage', 'organism-name', 'country-name' ] #, 
ret_var=['vn', 'gid','idprc', 'on', 'cn'] #,'gva'

query_in=f"""match $v isa virus, has virus-name $vn, has genbank-id $gid, has identity-percentage $idprc;
$ova(hosting-organism: $o, hosted-virus: $v) isa organism-virus-hosting;
$o isa organism, has organism-name $on;
$discv (discovered-virus: $v, discovering-location: $c) isa discovery;
$c isa country, has country-name $cn;
""" 

#$g isa gene, has gene-symbol $gs, has entrez-id $entid;
#$gva(associated-virus-gene: $g, associated-virus: $v) isa gene-virus-association;


VirAttrib=report_values(query_in, ret_var, attribute)
print(VirAttrib.shape)

VirAttrib['organism-name'] = VirAttrib['organism-name'].replace(['Homo sapiens (Human)'],'Human')

### not sure why duplicates are coming back from this query
### also some of the original viruses are not in the data set?
## this is related to the requirement for the gene virus association
## removing that updates teh list

VirAttrib=VirAttrib.drop_duplicates()
print(VirAttrib.shape)
print(VirAttrib)

############### write out the CSVs #############
VirAttrib.to_csv('virus_attributes.csv', index=False)


# In[9]:


#if VirAttrib[organism-name] == Homo sapiens (Human)
#VirAttrib.where('Homo sapiens (Human)', 'Human', inplace=True)


# In[96]:


# round trips
# virus1 > protien1 > virus2. - neighborhood analysis

attribute= ['virus-name1', 'uniprot-name', 'function-description', 'uniprot-entry-name',  'virus-name2', 'genbank-id1', 'genbank-id2'] #, , 'pathway-name', 'pathway-id'
ret_var=['vn1', 'un', 'fdesc', 'unpn', 'vn2', 'gid1', 'gid2'] #,'gva' , 'pwn', 'pwid'

query_in=f"""
match $v1 isa virus, has genbank-id $gid1, has virus-name $vn1; 
$p isa protein, has uniprot-name $un, has function-description $fdesc, has uniprot-entry-name $unpn;
(hosting-virus-protein: $p, associated-virus: $v1) isa protein-virus-association; 
$v2 isa virus, has genbank-id $gid2, has virus-name $vn2;
(hosting-virus-protein: $p, associated-virus: $v2) isa protein-virus-association; 
"""

vir_prot_vir_rt=report_values(query_in, ret_var, attribute)
print(vir_prot_vir_rt.shape)

### THIS DOES NOT HAVE duplicates are coming back from this query
### maybe the round trips are reciprical
df=vir_prot_vir_rt.drop_duplicates()
print(df.shape)

# remove self referential paths - this might be a mistake since it will remove proteins that are onlu linked to one virus - we want all unique paths including terminal paths

#df=df[df['virus-name1'] != df['virus-name2']]
#print(f'dropped self referential paths, shape: {df.shape}')
###remove redundant relationships## i.e a->b == b->a

#get the set of viruses on each line and make a new colunm
#vir_prot_vir_rt.vs_set=
def virus_set(x):
    """ this takes in the whole dataframe, pass N columns in next iteration"""
    x['vs_set'] = set([x['genbank-id1'],x['uniprot-entry-name'], x['genbank-id2']])     
    x['vs_set_str'] = str(x['vs_set']) 
    return x

df = df.apply(virus_set, axis=1)
df = df.drop_duplicates('vs_set_str', keep='last')

# # subset by virus 1 and 2 (a or b) 
print(f'dropped reverse paths shape accounting for redundant IDs: {df.shape}')
df.head()

df.to_csv('vir_prot_vir_rt.csv', index=False)
vir_prot_vir_rt=df
vir_prot_vir_rt


# In[97]:


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

## add lables ###
df1['cat'] = 'virus'
df2['cat'] = 'protein'
df3['cat']= 'virus'
frames = [df1, df2, df3]

nodes_all_labels=pd.concat(frames).drop_duplicates()
nodes_all_labels.reset_index(inplace=True, drop=True)


# In[99]:


print(f'{vir_prot_vir_rt.shape}')

print(f'{df1.shape}')
print(f'{df2.shape}')
print(f'{df3.shape}')

print(f'{nodes_all_labels.shape}')
print(f'{edges_all.shape}')


# In[100]:


import holoviews as hv
import dask.dataframe as dd
from holoviews.operation.datashader import (
    datashade, aggregate, dynspread,
    bundle_graph, split_dataframe,  regrid#, stack
)
from holoviews.element.graphs import layout_nodes
from datashader.layout import forceatlas2_layout, random_layout

hv.extension('bokeh')

get_ipython().run_line_magic('opts', "Graph Nodes [bgcolor='black' width=800 height=800 xaxis=None yaxis=None]")

def split_edges(graph):
    paths = split_dataframe(graph.edgepaths.data[0])
    return graph.clone((graph.data, graph.nodes, paths))

def join_edges(graph):
    return graph.clone((graph.data, graph.nodes, [graph.edgepaths.array()]))


# In[108]:


graph.edgepaths.data[0]


# In[101]:


get_ipython().run_cell_magic('opts', 'Nodes (size=5)', 'graph = layout_nodes(hv.Graph(edges_all), layout=forceatlas2_layout)\nforceatlas = bundle_graph(graph, split=False)\n#pad = dict(x=(-.5, 1.3), y=(-.5, 1.3))\n#datashade(forceatlas, width=800, height=800) * forceatlas.nodes.redim.range(**pad)')


# In[103]:


#print(split_atlas.__dict__)
#print(split_atlas._edgepaths.data)
#print(split_atlas._nodes.data)
len(forceatlas._nodes.data)


# In[114]:


# annotate the indexes

forceatlas2=forceatlas
#forceatlas2._nodes.data.iloc[0,2] = 'test'
forceatlas2._nodes.data.set_index(np.zeros(len(forceatlas._nodes.data)),inplace=True)
forceatlas2._nodes.data.set_index(nodes_all,inplace=True) 
print(forceatlas2._nodes.data)
forceatlas2._nodes.data.to_csv('node_index.csv')


# In[112]:


(forceatlas2._edgepaths.data)


# In[115]:


##forceatlas2
#########plot matplolib on top ############
# import matplotlib.pyplot as plt
# import matplotlib.transforms as mtransforms
# import numpy as np
# %matplotlib inline   
# # the above line allows the plot to appear in the notebook

# xs = np.arange(7)
# ys = xs**2

# fig = plt.figure(figsize=(5, 10))
# ax = plt.subplot(2, 1, 1)

# # If we want the same offset for each text instance,
# # we only need to make one transform.  To get the
# # transform argument to offset_copy, we need to make the axes
# # first; the subplot command above is one way to do this.
# trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
#                                        x=0.05, y=0.10, units='inches')

# for x, y in zip(xs, ys):
#     plt.plot((x,), (y,), 'ro')
#     plt.text(x, y, '%d, %d' % (int(x), int(y)), transform=trans_offset)


# In[109]:


print(nodes_all)


# In[107]:


get_ipython().run_cell_magic('opts', "Graph (edge_alpha=0 edge_hover_alpha=1 edge_nonselection_alpha=0 node_size=8 node_alpha=0.5) [inspection_policy='edges']", "\nsplit_atlas = split_edges(forceatlas)\nudp_style = dict(edge_hover_line_color='red', node_fill_color='red', edge_selection_line_color='red')\ntcp_style = dict(edge_hover_line_color='blue', node_fill_color='blue', edge_selection_line_color='blue')\nudp = split_atlas.select(protocol='udp', weight=(1000, None)).opts(style=udp_style)\ntcp = split_atlas.select(protocol='tcp', weight=(1000, None)).opts(style=tcp_style)\ndatashade(forceatlas2, width=800, height=800, normalization='log', cmap=['grey', 'white']) * tcp * udp  \n\n#points = hv.Points(df, ['x','y'])\n#labels = hv.Labels(df, ['x','y'], 'label')\n\n#datashade(points) * decimate(labels)")


# In[62]:


split_atlas.__dict__.keys()


# In[57]:


split_atlas.__dict__.keys()


# In[ ]:





# In[116]:


udp


# In[117]:


tcp


# In[119]:


get_ipython().run_cell_magic('opts', 'Nodes (size=10)', 'circular = bundle_graph(hv.Graph(edges_all), split=False)\npad = dict(x=(-1.2, 1.2), y=(-1.2, 1.2))\ndatashade(circular, width=800, height=800) * circular.nodes.redim.range(**pad)\n#127 node indexes. ')


# In[121]:


get_ipython().run_cell_magic('opts', "Graph (edge_line_color='white' edge_hover_line_color='blue')", 'split_circular = split_edges(circular)\ndatashade(circular, width=800, height=800) * split_circular.select(weight=(10000, None)).redim.range(**pad)\n')


# In[ ]:





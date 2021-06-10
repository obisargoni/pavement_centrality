# Script to compare network centrality measures between centre line road network and pavement network

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx


###########################
#
#
# Globals
#
#
###########################

data_dir = ".\\data"
img_dir = ".\\img"

pavement_network_gis_file = os.path.join(data_dir, "pedNetworkLinks.shp")
pavement_nodes_gis_file = os.path.join(data_dir, "pedNetworkNodes.shp")
road_network_gis_file = os.path.join(data_dir, "open-roads RoadLink Intersect Within simplify angles.shp")

output_road_network = os.path.join(data_dir, "open-roads RoadLink betcen diffs.shp")

##########################
#
#
# Functions
#
#
##########################



##########################
#
#
# Load data
#
#
##########################

gdfPaveNodes = gpd.read_file(pavement_nodes_gis_file)
gdfPaveLinks = gpd.read_file(pavement_network_gis_file)
gdfORLinks = gpd.read_file(road_network_gis_file)

gdfPaveLinks['length'] = gdfPaveLinks['geometry'].length
gdfORLinks['length'] = gdfORLinks['geometry'].length

edges_pavement = gdfPaveLinks.loc[:, ['MNodeFID', 'PNodeFID', 'fid', 'length']]
edges_road = gdfORLinks.loc[:, ['MNodeFID', 'PNodeFID', 'fid', 'length']]

g_pavement = nx.from_pandas_edgelist(edges_pavement, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)
g_road = nx.from_pandas_edgelist(edges_road, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)




########################
#
#
# Do centrality analysis
#
#
########################

pave_betcen = nx.edge_betweenness_centrality(g_pavement, normalized = True, weight='length')
road_betcen = nx.edge_betweenness_centrality(g_road, normalized = True, weight='length')


# Create lookup from pavement links to corresponding or road link
dfPaveNodes = gdfPaveNodes.reindex(columns=['fid', 'juncNodeID'])
dfLinks = pd.merge(gdfPaveLinks, dfPaveNodes, left_on = 'MNodeFID', right_on = 'fid', how = 'left')
dfLinks = dfLinks.rename(columns = {'juncNodeID':'juncNodeID_M', 'fid_x':'fid'}).drop('fid_y', axis=1)
dfLinks = pd.merge(dfLinks, gdfPaveNodes, left_on = 'PNodeFID', right_on = 'fid', how = 'left')
dfLinks = dfLinks.rename(columns = {'juncNodeID':'juncNodeID_P', 'fid_x':'fid'}).drop('fid_y', axis=1)

dfLinks = dfLinks.reindex(columns = ['MNodeFID', 'PNodeFID', 'fid', 'pedRLID', 'juncNodeID_M', 'juncNodeID_P'])
dfORLinks = gdfORLinks.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID']).rename(columns = {'fid':'pedRLID'})

dfLinks['pedRLCross'] = dfLinks['pedRLID']

# First select ped links that already have a road link assocaited
dfLinksA = dfLinks.loc[ ~dfLinks['pedRLID'].isnull()]

# Then find associated road link for links that dont have road link
dfLinksB = dfLinks.loc[ dfLinks['pedRLID'].isnull()].drop('pedRLID', axis=1)

dfLinksB1 = pd.merge(dfLinksB, dfORLinks, left_on = ['juncNodeID_M', 'juncNodeID_P'], right_on = ['MNodeFID', 'PNodeFID'], how='inner')
dfLinksB2 = pd.merge(dfLinksB, dfORLinks, left_on = ['juncNodeID_M', 'juncNodeID_P'], right_on = ['PNodeFID', 'MNodeFID'], how='inner')

dfLinksBFull = pd.concat([dfLinksB1, dfLinksB2], join='inner')

assert dfLinksBFull.shape[0] == dfLinksB.shape[0]

dfBetween = pd.concat([dfLinksA, dfLinksBFull])

# Now use lookup from pavement link to or link to compare betweenness centrality measures

# First convert the key from nodes to edge ids
edge_pave_betcen = {}
for k, v in pave_betcen.items():
	edge_id = g_pavement[k[0]][k[1]]['fid']
	edge_pave_betcen[edge_id] = v

edge_road_betcen = {}
for k, v in road_betcen.items():
	edge_id = g_road[k[0]][k[1]]['fid']
	edge_road_betcen[edge_id] = v

dfBetween['pave_link_betcen'] = dfBetween['fid'].replace(edge_pave_betcen)
dfBetween['or_link_betcen'] = dfBetween['pedRLID'].replace(edge_road_betcen)

# Save the data
dfBetween.to_csv("link_betcens.csv", index=False)


# Calculate difference between sides of the road - non crossing links associated to each road link
def betcen_diff(df, pave_betcen_col = 'pave_link_betcen'):
	if df.shape[0] != 2:
		print(df)
		return

	diff = (df[pave_betcen_col].values[0] - df[pave_betcen_col].values[1]) / df[pave_betcen_col].values[1]
	return diff

dfBetweenNoCross = dfBetween.loc[dfBetween['pedRLCross'].isnull()]
road_betcen_diffs = dfBetweenNoCross.groupby('pedRLID').apply(betcen_diff)
road_betcen_diffs.name = 'betcen_diff'

# Add back to the road link data and save
dfBetCenDiffs = pd.DataFrame(road_betcen_diffs).reset_index()

gdfORLinks = pd.merge(gdfORLinks, dfBetCenDiffs, left_on = 'fid', right_on = 'pedRLID', how='left')
gdfORLinks.to_file(output_road_network)

################################
#
#
# Viualise
#
#
###############################


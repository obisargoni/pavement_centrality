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

output_links_lookup = os.path.join(data_dir, "pavement_links_to_or_links.csv")
output_or_roads = os.path.join(data_dir, "open_roads_clean.shp")

output_road_network = os.path.join(data_dir, "open-roads RoadLink betcen diffs")
output_pave_links = os.path.join(data_dir, "pednetworkLinksWithCentralities")
output_pave_ex_diag_links = os.path.join(data_dir, "pednetworkLinksExDiagWithCentralities")

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

gdfORLinks = gpd.read_file(output_or_roads)
gdfORLinks = gdfORLinks.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID', 'geometry', 'length'])


gdfPaveLinks['length'] = gdfPaveLinks['geometry'].length

gdfORLinks = gdfORLinks.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID', 'geometry'])
gdfORLinks['length'] = gdfORLinks['geometry'].length

edges_pavement_ex_diag = gdfPaveLinks.loc[ gdfPaveLinks['linkType']!='diag_cross', ['MNodeFID', 'PNodeFID', 'fid', 'length']]
edges_pavement = gdfPaveLinks.loc[:, ['MNodeFID', 'PNodeFID', 'fid', 'length']]
edges_road = gdfORLinks.loc[:, ['MNodeFID', 'PNodeFID', 'fid', 'length']]

g_pavement_ex_diag = nx.from_pandas_edgelist(edges_pavement_ex_diag, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)
g_pavement = nx.from_pandas_edgelist(edges_pavement, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)
g_road = nx.from_pandas_edgelist(edges_road, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)

########################
#
#
# Create lookup between pavement links and road links
#
#
########################

# Create lookup from pavement links to corresponding or road link
'''
dfPaveNodes = gdfPaveNodes.reindex(columns=['fid', 'juncNodeID'])
dfLinks = pd.merge(gdfPaveLinks, dfPaveNodes, left_on = 'MNodeFID', right_on = 'fid', how = 'left')
dfLinks = dfLinks.rename(columns = {'juncNodeID':'juncNodeID_M', 'fid_x':'fid'}).drop('fid_y', axis=1)
dfLinks = pd.merge(dfLinks, gdfPaveNodes, left_on = 'PNodeFID', right_on = 'fid', how = 'left')
dfLinks = dfLinks.rename(columns = {'juncNodeID':'juncNodeID_P', 'fid_x':'fid'}).drop('fid_y', axis=1)

dfLinks = dfLinks.reindex(columns = ['MNodeFID', 'PNodeFID', 'fid', 'pedRLID', 'juncNodeID_M', 'juncNodeID_P'])
dfORLinks = gdfORLinks.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID']).rename(columns = {'fid':'pedRLID'})

dfLinks['or_link_cross'] = dfLinks['pedRLID']

# First select ped links that already have a road link assocaited
dfLinksA = dfLinks.loc[ ~dfLinks['pedRLID'].isnull()]

# Then find associated road link for links that dont have road link
dfLinksB = dfLinks.loc[ dfLinks['pedRLID'].isnull()].drop('pedRLID', axis=1)

dfLinksB1 = pd.merge(dfLinksB, dfORLinks, left_on = ['juncNodeID_M', 'juncNodeID_P'], right_on = ['MNodeFID', 'PNodeFID'], how='inner')
dfLinksB2 = pd.merge(dfLinksB, dfORLinks, left_on = ['juncNodeID_M', 'juncNodeID_P'], right_on = ['PNodeFID', 'MNodeFID'], how='inner')

dfLinksBFull = pd.concat([dfLinksB1, dfLinksB2], join='inner')
dfLinksBFull.drop(['MNodeFID_y', 'PNodeFID_y'], axis=1, inplace=True)
dfLinksBFull.rename(columns = {'MNodeFID_x':'MNodeFID', 'PNodeFID_x':'PNodeFID'}, inplace=True)

# Duplcated entries in dfLinksBFull caused by OR Links having multi links. Not sure how these have krept in, but remove now
or_links_to_remove = dfLinksBFull.loc[ dfLinksBFull['fid'].duplicated(), 'pedRLID'].values
dfLinksBFull = dfLinksBFull.loc[ ~dfLinksBFull['pedRLID'].isin(or_links_to_remove)]
gdfORLinks = gdfORLinks.loc[~gdfORLinks['fid'].isin(or_links_to_remove)]

assert dfLinksBFull.shape[0] == dfLinksB.shape[0]

dfLinksLookup = pd.concat([dfLinksA, dfLinksBFull])

dfLinksLookup.rename(columns={'pedRLID':'or_fid'}, inplace=True)
dfLinksLookup.to_csv(output_links_lookup, index=False)

gdfORLinks.to_file(output_or_roads)

gdfPaveLinks.to_file(pavement_network_gis_file)
'''

dfLinksLookup = pd.read_csv(output_links_lookup)

########################
#
#
# Calculate centralities
#
#
########################

pave_ex_diag_betcen = nx.edge_betweenness_centrality(g_pavement_ex_diag, normalized = True, weight='length')
pave_betcen = nx.edge_betweenness_centrality(g_pavement, normalized = True, weight='length')
road_betcen = nx.edge_betweenness_centrality(g_road, normalized = True, weight='length')

# Now use lookup from pavement link to or link to compare betweenness centrality measures

# First convert the key from nodes to edge ids
pave_betcen_ex_diag = {'fid':[], 'paveExDBC':[]}
for k, v in pave_ex_diag_betcen.items():
	edge_id = g_pavement_ex_diag[k[0]][k[1]]['fid']
	pave_betcen_ex_diag['fid'].append(edge_id)
	pave_betcen_ex_diag['paveExDBC'].append(v)

paveBC = {'fid':[], 'paveBC':[]}
for k, v in pave_betcen.items():
	edge_id = g_pavement[k[0]][k[1]]['fid']
	paveBC['fid'].append(edge_id)
	paveBC['paveBC'].append(v)

roadBC = {'or_fid':[], 'roadBC':[]}
for k, v in road_betcen.items():
	edge_id = g_road[k[0]][k[1]]['fid']
	roadBC['or_fid'].append(edge_id)
	roadBC['roadBC'].append(v)

dfPaveBCExDiag = pd.DataFrame(pave_betcen_ex_diag)
dfPaveBC = pd.DataFrame(paveBC)
dfORBC = pd.DataFrame(roadBC)

dfLinksLookup = pd.merge(dfLinksLookup, dfPaveBCExDiag, left_on = 'fid', right_on = 'fid', how = 'left')
dfLinksLookup = pd.merge(dfLinksLookup, dfPaveBC, left_on = 'fid', right_on = 'fid', how = 'left')
dfLinksLookup = pd.merge(dfLinksLookup, dfORBC, left_on = 'or_fid', right_on = 'or_fid', how = 'left')

# Calculate diff etween OR link and pave link betcen, useful check
dfLinksLookup['rdPvDiff'] = abs(dfLinksLookup['roadBC'] - dfLinksLookup['paveBC']) / dfLinksLookup['roadBC']
dfLinksLookup['rdPvExDDif'] = abs(dfLinksLookup['roadBC'] - dfLinksLookup['paveExDBC']) / dfLinksLookup['roadBC']


# Save the data
dfLinksLookup.to_csv("link_betcens.csv", index=False)

# Calculate difference between sides of the road - non crossing links associated to each road link
def betcen_diff(df, pave_betcen_col = 'paveBC'):
	# Expect two non crossing links, if not don't calculate value
	if df.shape[0] != 2:
		print(df)
		return

	diff = (df[pave_betcen_col].values[0] - df[pave_betcen_col].values[1]) / df[pave_betcen_col].values[1]
	return diff

dfBetweenNoCross = dfLinksLookup.loc[dfLinksLookup['or_link_cross'].isnull()]

road_betcen_diffs_ex_diag = dfBetweenNoCross.groupby('or_fid').apply(betcen_diff, pave_betcen_col = 'paveExDBC')
road_betcen_diffs_ex_diag.name = 'BCDiffExDi'

road_betcen_diffs = dfBetweenNoCross.groupby('or_fid').apply(betcen_diff, pave_betcen_col = 'paveBC')
road_betcen_diffs.name = 'BCDiff'

# Add back to the road link data and save
dfBetCenDiffsExDiag = pd.DataFrame(road_betcen_diffs_ex_diag).reset_index()
dfBetCenDiffs = pd.DataFrame(road_betcen_diffs).reset_index()
dfORBetCen = dfLinksLookup.loc[:, ['or_fid', 'roadBC']].drop_duplicates()

gdfORLinks = pd.merge(gdfORLinks, dfBetCenDiffsExDiag, left_on = 'fid', right_on = 'or_fid', how='left')
gdfORLinks = pd.merge(gdfORLinks, dfBetCenDiffs, left_on = 'fid', right_on = 'or_fid', how='left')
gdfORLinks = pd.merge(gdfORLinks, dfORBetCen, left_on = 'fid', right_on = 'or_fid', how='left')
gdfORLinks.to_file(output_road_network)


dfPaveBC = dfLinksLookup.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveBC', 'roadBC', 'rdPvDiff'])
dfPaveBCExDiag = dfLinksLookup.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveExDBC', 'roadBC', 'rdPvExDDif']).dropna(axis=0, subset = ['paveExDBC'])


gdfPaveLinksWBC = pd.merge(gdfPaveLinks, dfPaveBC, left_on = 'fid', right_on = 'fid', how='inner')
gdfPaveLinksExDiagWBC = pd.merge(gdfPaveLinks, dfPaveBCExDiag, left_on = 'fid', right_on = 'fid', how='inner')
gdfPaveLinksWBC.to_file(output_pave_links)
gdfPaveLinksExDiagWBC.to_file(output_pave_ex_diag_links)
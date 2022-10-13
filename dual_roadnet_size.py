# script for finding the maximum path distance of dual road networks
# used to assess validity of upper bound of PH parameter

import re
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

import sys
#sys.path.append("C:\\Anaconda3\\Lib\\site-packages")
#sys.path.append("C:\\Anaconda3\\envs\\cityimage\\Lib\\site-packages")
sys.path.append("C:\\Users\\obisargoni\\Documents\\CASA\\cityImage")

from cityImage import dual_gdf, dual_graph_fromGDF

def get_dual_network(gdfJunctions, gdfLinks, cols_rename_dict, epsg = 27700):
	'''Formats links geodataframe to be compatible with the cityImage library. Then use cityImage functions to convert to dual representation
	'''
	gdfLinks = gdfLinks.rename(columns = cols_rename_dict)

	gdfNodes_dual, gdfEdges_dual = dual_gdf(gdfJunctions, gdfLinks, epsg, oneway = False, angle = 'degree')

	return gdfNodes_dual, gdfEdges_dual

def convert_dual_ids_to_ints(nodes_df, edges_df, network_type):

	str_replace = None
	regex = None
	if network_type == 'road':
		str_replace = "or_node_"
		regex = r'or_link_(\d*)'
	elif network_type == 'quad_road':
		str_replace = "node_"
		regex = r'quad_grid_(\d*)'
	else:
		str_replace = "pave_node_"
		regex = r'pave_link_(\d*)_(\d*)'

	# Convert node ids
	nodes_df = nodes_df.rename(columns = {'u':'u_orig', 'v':'v_orig', 'edgeID':'edgeID_orig'})
	edges_df = edges_df.rename(columns = {'u':'u_orig', 'v':'v_orig'})

	nodes_df['u'] = nodes_df['u_orig'].str.replace(str_replace, "")
	nodes_df['v'] = nodes_df['v_orig'].str.replace(str_replace, "")

	nodes_df['edgeID'] = nodes_df['edgeID_orig'].apply(lambda str_id: edge_id_to_int(str_id, regex))

	edges_df['u'] = edges_df['u_orig'].apply(lambda str_id: edge_id_to_int(str_id, regex))
	edges_df['v'] = edges_df['v_orig'].apply(lambda str_id: edge_id_to_int(str_id, regex))

	return nodes_df, edges_df

def edge_id_to_int(edge_id, regex):
	edge_regex = re.compile(regex)
	res = edge_regex.search(edge_id)
	if res is None:
		print(edge_id)
		return None

	new_id = ''
	for str_node_numer in res.groups():
		while len(str_node_numer)<4:
			str_node_numer = '0'+str_node_numer
		str_node_numer = '1'+str_node_numer
		new_id+=str_node_numer
	return int(new_id)



env_name = ""
data_dir = "S:\\CASA_obits_ucfnoth\\1. PhD Work\\GIS Data\\{}\\processed_gis_data".format(env_name)
img_dir = ".\\img"

gb_epsg = 27700

road_network_gis_file = os.path.join(data_dir, "open-roads RoadLink Intersect Within simplify angles.shp")
road_nodes_gis_file = os.path.join(data_dir, "open-roads RoadNode Intersect Within simplify angles.shp")

gdfORNodes = gpd.read_file(road_nodes_gis_file)
gdfORLinks = gpd.read_file(road_network_gis_file)
gdfORLinks = gdfORLinks.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID', 'class', 'geometry', 'length'])
gdfORLinks['length'] = gdfORLinks['geometry'].length

# create graph
edges_road = gdfORLinks.loc[:, ['MNodeFID', 'PNodeFID', 'fid', 'length']]
g_road = nx.from_pandas_edgelist(edges_road, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)


#
# Create dual networks
#
cols_rename_dict = {'MNodeFID':'u', 'PNodeFID':'v', 'fid':'edgeID'}
gdfORNodesDual, gdfORLinksDual = get_dual_network(gdfORNodes, gdfORLinks, cols_rename_dict)

gdfORNodesDualClean, gdfORLinksDualClean = convert_dual_ids_to_ints(gdfORNodesDual, gdfORLinksDual, network_type = 'road')
g_road_dual = dual_graph_fromGDF(gdfORNodesDualClean, gdfORLinksDualClean)


#
# Find max path length
#


# select origin nodes
routes_file = "C:\\Users\\obisargoni\\eclipse-workspace\\repastInterSim\\output\\batch\\model_run_data\\processed_pedestrian_routes.2022.Oct.05.22_22_22.csv"
dfPedRoutes = pd.read_csv(routes_file)
dfPedRoutes['FullStrategicPathString'] = dfPedRoutes['FullStrategicPathString'].map(lambda s: tuple(s.strip("('").strip("')").strip("',").split("', '")))
dfPedRoutes['first_link'] = dfPedRoutes['FullStrategicPathString'].map(lambda x: x[0])

# Lookup from link id to dual node id
lu = gdfORNodesDualClean.set_index('edgeID_orig')['edgeID'].to_dict()

dfPedRoutes['dual_node_paths'] = dfPedRoutes['FullStrategicPathString'].map(lambda p: [lu[i] for i in p])
dfPedRoutes['path_deg_weight'] = dfPedRoutes['dual_node_paths'].map(lambda p: nx.path_weight(g_road_dual, p, 'deg'))

print(dfPedRoutes['path_deg_weight'].describe())

#
# Find max path length
#

dest_edge = 'or_link_186'
#dest_edge = 'quad_grid_8_212' # 'quad_grid_8_43'
dest_id = gdfORNodesDualClean.loc[gdfORNodesDualClean['edgeID_orig']==dest_edge, 'edgeID'].values[0]

#source_edges = dfPedRoutes['first_link'].unique()
#source_ids = gdfORNodesDualClean.loc[gdfORNodesDualClean['edgeID_orig'].isin(source_edges), 'edgeID'].tolist()
sp_lengths = nx.shortest_path_length(g_road_dual, source=None, target=dest_id, weight='deg', method='dijkstra')
df = pd.DataFrame({'l':list(sp_lengths.values()), 's':list(sp_lengths.keys())})
#df_sub = df.loc[ df.s.isin(source_ids) & (df.l>360)]

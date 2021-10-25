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

output_road_network = os.path.join(data_dir, "open-roads RoadLink betcen diffs.gpkg")
output_pave_links = os.path.join(data_dir, "pednetworkLinksWithCentralities.gpkg")
output_pave_ex_diag_links = os.path.join(data_dir, "pednetworkLinksExDiagWithCentralities.gpkg")
output_pave_res_links = os.path.join(data_dir, "pednetworkLinksResWithCentralities.gpkg")
output_pave_res_time_links = os.path.join(data_dir, "pednetworkLinksResTimeWithCentralities.gpkg")

##########################
#
#
# Functions
#
#
##########################

# Calculate difference between sides of the road - non crossing links associated to each road link
def betcen_diff(df, pave_betcen_col = 'paveBC'):
	# Expect two non crossing links, if not don't calculate value
	if df.shape[0] != 2:
		print(df)
		return

	diff = (df[pave_betcen_col].values[0] - df[pave_betcen_col].values[1]) / (df[pave_betcen_col].values[0] + df[pave_betcen_col].values[1])
	return diff

def betcen_max(df, pave_betcen_col = 'paveBC'):
	s = df[pave_betcen_col].dropna()
	if s.shape[0] == 0:
		return np.nan
	else:
		return max(s)

def betcen_sum(df, pave_betcen_col = 'paveBC'):
	s = df[pave_betcen_col].dropna()
	if s.shape[0]==0:
		return np.nan
	else:
		return sum(s)

def disagg_centrality(df, id_col, bc_col):
	n = df.shape[0]
	return df.set_index(id_col)[bc_col] / n

def pave_bc_dif_from_av(df, id_col, bc_col):
	n = df[bc_col].dropna().shape[0]
	if n==0:
		return None
	av_bc = betcen_sum(df, pave_betcen_col = bc_col) / n

	return df.set_index(id_col)[bc_col] - av_bc

def calculate_graph_edge_bc_centralities(g, normalized, weight, id_col, value_col):
	bc_values = nx.edge_betweenness_centrality(g, normalized=normalized, weight=weight)

	# Unpack values to link to edge id
	output={id_col:[], value_col:[]}
	for k, v in bc_values.items():
		edge_id = g[k[0]][k[1]][id_col]
		output[id_col].append(edge_id)
		output[value_col].append(v)
	return pd.DataFrame(output)

def get_all_graph_bc_values(dfLinksLookup, dict_graphs, normalized, weight, id_col):
	for name, graph in dict_graphs.items():
		value_col = name+"BC"
		norm_value_col = name+"BCnorm"
		dfBC = calculate_graph_edge_bc_centralities(graph, normalized, weight, id_col, value_col)

		if name == 'road':
			dfBC.rename(columns={'fid':'or_fid'}, inplace=True)
			dfLinksLookup = pd.merge(dfLinksLookup, dfBC, on='or_fid', how = 'left')
		else:
			dfLinksLookup = pd.merge(dfLinksLookup, dfBC, on='fid', how = 'left')

		# Calculate normalised values
		norm_factor = ( 2 / ( (len(graph.nodes)-1) * (len(graph.nodes)-2) ) )
		dfLinksLookup[norm_value_col] = dfLinksLookup[value_col] * norm_factor

	return dfLinksLookup

def aggregate_bc_values(dfLinksBetCens, gdfORLinks, dict_graphs):
	'''Function to refactor aggregation process
	'''

	# First exclude direct crossing links from aggregation
	dfBetweenNoDirectCross = dfLinksBetCens.loc[ dfLinksBetCens['linkType']!= "direct_cross"]

	# For each graph, aggregate these BC values to the OR link level
	for name in dict_graphs.keys():
		# use normalised values
		value_col = name+"BCnorm"
		sum_col = name+"BCsum"
		range_col = name+"BCrange"

		bc_sum = dfBetweenNoDirectCross.groupby("or_fid").apply(betcen_sum, pave_betcen_col=value_col)
		bc_sum.name = sum_col

		bc_range = dfBetweenNoDirectCross.groupby("or_fid")[value_col].apply(lambda bcs: max(bcs) - min(bcs) )
		bc_range.name = range_col

		dfBCSum = pd.DataFrame(bc_sum).reset_index()
		dfBCRange = pd.DataFrame(bc_range).reset_index()

		# Merge into OR Links
		gdfORLinks = pd.merge(gdfORLinks, dfBCSum, left_on='fid', right_on='or_fid', how='left').drop('or_fid', axis=1)
		gdfORLinks = pd.merge(gdfORLinks, dfBCRange, left_on='fid', right_on='or_fid', how='left').drop('or_fid', axis=1)

	return gdfORLinks

def dissaggregate_bc_values(dfLinksBetCens, dict_graphs):
	# First exclude direct crossing links from aggregation
	dfBetweenNoDirectCross = dfLinksBetCens.loc[ dfLinksBetCens['linkType']!= "direct_cross"]

	# For each graph, aggregate these BC values to the OR link level
	for name in dict_graphs.keys():
		# use normalised values
		value_col = name+"BCnorm"
		disag_col = name+"BCdisag"
		diff_col = name+"BCdiff"

		bc_disag = dfBetweenNoDirectCross.groupby("or_fid").apply(disagg_centrality, id_col='fid', bc_col=value_col)
		bc_disag.name = disag_col

		# Merge into dfLinkBetCens and calculate difference between disagg bc and actual bc
		dfLinksBetCens = pd.merge(dfLinksBetCens, bc_disag, on='fid', how='left')
		dfLinksBetCens[diff_col] = dfLinksBetCens[value_col] - dfLinksBetCens[disag_col]

	return dfLinksBetCens

def dissaggregate_bc_values_pavement(dfLinksBetCens, dict_graphs):
	# First exclude direct crossing links from aggregation
	dfBetweenNoDirectCross = dfLinksBetCens.loc[ dfLinksBetCens['linkType']!= "direct_cross"]

	# For each graph, aggregate these BC values to the OR link level
	for name in dict_graphs.keys():
		# use normalised values
		value_col = name+"BCnorm"
		diff_col = name+"BCdiff_pv"

		bc_diff_pv = dfBetweenNoDirectCross.groupby("or_fid").apply(pave_bc_dif_from_av, id_col='fid', bc_col=value_col)
		bc_diff_pv.name = diff_col

		# Merge into dfLinkBetCens and calculate difference between disagg bc and actual bc
		dfLinksBetCens = pd.merge(dfLinksBetCens, bc_diff_pv, on='fid', how='left')

	return dfLinksBetCens

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
gdfORLinks = gdfORLinks.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID', 'class', 'geometry', 'length'])
gdfORLinks['length'] = gdfORLinks['geometry'].length

gdfPaveLinks['length'] = gdfPaveLinks['geometry'].length


########################
#
#
# Clean OR Link Classification
#
#
########################

class_rename_dict = {	'Unknown':'Unclassified',
						'Not Classified': 'Unclassified',
						'Unclassified_Unknown': 'Unclassified',
						'Unknown_Unclassified': 'Unclassified',
						'Unclassified_Not Classified': 'Unclassified',
						'Not Classified_Unclassified': 'Unclassified',
						'Not Classified_Unknown': 'Unclassified',
						'Unknown_Not Classified': 'Unknown',
						'Unknown_A Road': 'A Road',
						'Unclassified_A Road':'A Road',
						'Unclassified_B Road':'B Road',
						'B Road_Unclassified': 'B Road',
						'Unknown_Classified Unnumbered': 'Classified Unnumbered',
						'Unknown_Unclassified_Classified Unnumbered': 'Classified Unnumbered',
						'Unclassified_Classified Unnumbered':'Classified Unnumbered',
						'Not Classified_Classified Unnumbered': 'Classified Unnumbered',
						'Classified Unnumbered_A Road': 'A Road',
						'Classified Unnumbered_Unclassified': 'Classified Unnumbered',
						'B Road_A Road': 'A Road',
						'A Road_Not Classified':'A Road',
						'Not Classified_A Road': 'A Road',
						'Unclassified_B Road_A Road':'A Road',
						'B Road_Unclassified_A Road': 'A Road',
						'Classified Unnumbered_Unknown': 'Classified Unnumbered',
						'B Road_Unknown': 'B Road',
						'Not Classified_B Road': 'B Road',
						'B Road_Classified Unnumbered': 'B Road',
						'Unclassified_Classified Unnumbered_Unknown': 'Classified Unnumbered',
						'Unclassified_Unknown_A Road': 'A Road',
						'Unknown_Unclassified_A Road': 'A Road',
						'Not Classified_Unclassified_A Road': 'A Road',
						'Classified Unnumbered_B Road': 'B Road',
						'B Road_Not Classified': 'B Road',
						'Classified Unnumbered_Not Classified': 'Classified Unnumbered',
						'Unclassified_Not Classified_A Road': 'A Road'
					}

gdfORLinks['class'] = gdfORLinks['class'].replace(class_rename_dict)

assert gdfORLinks.loc[ ~gdfORLinks['class'].isin(['Unclassified','A Road','B Road', 'Classified Unnumbered'])].shape[0] == 0

########################
#
#
# Create lookup between pavement links and road links
#
# Now including road link classification
#
########################

# Create lookup from pavement links to corresponding or road link
dfPaveNodes = gdfPaveNodes.reindex(columns=['fid', 'juncNodeID'])
dfLinks = pd.merge(gdfPaveLinks, dfPaveNodes, left_on = 'MNodeFID', right_on = 'fid', how = 'left')
dfLinks = dfLinks.rename(columns = {'juncNodeID':'juncNodeID_M', 'fid_x':'fid'}).drop('fid_y', axis=1)
dfLinks = pd.merge(dfLinks, gdfPaveNodes, left_on = 'PNodeFID', right_on = 'fid', how = 'left')
dfLinks = dfLinks.rename(columns = {'juncNodeID':'juncNodeID_P', 'fid_x':'fid'}).drop('fid_y', axis=1)

dfLinks = dfLinks.reindex(columns = ['MNodeFID', 'PNodeFID', 'fid', 'pedRLID', 'linkType', 'juncNodeID_M', 'juncNodeID_P'])
dfORLinks = gdfORLinks.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID', 'class']).rename(columns = {'fid':'pedRLID'})

dfLinks['or_link_cross'] = dfLinks['pedRLID']

# First select ped links that already have a road link assocaited
dfLinksA = dfLinks.loc[ ~dfLinks['pedRLID'].isnull()]
dfLinksA = pd.merge(dfLinksA, dfORLinks.reindex(columns = ['pedRLID','class']), left_on = 'pedRLID', right_on = 'pedRLID', suffixes = ('','_or'))
assert dfLinksA['fid'].duplicated().any() == False

# Then find associated road link for links that dont have road link
dfLinksB = dfLinks.loc[ dfLinks['pedRLID'].isnull()].drop('pedRLID', axis=1)

dfLinksB1 = pd.merge(dfLinksB, dfORLinks, left_on = ['juncNodeID_M', 'juncNodeID_P'], right_on = ['MNodeFID', 'PNodeFID'], how='inner')
dfLinksB2 = pd.merge(dfLinksB, dfORLinks, left_on = ['juncNodeID_M', 'juncNodeID_P'], right_on = ['PNodeFID', 'MNodeFID'], how='inner')

dfLinksBFull = pd.concat([dfLinksB1, dfLinksB2], join='inner')
dfLinksBFull.drop(['MNodeFID_y', 'PNodeFID_y'], axis=1, inplace=True)
dfLinksBFull.rename(columns = {'MNodeFID_x':'MNodeFID', 'PNodeFID_x':'PNodeFID'}, inplace=True)

# Duplcated entries in dfLinksBFull caused by OR Links having multi links. Not sure how these have crept in, but remove now
or_links_to_remove = dfLinksBFull.loc[ dfLinksBFull['fid'].duplicated(), 'pedRLID'].values
dfLinksBFull = dfLinksBFull.loc[ ~dfLinksBFull['pedRLID'].isin(or_links_to_remove)]
gdfORLinks = gdfORLinks.loc[~gdfORLinks['fid'].isin(or_links_to_remove)]

assert dfLinksBFull.shape[0] == dfLinksB.shape[0]

dfLinksLookup = pd.concat([dfLinksA, dfLinksBFull])

dfLinksLookup.rename(columns={'pedRLID':'or_fid'}, inplace=True)
dfLinksLookup.to_csv(output_links_lookup, index=False)
gdfORLinks.to_file(output_or_roads)

dfLinksLookup = pd.read_csv(output_links_lookup)


########################
#
#
# Create networks
#
#
########################
edges_pavement_ex_diag = gdfPaveLinks.loc[ gdfPaveLinks['linkType']!='diag_cross', ['MNodeFID', 'PNodeFID', 'fid', 'length']]
edges_pavement = gdfPaveLinks.loc[:, ['MNodeFID', 'PNodeFID', 'fid', 'length']]

diag_links_on_unclassified_roads = dfLinksLookup.loc[ (dfLinksLookup['class'] == 'Unclassified') & (dfLinksLookup['linkType']=='diag_cross'), 'fid'].values
residential_diag_cross_edges = gdfPaveLinks.loc[ gdfPaveLinks['fid'].isin(diag_links_on_unclassified_roads), ['MNodeFID', 'PNodeFID', 'fid', 'length']]
edges_residential_jaywalk = pd.concat([edges_pavement_ex_diag, residential_diag_cross_edges])

direct_links_on_a_roads = dfLinksLookup.loc[ (dfLinksLookup['class']=='A Road') & (dfLinksLookup['linkType']=='direct_cross'), 'fid'].values
edges_time_res_jaywalk = edges_residential_jaywalk.copy()
edges_time_res_jaywalk['time'] = edges_time_res_jaywalk['length'] / 1.47 # From Willis 2004
edges_time_res_jaywalk.loc[ edges_time_res_jaywalk['fid'].isin(direct_links_on_a_roads), 'time'] = edges_time_res_jaywalk.loc[ edges_time_res_jaywalk['fid'].isin(direct_links_on_a_roads), 'time'] + 15 # 70% of peds crossed within 15s of arriving

edges_ex_diag_time = edges_pavement_ex_diag.copy()
edges_ex_diag_time['time'] = edges_ex_diag_time['length'] / 1.47 # From Willis 2004
edges_ex_diag_time.loc[ edges_ex_diag_time['fid'].isin(direct_links_on_a_roads), 'time'] = edges_ex_diag_time.loc[ edges_ex_diag_time['fid'].isin(direct_links_on_a_roads), 'time'] + 15 # 70% of peds crossed within 15s of arriving


edges_road = gdfORLinks.loc[:, ['MNodeFID', 'PNodeFID', 'fid', 'length']]

g_pavement_ex_diag = nx.from_pandas_edgelist(edges_pavement_ex_diag, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)
g_pavement = nx.from_pandas_edgelist(edges_pavement, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)
g_pavement_res = nx.from_pandas_edgelist(edges_residential_jaywalk, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)
g_pavement_res_time = nx.from_pandas_edgelist(edges_time_res_jaywalk, 'MNodeFID', 'PNodeFID', edge_attr=['fid','time'], create_using=nx.Graph)

g_road = nx.from_pandas_edgelist(edges_road, 'MNodeFID', 'PNodeFID', edge_attr=['fid','length'], create_using=nx.Graph)

dict_graphs = {'paveExD':g_pavement_ex_diag, 'pave':g_pavement, 'paveR':g_pavement_res, 'paveRT':g_pavement_res_time, 'road':g_road}
########################
#
#
# Calculate centralities
#
#
########################
dfLinksBetCens = get_all_graph_bc_values(dfLinksLookup, dict_graphs, normalized=False, weight = 'length', id_col='fid')

# Save the data
dfLinksBetCens.to_csv("link_betcens_unnorm_refactored.csv", index=False)

# Load the data
dfLinksBetCens = pd.read_csv("link_betcens_unnorm_refactored.csv")
dfLinksBetCens_unfactored = pd.read_csv("link_betcens_unnorm.csv")
dfLinksBetCens_orig = pd.read_csv("link_betcens.csv")

# Check normalised correctly
dfLinksCheck = pd.merge(dfLinksBetCens, dfLinksBetCens_orig.reindex(columns = ['fid','paveBC']).rename(columns = {'paveBC':'paveBCnorm'}), on='fid') # Need to rename bc I have change column naming convention since this data was saved.
dfLinksCheck['diff'] = (dfLinksCheck['paveBCnorm_x'] - dfLinksCheck['paveBCnorm_y']) / dfLinksCheck['paveBCnorm_y']
dfLinksCheck['diff'].describe()
assert dfLinksCheck['diff'].max()<0.0003

# Check refactoring hasn't changed values
col_pairs = [	('fid','fid'), ('paveExDBC', 'paveExDBC_un'), ('paveBC', 'paveBC_un'), ('paveRBC', 'paveRBC_un'), 
				('paveExDBCnorm', 'paveExDBC'), ('paveBCnorm', 'paveBC'), ('paveRBCnorm', 'paveResBC'), ('roadBC', 'roadBC_un')]
for c1, c2 in col_pairs:
	try:
		assert (dfLinksBetCens[c1] == dfLinksBetCens_unfactored[c2]).all()
	except AssertionError as e:
		try:
			# Try comparing another way
			assert (dfLinksBetCens[c1] - dfLinksBetCens_unfactored[c2]).max()<0.0000001
		except AssertionError as e:
			print(c1, c2)

###################################
#
#
# Aggregate Centrality Analysis
#
# Aggregate pavement network centralities to road link and compare to RCL centrality values
#
###################################

gdfORLinksBC = aggregate_bc_values(dfLinksBetCens, gdfORLinks, dict_graphs)

# Merge in road link BC values
dfORBetCen = dfLinksBetCens.loc[:, ['or_fid', 'roadBC', 'roadBCnorm']].drop_duplicates()
gdfORLinksBC = pd.merge(gdfORLinksBC, dfORBetCen, left_on = 'fid', right_on = 'or_fid', how='left')

gdfORLinksBC = gdfORLinksBC.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID', 'geometry', 'length', 'paveBCsum','paveExDBCsum', 'paveRBCsum', 'paveRTBCsum', 'paveBCrange','paveExDBCrange', 'paveRBCrange', 'paveRTBCrange', 'roadBCsum', 'roadBCrange'])

gdfORLinksBC.rename(columns = {'fid':'or_link_id'}, inplace=True)
gdfORLinksBC.to_file(output_road_network, driver='GPKG')

###################################
#
#
# Disaggregate Centrality Analysis
#
# Disaggregate RCL centrality and compare to pavement network centrality
#
####################################
dfLinksBetCens = dissaggregate_bc_values(dfLinksBetCens, dict_graphs)
dfLinksBetCens = dissaggregate_bc_values_pavement(dfLinksBetCens, dict_graphs)

dfPaveBC = dfLinksBetCens.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveBC', 'roadBC', 'paveBCdisag', 'BCdiff', 'paveBCnorm', 'roadBCnorm', 'paveBCdiff_pv'])
dfPaveBCExDiag = dfLinksBetCens.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveExDBC', 'roadBC', 'paveExDBCdisag', 'paveExDBCdiff', 'paveExDBCnorm', 'roadBCnorm', 'paveExDBCdiff_pv']).dropna(axis=0, subset = ['paveExDBCnorm'])
dfPaveBCRes = dfLinksBetCens.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveRBC', 'roadBC', 'paveRBCdisagg', 'paveRBCdiff', 'paveRBCnorm', 'roadBCnorm', 'paveRBCdiff_pv']).dropna(axis=0, subset = ['paveRBCnorm'])
dfPaveBCRT = dfLinksBetCens.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveRTBC', 'roadBC', 'paveRTBCdisag', 'paveRTBCdiff', 'paveRTBCnorm', 'roadBCnorm', 'paveRTBCdiff_pv']).dropna(axis=0, subset = ['paveRTBCnorm'])

gdfPaveLinksWBC = pd.merge(gdfPaveLinks, dfPaveBC, left_on = 'fid', right_on = 'fid', how='inner')
gdfPaveLinksExDiagWBC = pd.merge(gdfPaveLinks, dfPaveBCExDiag, left_on = 'fid', right_on = 'fid', how='inner')
gdfPaveLinksResWBC = pd.merge(gdfPaveLinks, dfPaveBCRes, left_on = 'fid', right_on = 'fid', how='inner')
gdfPaveLinksRTWBC = pd.merge(gdfPaveLinks, dfPaveBCRT, left_on = 'fid', right_on = 'fid', how='inner')

# Rename fid field so that data can get saved to geopackage format. Exception raised otherwise
gdfPaveLinksWBC.rename(columns = {'fid':'pave_link_id'}, inplace=True)
gdfPaveLinksExDiagWBC.rename(columns = {'fid':'pave_link_id'}, inplace=True)
gdfPaveLinksResWBC.rename(columns = {'fid':'pave_link_id'}, inplace=True)
gdfPaveLinksRTWBC.rename(columns = {'fid':'pave_link_id'}, inplace=True)

gdfPaveLinksWBC.to_file(output_pave_links, driver='GPKG')
gdfPaveLinksExDiagWBC.to_file(output_pave_ex_diag_links, driver='GPKG')
gdfPaveLinksResWBC.to_file(output_pave_res_links, driver='GPKG')
gdfPaveLinksRTWBC.to_file(output_pave_res_time_links, driver='GPKG')
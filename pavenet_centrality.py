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
output_pave_res_links = os.path.join(data_dir, "pednetworkLinksResWithCentralities")
output_pave_res_time_links = os.path.join(data_dir, "pednetworkLinksResTimeWithCentralities")

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
			dfBC.rename({'fid':'or_fid'}, inplace=True)
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

gdfORLinksBC = gdfORLinksBC.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID', 'geometry', 'length', 'BCSum','BCSumExDi', 'BCSumRes', 'BCSumRT', 'BCRange','BCRangeExDi', 'BCRangeRes', 'BCRangeRT', 'roadBC', 'roadBC_un'])
gdfORLinksBC.to_file(output_road_network)

###################################
#
#
# Disaggregate Centrality Analysis
#
# Disaggregate RCL centrality and compare to pavement network centrality
#
####################################

# dfBetweenNoDirectCross - need to use this df instead of dfLinksBetCens to avoid apportioning RCL centrality to direct crossing edges.

# Disaggregate RCL centrality among component pavement network links
rdBCPave = dfBetweenNoDirectCross.groupby('or_fid').apply(disagg_centrality, id_col = 'fid', bc_col = 'roadBC')
rdBCPave.name = 'rdBCPave'
rdBCPave = rdBCPave.reset_index().drop('or_fid', axis=1)
rdBCPaveExD = dfBetweenNoDirectCross.loc[ dfBetweenNoDirectCross['linkType']=='pavement'].groupby('or_fid').apply(disagg_centrality, id_col = 'fid', bc_col = 'roadBC')
rdBCPaveExD.name = 'rdBCPaveExD'
rdBCPaveExD = rdBCPaveExD.reset_index().drop('or_fid', axis=1)
rdBCPaveRes = dfBetweenNoDirectCross.loc[ dfBetweenNoDirectCross['fid'].isin(edges_residential_jaywalk['fid'])].groupby('or_fid').apply(disagg_centrality, id_col = 'fid', bc_col = 'roadBC')
rdBCPaveRes.name = 'rdBCPaveRes'
rdBCPaveRes = rdBCPaveRes.reset_index().drop('or_fid', axis=1)
rdBCPaveRT = dfBetweenNoDirectCross.loc[ dfBetweenNoDirectCross['fid'].isin(edges_time_res_jaywalk['fid'])].groupby('or_fid').apply(disagg_centrality, id_col = 'fid', bc_col = 'roadBC')
rdBCPaveRT.name = 'rdBCPaveRT'
rdBCPaveRT = rdBCPaveRT.reset_index().drop('or_fid', axis=1)

# Merge these into dfLinkBetCens
dfLinksBetCens = pd.merge(dfLinksBetCens, rdBCPave, on = 'fid', how = 'left')
dfLinksBetCens = pd.merge(dfLinksBetCens, rdBCPaveExD, on = 'fid', how = 'left')
dfLinksBetCens = pd.merge(dfLinksBetCens, rdBCPaveRes, on = 'fid', how = 'left')
dfLinksBetCens = pd.merge(dfLinksBetCens, rdBCPaveRT, on = 'fid', how = 'left')

# Calculate difference between disaggregate value and actual pavement network centrality value
dfLinksBetCens['BCDiff'] = dfLinksBetCens['paveBC'] - dfLinksBetCens['rdBCPave']
dfLinksBetCens['BCDiffExDi'] = dfLinksBetCens['paveExDBC'] - dfLinksBetCens['rdBCPaveExD']
dfLinksBetCens['BCDiffRes'] = dfLinksBetCens['paveResBC'] - dfLinksBetCens['rdBCPaveRes']
dfLinksBetCens['BCDiffRT'] = dfLinksBetCens['paveRTBC'] - dfLinksBetCens['rdBCPaveRT']

dfLinksBetCens['BCDiffFr'] = dfLinksBetCens['BCDiff'] / dfLinksBetCens['roadBC_un']
dfLinksBetCens['BCDiffExDiFr'] = dfLinksBetCens['BCDiffExDi'] / dfLinksBetCens['roadBC_un']
dfLinksBetCens['BCDiffResFr'] = dfLinksBetCens['BCDiffRes'] / dfLinksBetCens['roadBC_un']
dfLinksBetCens['BCDiffRTFr'] = dfLinksBetCens['BCDiffRT'] / dfLinksBetCens['roadBC_un']

# Also calculate the difference between average pavement centrality per pavement link and pavement centrality. This might be a better comparison for showing differences between sides of the road.
BCDiffPv = dfBetweenNoDirectCross.groupby('or_fid').apply(pave_bc_dif_from_av, 'fid', 'paveBC')
BCDiffPv.name = 'BCDiffPv'
BCDiffPv = BCDiffPv.reset_index().drop('or_fid', axis=1)
BCDfExDiPv = dfBetweenNoDirectCross.loc[ dfBetweenNoDirectCross['linkType']=='pavement'].groupby('or_fid').apply(pave_bc_dif_from_av, 'fid', 'paveExDBC')
BCDfExDiPv.name = 'BCDfExDiPv'
BCDfExDiPv = BCDfExDiPv.reset_index().drop('or_fid', axis=1)
BCDfResPv = dfBetweenNoDirectCross.loc[ dfBetweenNoDirectCross['fid'].isin(edges_residential_jaywalk['fid'])].groupby('or_fid').apply(pave_bc_dif_from_av, 'fid', 'paveResBC')
BCDfResPv.name = 'BCDfResPv'
BCDfResPv = BCDfResPv.reset_index().drop('or_fid', axis=1)
BCDfRTPv = dfBetweenNoDirectCross.loc[ dfBetweenNoDirectCross['fid'].isin(edges_time_res_jaywalk['fid'])].groupby('or_fid').apply(pave_bc_dif_from_av, 'fid', 'paveRTBC')
BCDfRTPv.name = 'BCDfRTPv'
BCDfRTPv = BCDfRTPv.reset_index().drop('or_fid', axis=1)

# Merge these in also
dfLinksBetCens = pd.merge(dfLinksBetCens, BCDiffPv, on = 'fid', how = 'left')
dfLinksBetCens = pd.merge(dfLinksBetCens, BCDfExDiPv, on = 'fid', how = 'left')
dfLinksBetCens = pd.merge(dfLinksBetCens, BCDfResPv, on = 'fid', how = 'left')
dfLinksBetCens = pd.merge(dfLinksBetCens, BCDfRTPv, on = 'fid', how = 'left')

dfPaveBC = dfLinksBetCens.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveBC', 'roadBC', 'rdBCPave', 'BCDiff', 'BCDiffFr', 'paveBC_un', 'roadBC_un', 'BCDiffPv'])
dfPaveBCExDiag = dfLinksBetCens.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveExDBC', 'roadBC', 'rdBCPaveExD', 'BCDiffExDi', 'BCDiffExDiFr', 'paveExDBC_un', 'roadBC_un', 'BCDfExDiPv']).dropna(axis=0, subset = ['paveExDBC'])
dfPaveBCRes = dfLinksBetCens.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveResBC', 'roadBC', 'rdBCPaveRes', 'BCDiffRes', 'BCDiffResFr', 'paveRBC_un', 'roadBC_un', 'BCDfResPv']).dropna(axis=0, subset = ['paveResBC'])
dfPaveBCRT = dfLinksBetCens.reindex(columns = ['fid', 'or_link_cross', 'or_fid', 'paveRTBC', 'roadBC', 'rdBCPaveRT', 'BCDiffRT', 'BCDiffRTFr', 'paveRTBC_un', 'roadBC_un', 'BCDfRTPv']).dropna(axis=0, subset = ['paveRTBC'])


gdfPaveLinksWBC = pd.merge(gdfPaveLinks, dfPaveBC, left_on = 'fid', right_on = 'fid', how='inner')
gdfPaveLinksExDiagWBC = pd.merge(gdfPaveLinks, dfPaveBCExDiag, left_on = 'fid', right_on = 'fid', how='inner')
gdfPaveLinksResWBC = pd.merge(gdfPaveLinks, dfPaveBCRes, left_on = 'fid', right_on = 'fid', how='inner')
gdfPaveLinksRTWBC = pd.merge(gdfPaveLinks, dfPaveBCRT, left_on = 'fid', right_on = 'fid', how='inner')

gdfPaveLinksWBC.to_file(output_pave_links)
gdfPaveLinksExDiagWBC.to_file(output_pave_ex_diag_links)
gdfPaveLinksResWBC.to_file(output_pave_res_links)
gdfPaveLinksRTWBC.to_file(output_pave_res_time_links)
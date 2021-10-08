

import os
import json

import pydot
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community.quality import intra_community_edges, modularity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import community as community_louvain
import matplotlib.cm as cm
from matplotlib.transforms import Bbox
from networkx import edge_betweenness_centrality as betweenness
import itertools
from sklearn import cluster

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

from src.codeFragment import CodeFragment

from src.settings import s as program_settings

class Service():
	id_counter = 0
	def __init__(self):
		self.id = 'ser' + str(Service.id_counter)
		self.code_fragments = set()
		self.interfaces = set()
		self.intra_edges = [] # edges within services
		self.inter_edges = [] # edges between services
		Service.id_counter += 1

	def add_code_fragment(self, code_fragment):
		self.code_fragments.add(code_fragment)
	
	def add_interface(self, interface):
		self.interfaces.add(interface)

	def has_interface(self):
		if len(self.interfaces) > 0:
			return True
		else:
			return False
	
	def add_intra_edge(self, intra_edge):
		self.intra_edges.append(intra_edge)

	def add_inter_edge(self, inter_edge):
		self.inter_edges.append(inter_edge)

	def get_total_intra_edge_weight(self):
		total_intra_edge_weight = 0
		for edge in self.intra_edges:
			total_intra_edge_weight += edge[2]['weight']
		return total_intra_edge_weight

	def get_total_inter_edge_weight(self):
		total_inter_edge_weight = 0
		for edge in self.inter_edges:
			total_inter_edge_weight += edge[2]['weight']
		return total_inter_edge_weight

	def get_intra_edges(self, setting='structural'):
		n_intra_edges = 0
		# Structural edge is found by either static or dynamic analysis
		for intra_edge in self.intra_edges:
			if setting == 'structural':
				if intra_edge[2]['static'] or intra_edge[2]['dynamic'] != 0:
					n_intra_edges += 1
			elif setting == 'conceptual':
				if intra_edge[2]['semantic'] != 0:
					n_intra_edges += 1
			else:
				print('Error. Not a valid setting.')
		return n_intra_edges

	def get_inter_edges(self, service, setting='structural'):
		n_inter_edges = 0
		for inter_edge in self.inter_edges:
			if inter_edge[1] in service.code_fragments:
				if setting == 'structural':
					if inter_edge[2]['static'] or inter_edge[2]['dynamic'] != 0:
						n_inter_edges += 1
				elif setting == 'conceptual':
					if inter_edge[2]['semantic'] != 0:
						n_inter_edges += 1
				else:
					print('Error. Not a valid setting.')
		return n_inter_edges

	# Formula: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.6755&rep=rep1&type=pdf
	def get_intra_connectivity(self, setting='structural'):
		n_code_fragments = len(self.code_fragments)
		n_intra_edges = self.get_intra_edges(setting=setting)
		max_intra_edges = pow(n_code_fragments, 2)
		total_intra_connectivity = n_intra_edges/max_intra_edges
		return total_intra_connectivity

	def get_inter_connectivity(self, services, setting='structural'):
		total_inter_connectivity = 0
		for service in services:
			inter_connectivity = 0
			if not service == self:
				n_inter_edges = self.get_inter_edges(service, setting=setting)
				inter_connectivity = n_inter_edges / (2*len(self.code_fragments)*len(service.code_fragments))
			total_inter_connectivity += inter_connectivity
		return total_inter_connectivity
	
	# CF
	def get_cluster_factor(self):
		cluster_factor = 0
		total_intra_edge_weight = self.get_total_intra_edge_weight()
		total_inter_edge_weight = self.get_total_inter_edge_weight()
		if total_intra_edge_weight and total_inter_edge_weight:
			cluster_factor = round((2*total_intra_edge_weight) / ((2*total_intra_edge_weight) + total_inter_edge_weight), 3)
		return cluster_factor

	# OPN
	def get_number_of_operations(self):
		n_operations = 0
		for interface in self.interfaces:
			for _ in interface.operations:
				n_operations += 1
		return n_operations

	def get_interface_number(self):
		return len([interface for interface in self.interfaces])

	def to_dictionary(self, services, program):
		service = {
			'id': self.id,
			'statistics': {
				'n_intra_edges': len(self.intra_edges),
				'total_weight_intra_edges': self.get_total_intra_edge_weight(),
				'n_inter_edges': len(self.inter_edges),
				'total_weight_inter_edges': self.get_total_inter_edge_weight(),
				'cluster_factor': self.get_cluster_factor(),
				'ifn': self.get_interface_number(),
				'opn': self.get_number_of_operations(),
				'scoh': self.get_intra_connectivity(setting='structural'),
				'scop': self.get_inter_connectivity(services, setting='structural'),
				'ccoh': self.get_intra_connectivity(setting='conceptual'),
				'ccop': self.get_inter_connectivity(services, setting='conceptual')
			},
			'code_fragments': [CodeFragment.search_code_fragment_with_id(program, cf).name for cf in self.code_fragments],
			'interfaces': [interface.to_dictionary() for interface in self.interfaces]
		}
		return service

	def print_nicely(self):
		S2 = ' '
		print('Service id:', self.id)
		print('Statistics:')
		print(S2 * 2 + 'Intra edges', self.intra_edges)
		print(S2 * 2 + 'Intra weights:', self.get_total_intra_edge_weight())
		print(S2 * 2 + 'Inter edges:', self.inter_edges)
		print(S2 * 2 + 'Inter weights:', self.get_total_inter_edge_weight())
		print(S2 * 2 + 'Cluster Factor:', self.get_cluster_factor())
		print('With code fragments:')
		for code_fragment in self.code_fragments:
			print(S2 * 2 + code_fragment)
		print('And interfaces:')
		if self.interfaces:
			for interface in self.interfaces:
				print(S2 * 2 + interface.name)
				print(S2 * 2 + 'with operations:')
				for operation in interface.operations:
					print(S2 * 4 + operation.id + ' (' + operation.name + ')')
		else:
			print(S2 * 2 + 'None')

# Make a graph for each partition
class Graph():
	def __init__(self, graph_type, cluster_type = 'louvain'):
		self.main_graph = graph_type
		self.cluster_type = cluster_type
		self.graph_partition = [] # Format: (['cf1', 'cf2], ['cf3', 'cf3]) -> iterable list of partitions
		self.node_labels = dict # Format: {'cf1': 1, 'cf2': 2}, {'cf1': 2} -> key = node_id, value = cluster_id
		self.services = set()
		self.partition = []
		self.louvain_labels = None
		self.louvain_clusters = []
		self.girvan_labels = dict()
		self.girvan_clusters = None
		self.hierarchical_labels = dict() 
		self.hierarchical_clusters = [] 
		
	def remove_isolated_nodes(self):
		self.remove_zero_weight_edges()
		self.main_graph.remove_nodes_from(list(nx.isolates(self.main_graph)))

	def remove_zero_weight_edges(self):
		edges_to_be_removed = []
		for edge in self.main_graph.edges.data('weight'):
			if edge[2] == 0:
				edges_to_be_removed.append(edge)
		self.main_graph.remove_edges_from(edges_to_be_removed)
	
	def add_node_list(self, node_list):
		for node in node_list:
			self.main_graph.add_node(node)
			
	def add_edge_list(self, edge_list):
		for edge in edge_list:
			self.main_graph.add_edge(edge[0], edge[1], weight=edge[2])
	
	def get_ordered_node_list(self):
		return sorted(self.main_graph.nodes())

	def to_pandas_adjacency(self):
		return nx.to_pandas_adjacency(self.main_graph, nodelist=self.get_ordered_node_list())

	def to_numpy_adjacency(self):
		# Order rows and columns according to CF_id
		return nx.to_numpy_matrix(self.main_graph, nodelist=self.get_ordered_node_list())

	def add_service(self, service):
		self.services.add(service)

	def get_services(self):
		if self.graph_partition:
			for c, cluster in enumerate(self.graph_partition):
				# Add 1 to save 0 for external edges
				ser = Service()
				for cf_id in cluster:
					ser.add_code_fragment(cf_id)
				for node1, node2, edge_data in self.main_graph.edges.data():
					# Only a dependency when weight is not zero
					if self.main_graph.edges[node1, node2]['community'] == c:
						if self.main_graph.edges[node1, node2]['weight'] > 0:
							if self.main_graph.edges[node1, node2]['type'] == 'internal':
								ser.add_intra_edge((node1, node2, edge_data))
							if self.main_graph.edges[node1, node2]['type'] == 'external':
								ser.add_inter_edge((node1, node2, edge_data))
				self.add_service(ser)
		else:
			print('A graph partition is needed to obtain services.')
			
	def set_node_community(self):
		'''Add community to node attributes'''
		for c, cluster in enumerate(self.graph_partition):
			for cf_id in cluster:
				# Add 1 to save 0 for external edges
				self.main_graph.nodes[cf_id]['community'] = c

	def set_edge_community(self):
		'''Find internal edges and add their community to their attributes'''
		for v, w, in self.main_graph.edges:
			if self.main_graph.nodes[v]['community'] == self.main_graph.nodes[w]['community']:
				# Internal edge, mark with community
				self.main_graph.edges[v, w]['type'] = 'internal'
				self.main_graph.edges[v, w]['community'] = self.main_graph.nodes[v]['community']
			else:
				# External edge, mark as 0
				self.main_graph.edges[v, w]['type'] = 'external'
				self.main_graph.edges[v, w]['community'] = self.main_graph.nodes[v]['community']
	
	def get_girvan_newman_partition(self, depth):
		def most_central_edge(G):
			centrality = betweenness(G, weight='weight')
			return max(centrality, key=centrality.get)
		com = community.girvan_newman(self.main_graph, most_valuable_edge=most_central_edge)
		for communities in itertools.islice(com, depth):
			partition = tuple(sorted(c) for c in communities)
		self.graph_partition = list(partition)
		node_labels = self.get_labels_dict_from_partition(partition)
		self.node_labels = node_labels
		self.set_node_community()
		self.set_edge_community()
		self.get_services()

	def get_louvain_partition(self, resolution):
		com = community_louvain.best_partition(self.main_graph, resolution=resolution, random_state=42)
		# com = {'cf1': 1}
		com_sorted = {key: val for key, val in sorted(com.items(), key = lambda ele: ele[0])}
		self.node_labels = com_sorted
		k_partitions = len(set(com.values()))
		k = 0
		partition = []
		while k < k_partitions:
			cluster = []
			for key, value in com.items():
				if value == k:
					cluster.append(key)
			partition.append(cluster)
			k += 1
		self.graph_partition = partition # list of lists
		self.set_node_community()
		self.set_edge_community()
		self.get_services()

	def get_clause_newman_moore_partition(self):
		com = community.greedy_modularity_communities(self.main_graph, weight='weight')
		self.graph_partition = [x for x in com]
		node_labels = self.get_labels_dict_from_partition(com)
		self.node_labels = node_labels
		self.set_node_community()
		self.set_edge_community()
		self.get_services()

	def get_lpa_partition(self):
		com = community.asyn_lpa_communities(self.main_graph, weight='weight', seed=42)
		self.graph_partition = [x for x in com]
		node_labels = self.get_labels_dict_from_partition(com)
		self.node_labels = node_labels
		self.set_node_community()
		self.set_edge_community()
		self.get_services()

	@staticmethod
	def get_labels_dict_from_partition(partition):
		# partition = (['cf1', 'cf2], ['cf3', 'cf3])
		# transform partitions into labels dictionary {'cf1': 1}
		labels = dict()
		for i, cluster in enumerate(partition):
			for cf_id in cluster:
				labels[cf_id] = i
		# Important to sort to make them match the adjancency matrix
		return {key: val for key, val in sorted(labels.items(), key = lambda ele: ele[0])}

	def get_hierarchical_partition(self):
		com = AgglomerativeClustering(affinity='precomputed', linkage='average').fit(self.to_numpy_adjacency())
		n_clusters = len(set(com.labels_))
		n = 0
		while n < n_clusters:
			cluster = []
			for node_id, label in zip(self.ordered_node_list, com.labels_):
				if label == n:
					cluster.append(node_id)
			self.hierarchical_clusters.append(cluster)
			n += 1
		for node_id, label in zip(self.ordered_node_list, com.labels_):
			self.hierarchical_labels[node_id] = label

	def get_label_dictionary_from_clusters(self, list_of_clusters):
		# list of clusters must be in the form:([cf0, cf3, cf4], [cf5, cf6, cf9])
		labels = dict()
		for i, cluster in enumerate(list_of_clusters):
			for cf_id in cluster:
				labels[cf_id] = i
		return labels

	def assign_cluster_id_to_code_fragments(self, program):
		for key, value in self.node_labels.items():
			code_fragment = CodeFragment.search_code_fragment_with_id(program, key)
			code_fragment.cluster_id = value

	def get_node_sizes(self, program):
		node_sizes = []
		for node in self.main_graph.nodes():
			cf = CodeFragment.search_code_fragment_with_id(program, node)
			node_size = 30
			for _ in cf.get_all_nested_code_fragments([]):
				node_size += 10
			node_sizes.append(node_size)
		return node_sizes

	def show_graph(self, program, name, draw_edge_labels=True, draw_node_labels=True):
		pos = nx.spring_layout(self.main_graph)
		if self.graph_partition:
			node_sizes = self.get_node_sizes(program)
			cmap = cm.get_cmap('rainbow', max(self.node_labels.values()) + 1)
			nx.draw_networkx_nodes(self.main_graph, pos, self.node_labels.keys(), node_size=node_sizes, cmap=cmap, node_color=list(self.node_labels.values()))
		else:
			nx.draw_networkx_nodes(self.main_graph, pos, self.main_graph.nodes, node_size=40)
		nx.draw_networkx_edges(self.main_graph, pos)
		if draw_edge_labels:
			edge_labels = nx.get_edge_attributes(self.main_graph, 'weight')
			for key, value in edge_labels.items():
				edge_labels[key] = round(value, 2)
			nx.draw_networkx_edge_labels(self.main_graph, pos, edge_labels=edge_labels, font_size=5)
		if draw_node_labels:
			def nudge(pos, x_shift, y_shift):
				# {k:[v[0] + x_shift,v[1]+y_shift] for k,v in pos.iteritems()}
				return {k:[v[0] + x_shift,v[1]+y_shift] for k,v in pos.items()}
			new_pos = nudge(pos, 0, 0.1)
			new_labels = {}
			for node in self.node_labels.keys():
				cf = CodeFragment.search_code_fragment_with_id(program, node)
				new_labels[node] = cf.name[11:]
			node_label_handles = nx.draw_networkx_labels(self.main_graph, pos=new_pos, labels=new_labels, font_size=6)
			# Add white bounding box behind the node labels
			[label.set_bbox(dict(facecolor='white', edgecolor='none')) for label in node_label_handles.values()]
		ax = plt.gca()
		ax.collections[0].set_edgecolor("#000000")
		plt.axis('off')
		plt.savefig(program.name + '_' + name + '.eps', bbox_inches='tight', pad_inches=0)
		plt.savefig(program.name + '_' + name + '.pdf', bbox_inches='tight', pad_inches=0)
		nx.write_gexf(self.main_graph, program.name + '_' + name + '.gexf')
		graph = nx.drawing.nx_pydot.to_pydot(self.main_graph)
		graph.write_png(program.name + '_' + name + '.png')
		graph.write_pdf(program.name + '_' + name + '.pdf')
		plt.show()

def get_combined_graphs(G1, G2, list_of_nodes, G1_weight=0.5):
	# new_graph = G1.main_graph.copy()
	G_combined = Graph(nx.Graph())
	G_combined.main_graph.add_nodes_from(list_of_nodes)
	for node1, node2 in G2.main_graph.edges():
		if G1.main_graph.has_edge(node1, node2):
			combined_weight = (G1.main_graph[node1][node2]['weight'] * G1_weight) + (G2.main_graph[node1][node2]['weight'] * (1-G1_weight))
			G_combined.main_graph.add_edge(node1,node2, weight=combined_weight)
		else:
			weight = G2.main_graph[node1][node2]['weight']
			G_combined.main_graph.add_edge(node1, node2, weight=weight)
	return G_combined

def get_triple_combined_graphs(G1, G2, G3, list_of_nodes, G1_weight=0.5, G2_weight=0.5):
	# new_graph = G1.main_graph.copy()
	G_combined = Graph(nx.Graph())
	G_combined.main_graph.add_nodes_from(list_of_nodes)
	for node1, node2 in G2.main_graph.edges():
		if G1.main_graph.has_edge(node1, node2):
			combined_weight = (G1.main_graph[node1][node2]['weight'] * G1_weight) + (G2.main_graph[node1][node2]['weight'] * (1-G1_weight))
			G_combined.main_graph.add_edge(node1,node2, weight=combined_weight)
		else:
			weight = G2.main_graph[node1][node2]['weight']
			G_combined.main_graph.add_edge(node1, node2, weight=weight)
	return G_combined

def get_single_graph(list_of_nodes, list_of_edges):
	# Networkx has a to_undirected function but it doesn't sum weights, 
	# it is just updating weight with the last found edge weight from 
	# the original graph. You should do this manually.
	# https://stackoverflow.com/questions/56169907/networkx-change-weighted-directed-graph-to-undirected
	G = Graph(nx.DiGraph())
	G.main_graph.add_nodes_from(list_of_nodes)
	G.main_graph.add_weighted_edges_from(list_of_edges)
	UG = G.main_graph.to_undirected()
	for node in G.main_graph:
		for ngbr in nx.neighbors(G.main_graph, node):
			if node in nx.neighbors(G.main_graph, ngbr):
				UG[node][ngbr]['weight'] = (G.main_graph[node][ngbr]['weight'] + G.main_graph[ngbr][node]['weight'])
	G.main_graph = UG
	return G

def get_graph_with_weighted_edges(edges_dict, setting, degree_w1=0, degree_w2=0):
	G = Graph(nx.Graph())
	for edge, edge_data in edges_dict.items():
		static_data = edge_data.get('static')
		dynamic_data = edge_data.get('dynamic')
		semantic_data = edge_data.get('semantic')
		if setting == 'static':
			weight = static_data
		elif setting == 'dynamic':
			weight = dynamic_data
		elif setting == 'semantic':
			weight = semantic_data
		elif setting == 'static_dynamic':
			weight = (degree_w1 * static_data) + ((1-degree_w1) * dynamic_data)
		elif setting == 'static_semantic':
			weight = (degree_w1 * static_data) + ((1-degree_w1) * semantic_data)
		elif setting == 'dynamic_semantic':
			weight = (degree_w1 * dynamic_data) + ((1-degree_w1) * semantic_data)
		elif setting == 'static_dynamic_semantic':
			weight = (degree_w1 * static_data) + (degree_w2 * dynamic_data) + (round(1-degree_w1-degree_w2,1) * semantic_data)
		else:
			print('Error. Not a valid setting.')
		# Add weight to edge tuple
		e = edge + (weight, )
		G.main_graph.add_weighted_edges_from([e], static=static_data, dynamic=dynamic_data, semantic=semantic_data)
	return G

def combine_edges(edges_dict, list_of_edges, data_name):
	# 3-tuples (u, v, d) where d is a dictionary containing edge data
	for raw_edge in list_of_edges:
		reversed_data_edge = tuple([raw_edge[1]] + [raw_edge[0]])
		data_edge = tuple([raw_edge[0]] + [raw_edge[1]])
		if data_edge in edges_dict.keys():
			edges_dict[data_edge][data_name] += raw_edge[2]
			continue
		elif reversed_data_edge in edges_dict.keys():
			edges_dict[reversed_data_edge][data_name] += raw_edge[2]
			continue
		else:
			edges_dict[data_edge] = {'static': 0, 'dynamic': 0, 'semantic': 0}
			edges_dict[data_edge][data_name] += raw_edge[2]
	return edges_dict

def normalise_edges_dict(edges_dict):
	max_static_weight = max([edge['static'] for edge in edges_dict.values()])
	max_semantic_weight = max([edge['semantic'] for edge in edges_dict.values()])
	max_dynamic_weight = max([edge['dynamic'] for edge in edges_dict.values()])
	for edge in edges_dict.keys():
		edges_dict[edge]['static'] = edges_dict[edge]['static']/max_static_weight
		edges_dict[edge]['semantic'] = edges_dict[edge]['semantic']/max_semantic_weight
		edges_dict[edge]['dynamic'] = edges_dict[edge]['dynamic']/max_dynamic_weight
	return edges_dict

# check edges
def check_edge(list_of_nodes, edge):
	if edge[0] in list_of_nodes and edge[1] in list_of_nodes:
		return True
	else:
		print('Edge not valid:', edge)
		return False

def execute_experiments(program):
	code_fragments = program.get_all_code_fragments_to_be_clustered()
	list_of_nodes = [cf.id for cf in code_fragments]
	list_of_static_edges = [tuple(edge)[:3] for cf in code_fragments for edge in cf.dependency_edges if check_edge(list_of_nodes, edge)]
	list_of_semantic_edges = [tuple(edge[:3]) for cf in code_fragments for edge in cf.semantic_edges if check_edge(list_of_nodes, edge)]
	list_of_dynamic_edges = [tuple(edge)[:3] for cf in code_fragments for edge in cf.dynamic_edges if check_edge(list_of_nodes, edge)]
	edges_dict = {}
	edges_dict = combine_edges(edges_dict, list_of_static_edges, 'static')
	edges_dict = combine_edges(edges_dict, list_of_semantic_edges, 'semantic')
	edges_dict = combine_edges(edges_dict, list_of_dynamic_edges, 'dynamic')
	edges_dict = normalise_edges_dict(edges_dict)
	settings = ['static', 'semantic', 'dynamic', 'static_dynamic', 'static_semantic', 'dynamic_semantic', 'static_dynamic_semantic']
	# settings = ['static', 'semantic', 'static_semantic', 'static_dynamic_semantic']
	# settings = ['static']
	# algorithms = ['louvain', 'girvan_newman', 'clauset_newman_moore', 'lpa']
	algorithms = ['louvain']
	depths = [1,2,3,4,5]
	# resolutions = [0.3, 0.65, 1]
	resolutions = [1]
	results = []
	for s in settings:
		for algo in algorithms:
			if s in ['static', 'semantic', 'dynamic']:
				if algo == 'girvan_newman':
					for d in depths:
						res = single_experiment(code_fragments, edges_dict, program, s, algo, param=d)
						results.append(res)
				elif algo == 'louvain':
					for res in resolutions:
						res = single_experiment(code_fragments, edges_dict, program, s, algorithm=algo, param=res)
						results.append(res)
				else:
					res = single_experiment(code_fragments, edges_dict, program, s, algorithm=algo)
					results.append(res)
			if s in ['static_dynamic', 'static_semantic', 'dynamic_semantic']:
                # [0.5]
				# [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
				for w1 in [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]:
					if algo == 'girvan_newman':
						for d in depths:
							res = single_experiment(code_fragments, edges_dict, program, s, algo, param=d, degree_w1=w1)
							results.append(res)
					elif algo == 'louvain':
						for res in resolutions:
							res = single_experiment(code_fragments, edges_dict, program, s, algo, param=res, degree_w1=w1)
							results.append(res)
					else:
						res = single_experiment(code_fragments, edges_dict, program, s, algo, degree_w1=w1)
						results.append(res)
			if s in ['static_dynamic_semantic']:
                # [0.333]
				# [0, 0.2, 0.3333, 0.4, 0.5, 0.6, 0.8, 1]
				for w1 in [0.333]:
					for w2 in [0.333]:
						if w1 + w2 > 1:
							continue
						if algo == 'girvan_newman':
							for d in depths:
								res = single_experiment(code_fragments, edges_dict, program, s, algo, param=d, degree_w1=w1, degree_w2=w2)
								results.append(res)
						elif algo == 'louvain':
							for res in resolutions:
								res = single_experiment(code_fragments, edges_dict, program, s, algo, param=res, degree_w1=w1, degree_w2=w2)
								results.append(res)
						else:
							res = single_experiment(code_fragments, edges_dict, program, s, algo, degree_w1=w1, degree_w2=w2)
							results.append(res)		
	df = pd.DataFrame.from_dict(results)
	df.to_csv(f'{program.name}.csv')
	df.to_excel(f'{program.name}.xlsx')
	print(df)

def single_experiment(list_of_nodes, edges_dict, program, setting, algorithm, param=None, degree_w1=None, degree_w2=None):
	G = get_graph_with_weighted_edges(edges_dict, setting, degree_w1=degree_w1, degree_w2=degree_w2)
	G.remove_isolated_nodes()
	if algorithm == 'girvan_newman':
		G.get_girvan_newman_partition(param)
	elif algorithm == 'louvain':
		G.get_louvain_partition(param)
	elif algorithm == 'clauset_newman_moore':
		G.get_clause_newman_moore_partition()
	else:
		G.get_lpa_partition()
	G.assign_cluster_id_to_code_fragments(program)
	add_interfaces_to_services(program, G.services)
	params = [setting, algorithm, param, degree_w1, degree_w2]
	list_of_service_dict = []
	for ser in G.services:
		list_of_service_dict.append(ser.to_dictionary(G.services, program))
	MS_metrics = compute_MS_metrics(G, list_of_nodes, params)
	export_results_to_json(list_of_service_dict, MS_metrics, params)
	G.show_graph(program, name=setting)
	del G
	return MS_metrics

def make_graph(program, setting='static', G1_weight=0.5):
	code_fragments = program.get_all_code_fragments_to_be_clustered()
	list_of_nodes = [cf.id for cf in code_fragments]
	list_of_static_edges = [tuple(edge)[:3] for cf in code_fragments for edge in cf.dependency_edges]
	list_of_semantic_edges = [tuple(edge[:2]+edge[4:5]) for cf in code_fragments for edge in cf.semantic_edges]
	list_of_dynamic_edges = [tuple(edge)[:3] for cf in code_fragments for edge in cf.dynamic_edges]
	if setting == 'static':
		G = get_single_graph(list_of_nodes, list_of_static_edges)
	if setting == 'semantic':
		G = get_single_graph(list_of_nodes, list_of_semantic_edges)
	if setting == 'dynamic':
		G = get_single_graph(list_of_nodes, list_of_dynamic_edges)
	if setting == 'static_semantic':
		G_static = get_single_graph(list_of_nodes, list_of_static_edges)
		G_semantic = get_single_graph(list_of_nodes, list_of_semantic_edges)
		G = get_combined_graphs(G_static, G_semantic, list_of_nodes, G1_weight=G1_weight)
	if setting == 'static_dynamic':
		G_static = get_single_graph(list_of_nodes, list_of_static_edges)
		G_dynamic = get_single_graph(list_of_nodes, list_of_dynamic_edges)
		G = get_combined_graphs(G_static, G_dynamic, list_of_nodes, G1_weight=G1_weight)
	if setting == 'semantic_dynamic':
		G_semantic = get_single_graph(list_of_nodes, list_of_semantic_edges)
		G_dynamic = get_single_graph(list_of_nodes, list_of_dynamic_edges)
		G = get_combined_graphs(G_semantic, G_dynamic, G1_weight=G1_weight)
	if setting == 'static_semantic_dynamic':
		G_static = get_single_graph(list_of_nodes, list_of_static_edges)
		G_semantic = get_single_graph(list_of_nodes, list_of_semantic_edges)
		G_dynamic = get_single_graph(list_of_nodes, list_of_dynamic_edges)
	return G

def community_detection(G, G_static, G_semantic, algorithm, program):
	# G.remove_isolated_nodes()
	adj = G.to_numpy_adjacency()
	if algorithm == 'louvain':
		G.get_louvain_partition()
	elif algorithm == 'girvan-newman':
		# G.remove_isolated_nodes()
		G.get_girvan_newman_partition()
		print('Silhouette score:', silhouette_score(adj, list(G.girvan_labels.values())))
	elif algorithm == 'hierarchical':
		G.get_hierarchical_partition()
		print('Silhouette score:', silhouette_score(adj, list(G.hierarchical_labels.values())))
		print('Structural modularity', community.modularity(G_static.main_graph, G.hierarchical_clusters))
		print('Semantic modularity', community.modularity(G_semantic.main_graph, G.hierarchical_clusters))
	nicely_print_clusters(program, G.graph_partition)
	G.show_graph()
	G.assign_cluster_id_to_code_fragments(program)
	interfaces = get_interfaces(G, program, G.graph_partition)
	MS_metrics = compute_MS_metrics(interfaces, G.graph_partition)
	print(MS_metrics)
	# print('Structural modularity', community.modularity(G_static.main_graph, G.girvan_clusters))
	# G_semantic.remove_isolated_nodes()
	# print('Semantic modularity', community.modularity(G_semantic.main_graph, G.girvan_clusters))

def nicely_print_clusters(program, clusters):
	for i, cluster in enumerate(clusters):
		print('Cluster', i)
		for id in cluster:
			code_fragment = CodeFragment.search_code_fragment_with_id(program, id)
			print(code_fragment.name, code_fragment.id)

def add_interfaces_to_services(program, set_of_services):
	for curr_service in set_of_services:
		cfs_other_services = [cf for service in set_of_services for cf in service.code_fragments if cf not in curr_service.code_fragments]
		for code_fragment_id in curr_service.code_fragments:
			# For each code fragment in the cluster
			code_fragment_A = CodeFragment.search_code_fragment_with_id(program, code_fragment_id)
			structural_dependencies = set(code_fragment_A.dependencyManager.internal_dependencies + code_fragment_A.dynamic_dependencies)
			for dependency in structural_dependencies:
				code_fragment_B = CodeFragment.search_code_fragment_with_name(program, dependency)
				if not code_fragment_B:
					print('Code fragment not found.')
					continue
				encapsulator_code_fragment_A = code_fragment_A.get_cluster_code_fragment()
				encapsulator_code_fragment_B = code_fragment_B.get_cluster_code_fragment()
				if not encapsulator_code_fragment_A is encapsulator_code_fragment_B:
					if encapsulator_code_fragment_B.id in cfs_other_services:
						code_fragment_B.is_operation = True
						for service in set_of_services:
							# Find the service
							if encapsulator_code_fragment_B.id in service.code_fragments:
								# Check if the interface already exists
								if not encapsulator_code_fragment_B.name in [interface.name for interface in service.interfaces]:
									interface = Interface(encapsulator_code_fragment_B.name)
									interface.add_operation(code_fragment_B)
									service.add_interface(interface)
								else:
									for interface in service.interfaces:
										if interface.name == encapsulator_code_fragment_B.name:
											interface.add_operation(code_fragment_B)

def get_interfaces(program, set_of_services):
	interfaces = []
	for curr_service in set_of_services:
		ids_other_clusters = [cf for service in set_of_services for cf in service.code_fragments if cf not in curr_service.code_fragments]
		for code_fragment_id in curr_service.code_fragments:
			# For each code fragment in the cluster
			code_fragment_A = CodeFragment.search_code_fragment_with_id(program, code_fragment_id)
			code_fragment_A.service_id = curr_service.id
			for dependency in code_fragment_A.dependencyManager.internal_dependencies:
				code_fragment_B = CodeFragment.search_code_fragment_with_name(program, dependency)
				encapsulator_code_fragment_A = code_fragment_A.get_cluster_code_fragment()
				encapsulator_code_fragment_B = code_fragment_B.get_cluster_code_fragment()
				if not encapsulator_code_fragment_A is encapsulator_code_fragment_B:
					if encapsulator_code_fragment_B.id in ids_other_clusters:
						code_fragment_B.is_operation = True
						# encapsulator_code_fragment_B.service_id = curr_service.id
						print(code_fragment_B.name, 'should be exposable for', encapsulator_code_fragment_A.name)
	class_interfaces = get_class_interfaces(program, set_of_services)
	module_interfaces = get_module_interfaces(program, set_of_services)
	interfaces = class_interfaces + module_interfaces
	print(interfaces)
	return interfaces

def search_service_with_id(services, id):
	for service in services:
		if service.id == id:
			return service
		else:
			print('Service not found.')

def get_class_interfaces(program, services):
	class_interfaces = []
	code_fragments = program.get_all_code_fragments_to_be_clustered()
	code_fragments_classes = [cf for cf in code_fragments if cf.type == 'class']
	for code_fragment_class in code_fragments_classes:
		operations = []
		for code_fragment in code_fragment_class.get_all_nested_code_fragments([]):
			if code_fragment.is_operation:
				operations.append(code_fragment)
		if operations:
			interface = Interface(code_fragment_class.name)
			for operation in operations:
				interface.add_operation(operation)
			print(code_fragment_class.service_id)
			service = search_service_with_id(services, code_fragment_class.service_id)
			service.add_interface(interface)
			class_interfaces.append(interface)
			# program.interfaces.append(interface)
	return class_interfaces

def get_module_interfaces(program, services):
	module_interfaces = []
	code_fragments = program.get_all_code_fragments_to_be_clustered()
	code_fragments_modules = [cf for cf in code_fragments if cf.type == 'module']
	for code_fragment_module in code_fragments_modules:
		operations = []
		for code_fragment in code_fragment_module.get_all_nested_code_fragments([]):
			if code_fragment.is_operation:
				operations.append(code_fragment)
		if operations:
			interface = Interface(code_fragment_module.name)
			for operation in operations:
				interface.add_operation(operation)
			service = search_service_with_id(services, code_fragment_module.service_id)
			service.add_interface(interface)
			module_interfaces.append(interface)
			# program.interfaces.append(interface)
	return module_interfaces

def get_function_operations(program):
	code_fragments = program.get_all_code_fragments()
	code_fragments_functions = [cf for cf in code_fragments if cf.type == 'function']
	for code_fragment_function in code_fragments_functions:
		if code_fragment_function.parent.type != 'class':
			if code_fragment_function.is_operation:
				interface = Interface(code_fragment_function.name, code_fragment_function.cluster_id, [code_fragment_function])
				program.interfaces.append(interface)
				interface.nicely_print()

class Interface():
	def __init__(self, name):
		self.name = name
		self.operations = set()

	def add_operation(self, code_fragment):
		self.operations.add(code_fragment)

	# CHM
	def get_cohesion_at_message_level(self):
		cohesion_at_message_level = 1
		if not len(self.operations) == 1:
			total_sim = 0
			for operation_A in self.operations:
				for operation_B in self.operations:
					if operation_A is operation_B:
						continue
					intersection_input = len(set(operation_A.input_params) & set(operation_B.input_params))
					union_input = len(set(operation_A.input_params) | set(operation_B.input_params))
					intersection_output = len(set(operation_A.output_params) & set(operation_B.output_params))
					union_output = len(set(operation_A.output_params) | set(operation_B.output_params))
					if intersection_input != 0 and union_input != 0:
						intput_sim = intersection_input / union_input
					else:
						intput_sim = 0
					if intersection_output != 0 and union_output != 0:
						output_sim = intersection_output / union_output
					else:
						output_sim = 0
					total_sim += (intput_sim + output_sim)/2
			denominator = (len(self.operations) * (len(self.operations)-1))/2
			cohesion_at_message_level = total_sim / denominator
		return cohesion_at_message_level

	# CHD
	def get_cohesion_at_domain_level(self):
		cohesion_at_domain_level = 1
		if not len(self.operations) == 1:
			total_operation_sim = 0
			for operation_A in self.operations:
				for operation_B in self.operations:
					if operation_A is operation_B:
						continue
					domain_terms_A = set(operation_A.get_list_of_name_terms())
					domain_terms_B = set(operation_B.get_list_of_name_terms())
					intersection_names = len(domain_terms_A & domain_terms_B)
					union_names = len(domain_terms_A | domain_terms_B)
					if intersection_names != 0 and union_names != 0:
						operation_sim = intersection_names / union_names
					else:
						operation_sim = 0
					total_operation_sim += operation_sim
			denominator = (len(self.operations) * (len(self.operations)-1))/2
			cohesion_at_domain_level = total_operation_sim / denominator
		return cohesion_at_domain_level

	def to_dictionary(self):
		interface = {
			'name': self.name,
			'operations': [operation.name for operation in self.operations],
			'chm': self.get_cohesion_at_message_level(),
			'chd': self.get_cohesion_at_domain_level()
		}
		return interface

def average_chd(services):
	chd = 0
	n_interfaces = number_of_interfaces(services)
	if n_interfaces > 0:
		for service in services:
			for interface in service.interfaces:
				chd += interface.get_cohesion_at_domain_level()
		chd = round(chd/n_interfaces, 3)
	return chd

def average_chm(services):
	chm = 0
	n_interfaces = number_of_interfaces(services)
	if n_interfaces > 0:
		for service in services:
			for interface in service.interfaces:
				chm += interface.get_cohesion_at_message_level()
		chm = round(chm/n_interfaces, 3)
	return chm

def average_ifn(services):
	# Number of clusters that provide interfaces
	# The smaller the better
	# If having no interfaces, this means that services assume a single responsiblity 
	ifn = 0
	n_interfaces = number_of_interfaces(services)
	if n_interfaces > 0:
		n_services_with_interface = number_of_services_with_interface(services)
		ifn = round(n_interfaces/n_services_with_interface, 3)
	return ifn

# NSC
def number_of_singleton_services(services):
	return len([ser for ser in services if len(ser.code_fragments) == 1])

# MCS
def max_service_size(services):
	return max([len(ser.code_fragments) for ser in services])

def number_of_interfaces(services):
	n_interfaces = 0
	for service in services:
		for _ in service.interfaces:
			n_interfaces += 1
	return n_interfaces

def number_of_operations(services):
	n_operations = 0
	for service in services:
		n_operations += service.get_number_of_operations()
	return n_operations

def number_of_services_with_interface(services):
	return len(set([ser for ser in services if ser.has_interface()]))

# MQ by Mancoridis
def modularity_quality(services, setting='structural'):
	intra = 0
	inter = 0
	for ser in services:
		intra += ser.get_intra_connectivity(setting=setting)
		inter += ser.get_inter_connectivity(services, setting=setting)
	denominator = (len(services)*(len(services)-1))/2
	if denominator:
		modularity_quality = (intra/len(services)) - (inter/denominator)
	else:
		 modularity_quality = intra/len(services)
	return round(modularity_quality, 3)

# mCF
def average_cluster_factor(services):
	total_cluster_factor = 0
	for ser in services:
		total_cluster_factor += ser.get_cluster_factor()
	return round(total_cluster_factor / len(services), 3)

# SMQ
def structural_modularity_quality(G_static, clusters):
	# Clusters must be a partition of G_static
	return community.modularity(G_static.main_graph, clusters)

# CMQ
def conceptual_modularity_quality(G_semantic, clusters):
	# TODO: How to calculate CMQ when clusters are not a partition of main_graph (becomes a node from static)
	return community.modularity(G_semantic.main_graph, clusters)

def get_missing_reason(missing_cfs):
	no_behavior = []
	third_party = []
	other_reason = []
	# missing_class is an code fragment object
	for cf in missing_cfs:
		# No nested functions/methods
		if not cf.code_fragments:
			no_behavior.append(cf)
		# No internal dependencies
		elif not cf.dependencyManager.internal_dependencies and cf.dynamic_dependencies:
			# But has other dependencies (3rd party)
			if cf.dependencyManager.depend:
				third_party.append(cf)
			else:
				other_reason.append(cf)
		else:
			other_reason.append(cf)
	return no_behavior, third_party, other_reason

def get_modules(code_fragments):
	mods = set()
	for cf in code_fragments:
		if cf.type == 'module':
			n_nested_functions = len([elem.name for elem in cf.get_all_nested_code_fragments([]) if elem.type == 'function'])
			if cf.dependencyManager.depend or cf.dynamic_dependencies or n_nested_functions:
				mods.add(cf)
	return mods

def compute_MS_metrics(G, code_fragments, params):
	# all_modules = [cf for cf in code_fragments if cf.type == 'module']
	all_modules = get_modules(code_fragments)
	all_module_ids = [cf.id for cf in all_modules]
	missing_modules = set(all_module_ids) - set(G.main_graph.nodes())
	module_cfs = []
	for mod in missing_modules:
		for cf_mod in all_modules:
			if mod == cf_mod.id:
				module_cfs.append(cf_mod)
	all_classes = [cf for cf in code_fragments if cf.type == 'class']
	all_class_ids = [cf.id for cf in all_classes]
	missing_classes = set(all_class_ids) - set(G.main_graph.nodes())
	class_cfs = []
	for clas in missing_classes:
		for cf_clas in all_classes:
			if clas == cf_clas.id:
				class_cfs.append(cf_clas)
	no_behavior_classes, third_party_classes, other_reason_classes = get_missing_reason(class_cfs)
	no_behavior_modules, third_party_modules, other_reason_modules = get_missing_reason(module_cfs)
	all_cfs = len(all_classes)+len(all_modules)
	all_missing_cfs = len(missing_modules) + len(missing_classes)
	print(missing_classes)
	MS_metrics = {
   		'data': params[0],
		'algorithm': params[1],
		'param': params[2],
		'degree_source_1': params[3],
		'degree_source_2': params[4],
		'number_of_nodes': len(G.main_graph.nodes()),
		'coverage': round(len(G.main_graph.nodes())/all_cfs, 3),
		# '%_no_behavior': round(len([cf.name for cf in no_behavior_modules or no_behavior_classes])/all_missing_cfs, 3),
		# '%_third_party': round(len([cf.name for cf in third_party_modules or third_party_classes])/all_missing_cfs, 3),
		# '%_other_reason': round(len([cf.name for cf in other_reason_modules or other_reason_classes])/all_missing_cfs, 3),
		'number_of_modules': len(all_modules),
		'number_of_missing_modules': len(missing_modules),
		'no_behavior_modules': len([cf.name for cf in no_behavior_modules]),
		'third_party_modules': len([cf.name for cf in third_party_modules]),
		'other_reason_modules': len([cf.name for cf in other_reason_modules]),
		'number_of_classes': len(all_classes),
		'number_of_missing_classes': len(missing_classes),
		'no_behavior_classes': len([cf.name for cf in no_behavior_classes]),
		'third_party_classes': len([cf.name for cf in third_party_classes]),
		'other_reason_classes': len([cf.name for cf in other_reason_classes]),
		'number_of_edges': len(G.main_graph.edges()),
		'number_of_services': len(G.services),
		'number_of_services_with_interface': number_of_services_with_interface(G.services),
		'number_of_interfaces': number_of_interfaces(G.services),
		'number_of_operations': number_of_operations(G.services),
		'average_cohesion_at_domain_level': average_chd(G.services),
		'average_cohesion_at_message_level': average_chm(G.services),
		'average_interface_number': average_ifn(G.services),
		'number_of_singleton_services': number_of_singleton_services(G.services),
		'average_cluster_factor': average_cluster_factor(G.services),
		'structural_modularity_quality': modularity_quality(G.services, setting='structural'),
		'conceptual_modularity_quality': modularity_quality(G.services, setting='conceptual'),
		'max_service_size': max_service_size(G.services)
	}
	return MS_metrics

RESULTS_COUNTER = 0

def export_results_to_json(service_dict, MS_metrics_dict, params):
	name = f'{params[1]} clustering with {params[0]} data' 
	final_dict = {
		'name': name,
		'microservice_metrics': MS_metrics_dict,
		'services': sorted(service_dict, key=lambda k: int(k['id'][3:]))
	}
	
	global RESULTS_COUNTER

	with open(os.path.join(program_settings.DATA_PATH, program_settings.program_name + '_results_' + str(RESULTS_COUNTER) + '.json'), 'w') as f: 
		json.dump(final_dict, f, indent=4)

	RESULTS_COUNTER += 1
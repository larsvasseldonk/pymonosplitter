
import keyword
import re
import os
import csv
import math
import pandas as pd

import nltk
from pandas.core.frame import DataFrame
nltk.download('stopwords')

from src.graph import Graph
from src.settings import s

from sklearn.metrics import jaccard_score
from collections import Counter

import numpy as np
from sklearn.cluster import SpectralClustering

def print_header():
	print(36*'*')
	print(2*'*', 'Monolith to Microservices v.01', 2*'*')
	print(2*'*', 'Lars van Asseldonk', 11*' ', 2*'*')
	print(2*'*', 'Utrecht University 2021', 6*' ', 2*'*')
	print(36*'*')

def print_main_menu():
	print('\n-- PyMonoSplitter Main Menu --\n')
	print('1. Set program')
	print('2. Extract data')
	print('3. Make clustering')
	print('4. View program')
	print('0. Exit\n')
	return input('Enter [1-4]: ')

def print_data_menu():
	print('\n-- PyMonoSplitter Extract Data Menu --\n')
	print('1. Get dependencies')
	print('2. Extract semantics')
	print('3. Instrument logging')
	print('4. Load logging')
	print('0. Back \n')
	return input('Enter [1-4]: ')

def print_view_menu():
	print('\n-- PyMonoSplitter View Program Menu --\n')
	print('1. Program structure')
	print('2. Code fragments')
	print('3. Graph')
	print('0. Back \n')
	return input('Enter [1-3]: ')

def print_clustering_menu():
	print('\n-- PyMonoSplitter Clustering Menu --\n')
	print('1. Make similarity matrix')
	print('2. Make clustering')
	print('3. View microservices')
	print('0. Back \n')
	return input('Enter [1-3]: ')

def print_program_summary(program):
	S2 = ' ' * 2
	print('Root path:', program.root_path)
	print('Source path:', program.src_path)
	print('\nProgram structure:')
	print(program.name, '(Program)')
	for package in program.packages:
		print(S2 + package.name, '(Package)')
		for module in package.modules:
			print(S2*2 + module.name, '(Module)')
			for code_fragment in module.code_fragments:
				# print(S2*3 + code_fragment.name, '(Code fragment |', code_fragment.type, ')')
				if code_fragment.is_first:
					print(S2*3 + code_fragment.name, '(Code fragment |', code_fragment.type, ')')
					cfs = code_fragment.get_all_nested_code_fragments([])
					print('nested functions', cfs)
					for cf in cfs:
						print(cf.name)
				# if code_fragment.code_fragments:
				# 	for cf in code_fragment.code_fragments:
				# 		print(S2*4 + cf.name, ' (Sub code fragment|', cf.type, ')')

def nicely_print_code_fragments(code_fragments, indent):
	'''Recursive function to print code fragments.'''
	for code_fragment in code_fragments:
		print(indent * ' ' + code_fragment.name)
		# If code fragment has children
		if code_fragment.code_fragments:
			indent += 2
			nicely_print_code_fragments(code_fragment.code_fragments, indent)
	return indent

def print_module_dependencies(program):
	S2 = ' ' * 2
	print('DEPENDENCIES PER MODULE:')
	print('PROGRAM NAME:', program.name)
	for package in program.packages:
		print('PACKAGE NAME:', package.name)
		for module in package.modules:
			print('MODULE:', module.name)
			print(S2 + 'RAW:', module.dependencyManager.depend)
			print(S2 + 'EDGES:', module.dependencyManager.edges)

def print_code_fragments(program):
	S2 = ' ' * 2
	print('EXTRACTED CODE FRAGMENTS:')
	print('PROGRAM NAME:', program.name)
	for package in program.packages:
		print('PACKAGE NAME:', package.name)
		for module in package.modules:
			print('MODULE:', module.name)
			for code_fragment in module.code_fragments:
				print(code_fragment.nicely_print_code_fragments())
				# print('CODE FRAGMENT NAME:', code_fragment.name)
				# # print(code_fragment.code)
				# print('DEPENDENCIES:', code_fragment.dependencyManager.depend)
				# print('EDGES:', code_fragment.dependencyManager.edges)

def get_all_code_fragments(program):
	code_fragments = []
	for package in program.packages:
		for module in package.modules:
			for code_fragment in module.code_fragments:
				code_fragments.append(code_fragment.name)
	return code_fragments

def get_name_combinations(name):
	'''
	input: string with points e.g. 'beets.autotag.hooks'
	output: list with combinations e.g. ['hooks', 'autotag.hooks', 'beets.autotag.hooks']
	'''
	names = list()
	for i, elem in enumerate(reversed(name.split('.'))):
		if i == 0:
			name = elem
		else:
			name = elem + '.' + name
		names.append(name)
	return names

def make_vocab(code: list):
	'''
	Input: list of code tuples (name, identation, code)
	output: list of terms
	'''
	# Make set so all words are unique
	exclude_words = set(keyword.kwlist + nltk.corpus.stopwords.words('english'))
	vocab = []
	for line in code:
		# Remove all special characters and number
		# Only keep text
		line = re.sub(r'[^A-Za-z]+', r' ', line.strip())
		# If line is not null
		if line: 
			words = line.split()
			vocab = vocab + [word.lower() for word in words if word not in exclude_words]
	return vocab

def make_BoS(program):
	'''Make a Bag of Semantics (BoS)'''
	semanticManagers = list()
	for package in program.packages:
		for module in package.modules:
			for code_fragment in module.code_fragments:
				semanticManagers.append((code_fragment.name, code_fragment.semanticManager))
	# Create a list of all vocabs
	vocabs = [semanticManager[1].vocab for semanticManager in semanticManagers]
	# Merge vocabs to create a unique vocab
	vocab = sorted(set([item for elem in vocabs for item in elem]))
	BoS = dict() # Bag of Semantics
	BoS['vocab'] = vocab
	# For every vocab in the semanticManager
	# count the frequency of the words
	for semanticManager in semanticManagers:
		freq = [semanticManager[1].vocab.count(word) for word in vocab]
		BoS[semanticManager[0]] = freq
	export_BoS(BoS, program)
	print_BoS(BoS, program)

def export_BoS(BoS, program):
	'''Export Bag of Semantics to a csv file.'''
	with open(os.path.join(s.DATA_PATH, program.name + '_bag_of_semantics.csv'), 'w', newline='') as f:
		writer = csv.writer(f, dialect='excel')
		for key, value in BoS.items():
			writer.writerow([key] + value)

def print_BoS(BoS, program):
	df = pd.read_csv(os.path.join(s.DATA_PATH, program.name + '_bag_of_semantics.csv'), index_col='vocab')
	df.to_excel(os.path.join(s.DATA_PATH, program.name + '_BoS.xlsx'))
	print(df)

def make_bag_of_dependencies(program):
	dependencyManagers = list()
	for package in program.packages:
		for module in package.modules:
			for code_fragment in module.code_fragments:
				dependencyManagers.append((code_fragment.name, code_fragment.dependencyManager))
	# Create a list of all dependencies
	# Get all the values from the depend dictionary
	depends = [list(dependencyManager[1].depend.values()) for dependencyManager in dependencyManagers]
	# Merge dependencies to create a unique list of dependencies
	unique_depends = list()
	for depend in depends:
		# If list is not empty
		if depend:
			# Iterate over each item
			for d in depend:
				unique_depends = unique_depends + list(d)
	# List of all unique dependencies
	# ['<builtin>.dict', 'src.gpodder.feedcore.Result']
	unique_depends = sorted(set(unique_depends))
	bag_of_dependencies = dict() 
	bag_of_dependencies['dependencies'] = unique_depends
	# For every depend in the dependencyManager
	# count the frequency of the dependencies
	for dependencyManager in dependencyManagers:
		depend = list(dependencyManager[1].depend.values())
		# Examples of depend:
		# [{'<builtin>.dict', 'src.gpodder.feedcore.Result'}, {'src.gpodder.feedcore.Result'}]
		# [{'<builtin>.dict', 'src.gpodder.feedcore.Result'}]

		# Remove set from list
		# Result: ['<builtin>.dict', 'src.gpodder.feedcore.Result']
		depend = [item for elem in depend for item in elem]
		freq = [depend.count(word) for word in unique_depends]
		bag_of_dependencies[dependencyManager[0]] = freq
	export_bag_of_dependencies(bag_of_dependencies, program)
	print_bag_of_dependencies(bag_of_dependencies, program)

def export_bag_of_dependencies(bag_of_dependencies, program):
	'''Export Bag of Dependencies to a csv file.'''
	with open(os.path.join(s.DATA_PATH, program.name + '_bag_of_dependencies.csv'), 'w', newline='') as f:
		writer = csv.writer(f, dialect='excel')
		for key, value in bag_of_dependencies.items():
			writer.writerow([key] + value)

def print_bag_of_dependencies(bag_of_dependencies, program):
	df = pd.read_csv(os.path.join(s.DATA_PATH, program.name + '_bag_of_dependencies.csv'), index_col='dependencies')
	df.to_excel(os.path.join(s.DATA_PATH, program.name + '_BoD.xlsx'))
	print(df)

def load_logging(program):
	print('Load logging..')
	log_path = os.path.join(s.DATA_PATH, program.name + '_dynamics.log')
	with open(log_path) as f:
		logs = list() # List of all consecutive logs
		for i, line in enumerate(f.readlines()):
			log = line.split('::')
			name = re.sub(r'<locals>\.', '', log[13]) # sometimes the log contains <locals> tokens that we want to remove
			file_path = log[17][:-3].replace(os.sep, '.')
			# Make sure the name of the function is the same as the one defined for the code fragments
			for i, elem in enumerate(reversed(file_path.split('.'))):
				if i == 0:
					file_name = elem
				else:
					file_name = elem + '.' + file_name
				if elem == os.path.basename(program.src_path):
					break
			logs.append(file_name + name)
		log_groups = make_log_groups(logs)
	return log_groups

def make_log_groups(logs):
	all_traces = list()
	trace = list()
	last_log = None
	trace_id = 1
	n_permutations = 0
	for log in logs:
		# If log is not the same as the one before
		# and the max size of the trace is not achieved
		if n_permutations < s.LOGGING_WINDOW:
			# check if the function has changed
			if log is not last_log:
				n_permutations += 1
			trace.append(log)
			last_log = log
		else:
			all_traces.append((trace_id, trace))
			trace_id += 1
			trace = list()
			n_permutations = 0
	# If last trace is not empty
	if trace:
	 	all_traces.append((trace_id, trace))
	print(all_traces)
	return all_traces

def make_bag_of_dynamics(program):
	traces = load_logging(program)
	all_dynamics = list()
	headers = list()
	for package in program.packages:
		for module in package.modules:
			for code_fragment in module.code_fragments:
				headers.append(code_fragment.name)
				bof = []
				# trace is a tuple (trace_id, ['functions'])
				for trace in traces:
					# If the code fragment is called, put a 1
					# e.g. trace [(1, ['salie.db.make', 'salie.order.__init__', 'salie.db.make', ...])]
					if code_fragment.name in trace[1]:
						# Count the number of occurences
						count = trace[1].count(code_fragment.name)
						code_fragment.dynamics.append(count) # [6,0,0,1]
						bof = bof + [f'trace_{trace[0]}' for i in range(count)] # [1,1,1,1,1,1]
					else:
						code_fragment.dynamics.append(0)
				all_dynamics.append(code_fragment.dynamics)
				code_fragment.add_trace_manager(bof)
				print('Counter:', code_fragment.trace_manager.counter)
				code_fragment.bag_of_traces = bof
				print('code fragment:', code_fragment.name)
				print('counted', bof)
				# print('all', all_dynamics)
	columns = [f'Trace {i}' for i in range(len(traces))]
	df = pd.DataFrame(all_dynamics, columns=columns)
	df.insert(0, column='Code fragments', value=headers)
	df.set_index('Code fragments', inplace=True)
	program.bag_of_dynamics = df
	# df.to_excel(os.path.join(s.DATA_PATH, program.name + '_BoL.xlsx'))

def counter_cosine_similarity(c1, c2):
	# https://stackoverflow.com/questions/14720324/compute-the-similarity-between-two-lists
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def calculate_similarity(program):
	# Long list comprehension to collect all code fragments
	with open('temp.txt', 'w') as f:
		sim_matrix = list()
		headers = list()
		for cf_a in program.get_all_code_fragments():
			sim_list = list()
			headers.append(cf_a.name)
			for cf_b in program.get_all_code_fragments():
				if cf_a.dependencyManager.counter and cf_b.dependencyManager.counter:
					cos_depend = counter_cosine_similarity(cf_a.dependencyManager.counter, cf_b.dependencyManager.counter)
				else:
					cos_depend = 0.0
				if cf_a.semanticManager.counter and cf_b.semanticManager.counter:
					cos_semantic = counter_cosine_similarity(cf_a.semanticManager.counter, cf_b.semanticManager.counter)
				else:
					cos_semantic = 0.0
				if cf_a.trace_manager.counter and cf_b.trace_manager.counter:
					cos_traces = counter_cosine_similarity(cf_a.trace_manager.counter, cf_b.trace_manager.counter)
				else:
					cos_traces = 0.0
				total_sim = (0.33 * cos_depend) + (0.33 * cos_semantic) + (0.33 * cos_traces)
				sim_list.append(round(total_sim, 2))
				f.write(f'Cosine similarity between A: {cf_a.name} and B: {cf_b.name} is DEPEND: {cos_depend}, SEMANTIC: {cos_semantic}, TRACES: {cos_traces}\n')
				print(f'Cosine similarity between A: {cf_a.name} and B: {cf_b.name} is DEPEND: {cos_depend}, SEMANTIC: {cos_semantic}, TRACES: {cos_traces}')
			sim_matrix.append(sim_list)
	program.sim_matrix = sim_matrix
	# headers = [code_fragment.name for code_fragment in program.get_all_code_fragments()]
	cluster(sim_matrix, headers)
	df = pd.DataFrame(sim_matrix, columns=headers, index=headers)
	df.to_excel(os.path.join(s.DATA_PATH, s.program_name + '_similarity-matrix.xlsx'))

def cluster(sim_matrix, headers):
	mat = np.matrix(sim_matrix)
	res = SpectralClustering(2).fit_predict(mat)
	df = DataFrame(res, index=headers, columns=['Cluster'])
	print(df)
	
def calculate_jacc_similarity(program):
	print('Calculate Jaccard similarity..')
	# Long list comprehension to collect all code fragments
	code_fragments = [cf for pkg in program.packages for mod in pkg.modules for cf in mod.code_fragments]
	jacc_similarity = list()
	for cf_A in code_fragments:
		# For each logging vector, except for the current processed one
		cf_sim = list()
		for cf_B in code_fragments:
			if not cf_A.bag_of_traces:
				print(f'Cosine similarity between A: {cf_A.name} - {cf_A.bag_of_traces} and B: {cf_B.name} - {cf_B.bag_of_traces} is {0.0}')
				cf_sim.append(0.0)
				continue
			if not cf_B.bag_of_traces:
				print(f'Cosine similarity between A: {cf_A.name} - {cf_A.bag_of_traces} and B: {cf_B.name} - {cf_B.bag_of_traces} is {0.0}')
				cf_sim.append(0.0)
				continue
			count_A = Counter(cf_A.bag_of_traces)
			count_B = Counter(cf_B.bag_of_traces)
			cos = counter_cosine_similarity(count_A, count_B)
			print(f'Cosine similarity between A: {cf_A.name} - {cf_A.bag_of_traces} and B: {cf_B.name} - {cf_B.bag_of_traces} is {cos}')
			cf_sim.append(round(cos,2))
			# cf_sim.append(jaccard_score(cf_A.dynamics, cf_B.dynamics, zero_division=1.0))
		jacc_similarity.append(cf_sim)
	headers = [code_fragment.name for code_fragment in code_fragments]
	df = pd.DataFrame(jacc_similarity, columns=headers, index=headers)
	df.to_excel(os.path.join(s.DATA_PATH, program.name + '_dynamic-similarity-matrix.xlsx'))
	print(df)


def create_graph():
	g = Graph(5)
	g.add_edge(0,1)
	g.add_edge(0,2)
	g.add_edge(1,2)
	g.add_edge(2,0)
	g.add_edge(2,3)
	g.print_matrix()
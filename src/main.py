'''PyMonoSplitter Main Script

This scripts allows users to run PyMonoSplitter on user-
specified input programs. The tool assumes that the input program 
is a monolithic Python system.

The tool requires that all the dependencies listed in the `requirments`
file are installed within the Python environment you are running this
script in. 

It is also required to run the script as a module and, thus, not as 
a Python script. To do this, use the following command `python -m src.main`.

'''

import os
import re

import src.util
#import src.run_tests
import src.graph

from src.settings import s
from src.program import Program
from src.loggingManager import LoggingManager
from src.dependencyManager import dependencyManager
from src.semanticManager import semanticManager
#from src.similarity import compute_dependency_similarity, compute_semantic_similarity, compute_trace_similarity
#from src.clustering import Clustering, Hierarchical

if __name__ == '__main__':
	loop = True
	program = None

	src.util.print_header()
	while loop:
		choice = src.util.print_main_menu()

		# Set target
		if choice == '1':
			print('Current working directory: ', os.getcwd())
			target_root_path = ''
			target_src_path = ''
			target_name = ''
			target_default = input('Set target system to default? Y/N ')
			if target_default == 'Y' or target_default =='y':
				target_root_path = os.path.join(s.ROOT_PATH, 'salie-backup')
				target_src_path = os.path.join(target_root_path, 'salie')
				target_name = 'salie-backup'
				print(target_root_path, target_src_path)
			while len(target_name) == 0:
				target_name = input('Enter the name of the target app (no spaces): ')
				target_name = re.sub(r'\s', '_', target_name)
			while len(target_root_path) == 0:
				target_root_path = input('Enter the path of the target repository: ')
				if os.path.isdir(target_root_path):
					continue
				else:
					print('Not a valid path. Please try again.') 
					target_root_path = ''
			while len(target_src_path) == 0:
				src_folder = input('Select the folder in which the source files are located: ')
				target_src_path = os.path.join(target_root_path, src_folder)
				if os.path.isdir(target_src_path):
					continue
				else:
					print('Not a valid folder. Please try again.')
					target_src_path = ''
			program = Program(name=target_name, root_path=target_root_path, src_path=target_src_path)
			s.program_name = target_name
			program.search_packages()
			for package in program.packages:
				package.search_modules()
				for module in package.modules:
					module.search_code_fragments()
					# module.show_code_fragments()
			# program.make_backup()
			# program.get_logging_coverage()
			# src.util.print_program_summary(program)
			program.add_defined_in()
			program.export_data_code_fragments()
			# program.copy_src_files_to_data_path()
		# Extract data
		elif choice == '2':
			if program:
				data_loop = True
				while data_loop:
					data_choice = src.util.print_data_menu()
					# Get dependencies
					if data_choice == '1':
						print('Extracting static dependencies..')
						for package in program.packages:
							package.add_dependencyManager()
						dependencyManager.allocate_dependencies_to_modules(program)
						dependencyManager.allocate_dependencies_to_code_fragments(program)
						dependencyManager.dependencies_to_module_code_fragments(program)
						src.util.print_module_dependencies(program)
						# src.util.print_code_fragments(program)
						# src.util.make_bag_of_dependencies(program)
						# bod = dependencyManager.make_bod(program.get_all_code_fragments())
						dependencyManager.get_dependency_edges(program)
						program.export_dependencies_code_fragments()
					# Extract semantics
					elif data_choice == '2':
						print('Extracting semantics..')
						semanticManager.allocate_semantics_to_code_fragments(program)
						bow = semanticManager.make_bow(program.get_all_code_fragments())
						semanticManager.get_semantic_edges(program)
						program.export_vocab_code_fragments()
					# Instrument logging
					elif data_choice == '3':
						print('Instrumenting logging..')
						program.insert_log_files()
						for package in program.packages:
							for module in package.modules:
								module.add_loggingManager()
								module.loggingManager.instrument_logging()
						print('Codebase successfully updated. You can run tests now to capture logging.')
						# src.run_tests.main()
					# Load logging
					elif data_choice == '4':
						print('Load logging..')
						# try:
						logs = LoggingManager.load_logging(program)
						LoggingManager.get_dynamic_edges(program)
						program.export_dynamics_code_fragments()
						# if s.USE_K_FUNCTION_PERMUTATIONS:
						# 	traces = traceManager.make_traces_k_function_permutations(logs)
						# if s.USE_K_FUNCTION_PERMUTATIONS_STARTOVER:
						# 	traces = traceManager.make_traces_k_function_permutations_startover(logs)
						# if s.USE_K_FUNCTIONS:
						# 	traces = traceManager.make_traces_k_functions(logs)
						# traceManager.allocate_traces_to_code_fragments(program, traces)
						# traceManager.make_bot(program.get_all_code_fragments(), traces)
						print('Logging successfully loaded.')
						# except:
							# print('Something went wrong.')
						# src.util.make_bag_of_dynamics(program)
						# src.util.calculate_similarity(program)
					elif data_choice == '0':
						data_loop = False
					else:
						print('Wrong menu selection.')
			else:
				print('Please first select the target application.')
		# Make clustering
		elif choice == '3':
			if program:
				clustering_loop = True
				sim = None
				while clustering_loop:
					clustering_choice = src.util.print_clustering_menu()
					# Make similarity matrix
					if clustering_choice == '1':
						print('Compute similarity matrix..')
						# compute_semantic_similarity()
						# compute_dependency_similarity()
						# compute_trace_similarity()
						# sim = SimilarityMatrix(program.get_all_code_fragments(), cosine_similarity)
						print('Similarity matrix successfully constructed..')
						# sim.print_sim_matrix()
						# sim.export_sim_matrix()
					# Make clustering
					elif clustering_choice == '2':
						if sim:
							# result = Clustering(sim.sim_matrix)
							headers = program.get_all_code_fragments_names()
							# print(result.silhouette_score(sim.sim_matrix))
							# result.cohesion()
							# Clustering.print_clustering(result.spectral, headers)
							# Clustering.print_clustering(result.DBSCAN, headers)
							# Clustering.print_clustering(result.affinity, headers)
							# Clustering.print_clustering(result.hierarchical_res, headers)
							# print(Hierarchical().predict(sim.sim_matrix))
							# result.print_dendrogram()
						else:
							print('No similarity matrix available.')
					# View microservices
					elif clustering_choice == '3':
						pass
					elif clustering_choice == '0':
						clustering_loop = False
					else:
						print('Wrong menu selection.')
			else:
				print('Please first select the target application.')
		# View program
		elif choice == '4':
			if program:
				view_loop = True
				while view_loop:
					view_choice = src.util.print_view_menu()
					# Program structure
					if view_choice == '1':
						src.util.print_program_summary(program)
					# Code fragments
					elif view_choice == '2':
						src.util.print_code_fragments(program)
					# Graphs
					elif view_choice == '3':
						src.graph.execute_experiments(program)
						if program.interfaces:
							chm = 0
							chd = 0
							for interface in program.interfaces:
								chm += interface.chm()
								chd += interface.chd()
							print('N interfaces:', len(program.interfaces))
							print('CHM:', chm/len(program.interfaces))
							print('CHD:', chd/len(program.interfaces))
					elif view_choice == '0':
						view_loop = False
					else:
						print('Wrong menu selection.')
			else:
				print('Please first select the target application.')
		elif choice == '0':
			print('Exiting..')
			del program
			loop = False
		else:
			print('Wrong menu selection.')

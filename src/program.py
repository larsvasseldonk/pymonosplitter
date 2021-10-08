'''Program Script

This script allows the system to create a Program class.
'''

import os
import re
import shutil
import warnings
import json

import src.util

from src.settings import s
from src.package import Package

class Program:
	def __init__(self, name, root_path, src_path, logging=False):
		self.name = name
		self.root_path = root_path # Root path of the repository
		self.src_path = src_path # Path of the folder that contains all the code
		self.backup_path = os.path.join(s.DATA_PATH, self.name + '_BACKUP') # The backup path contains the code with comments includes. These comments are deleted in the orginal file after adding logging information.
		self.packages = set()
		self.sim_matrix = None
		self.bag_of_dependencies = []
		self.bag_of_logging = []
		self.bag_of_semantics = []
		self.interfaces = []
		self.static_coverage = {}
		self.semantic_coverage = {}
		self.dynamic_coverage = {}

	def add_package(self, package):
		'''Add Package instance'''
		self.packages.add(package)

	def remove_package(self, package):
		'''Remove Package instance'''
		self.packages.discard(package)

	def search_packages(self):
		'''
		Search for all packages inside the repository. A package in Python
		is a directory with .py files and a file with the name __init__.py.
		'''
		print('Search for packages..')
		has_package = False
		_, src_name = os.path.split(self.src_path) # Get everything after the last slash
		# Walk over each folder in the program
		for root, _, files in os.walk(self.src_path, topdown=True):
			# If the folder has a __init__ file, we consider it a package
			if '__init__.py' in files:
				name = root.replace(self.src_path, '')
				name = src_name + name.replace(os.sep, '.')
				# _, package = os.path.split(root) # Get everything after the last slash
				has_package = True
				self.add_package(Package(name, root))
		if not has_package:
			warnings.warn('The program does not have any packages!')

	# Filter out modules that have no
	def get_all_code_fragments(self):
		return list(set([cf for pkg in self.packages for mod in pkg.modules for cf in mod.code_fragments]))

	def get_all_code_fragments_to_be_clustered(self, setting='bound_functions_to_modules'):
		'''This are all the code fragments that will be clustered!'''
		cfs_to_be_clustered = set()
		for package in self.packages:
			for module in package.modules:
				for code_fragment in module.code_fragments:
					if code_fragment.is_first:
						if code_fragment.type == 'module':
							n_nested_functions = len([elem.name for elem in code_fragment.get_all_nested_code_fragments([]) if elem.type == 'function'])
							if code_fragment.dependencyManager.depend or code_fragment.dynamic_dependencies or n_nested_functions:
								cfs_to_be_clustered.add(code_fragment)
						if code_fragment.type == 'class': # This means it is defined outside a class
							cfs_to_be_clustered.add(code_fragment)
		return list(cfs_to_be_clustered)

	def get_all_code_fragments_names(self):
		return [code_fragment.name for code_fragment in self.get_all_code_fragments()]

	def add_defined_in(self):
		for code_fragment in self.get_all_code_fragments():
			code_fragment.search_defined_in()

	def make_backup(self):
		if not os.path.exists(self.backup_path):
			os.mkdir(self.backup_path)
		if self.packages:
			for package in self.packages:
				for module in package.modules:
					shutil.copy2(module.path, self.backup_path)
		else:
			warnings.warn('The program does not have any packages!')
		
	def insert_log_files(self):
		'''
		Copy the log files (external_logger.py, external_logger.config) to the target app.
		'''
		print(f'Insert external_logger.py file at {self.src_path}')
		shutil.copy2(os.path.join(s.SRC_PATH, 'external_logger.py'), self.src_path)
		s.PROGRAM_IMPORT_MOD_NAME = os.path.basename(self.src_path) + '.external_logger' # {name}.external_logger
		# shutil.copy2(os.path.join(s.SRC_PATH, 'external_logger.conf'), self.src_path)

	def load_semantics(self):
		if os.path.exists(os.path.join(s.DATA_PATH, self.name + '_BoS.xlsx')):
			pass
		else:
			print('No semantic data found.')

	def code_fragments_dict(self):
		cf_dict = []
		for package in self.packages:
			for module in package.modules:
				for code_fragment in module.code_fragments:
					cf_dict.append(code_fragment.code_fragment_to_dict())
		return sorted(cf_dict, key=lambda k: int(k['id'][2:]))

	def vocab_code_fragments_dict(self):
		cf_dict = []
		for package in self.packages:
			for module in package.modules:
				for code_fragment in module.code_fragments:
					cf_dict.append(code_fragment.code_fragment_vocab_to_dict())
		return sorted(cf_dict, key=lambda k: int(k['id'][2:]))

	def export_data_code_fragments(self):
		final_dict = {
			'program': self.name,
			'path': self.src_path,
			'packages': [pkg.name for pkg in self.packages],
			'modules': [mod.name for pkg in self.packages for mod in pkg.modules],
			'statistics': {
				'n_modules': len([cf for cf in self.get_all_code_fragments() if cf.type == 'module']),
				'n_classes': len([cf for cf in self.get_all_code_fragments() if cf.type == 'class']),
				'n_functions': len([cf for cf in self.get_all_code_fragments() if cf.type == 'function']),
				'n_methods': len([cf for cf in self.get_all_code_fragments() if cf.type == 'method'])
			},
			'code_fragments': self.code_fragments_dict()
		}
		with open(os.path.join(s.DATA_PATH, s.program_name + '_code_fragments.json'), 'w') as f: 
			json.dump(final_dict, f, indent=4)

	def export_vocab_code_fragments(self):
		final_dict = {
			'program': self.name,
			'path': self.src_path,
			'statistics': {
				'modules_covered': self.semantic_coverage.get('modules_covered'),
				'module_coverage': self.semantic_coverage.get('module_coverage'),
				'classes_covered': self.semantic_coverage.get('classes_covered'),
				'class_coverage': self.semantic_coverage.get('class_coverage')
			},
			'code_fragments': self.vocab_code_fragments_dict()
		}
		with open(os.path.join(s.DATA_PATH, s.program_name + '_code_fragments_semantics.json'), 'w') as f: 
			json.dump(final_dict, f, indent=4)

	def export_dependencies_code_fragments(self):
		cf_dict = []
		for package in self.packages:
			for module in package.modules:
				for code_fragment in module.code_fragments:
					cf_dict.append(code_fragment.code_fragment_dependencies_to_dict())
		sorted_cf_dict = sorted(cf_dict, key=lambda k: int(k['id'][2:]))
		final_dict = {
			'program': self.name,
			'path': self.src_path,
			'statistics': {
				'modules_covered': self.static_coverage.get('modules_covered'),
				'module_coverage': self.static_coverage.get('module_coverage'),
				'classes_covered': self.static_coverage.get('classes_covered'),
				'class_coverage': self.static_coverage.get('class_coverage')
			},
			'code_fragments': sorted_cf_dict
		}
		with open(os.path.join(s.DATA_PATH, s.program_name + '_code_fragments_statics.json'), 'w') as f: 
			json.dump(final_dict, f, indent=4)

	def export_dynamics_code_fragments(self):
		cf_dict = []
		for package in self.packages:
			for module in package.modules:
				for code_fragment in module.code_fragments:
					cf_dict.append(code_fragment.code_fragment_dynamics_to_dict())
		sorted_cf_dict = sorted(cf_dict, key=lambda k: int(k['id'][2:]))
		final_dict = {
			'program': self.name,
			'path': self.src_path,
			'statistics': {
				'modules_covered': self.dynamic_coverage.get('modules_covered'),
				'module_coverage': self.dynamic_coverage.get('module_coverage'),
				'classes_covered': self.dynamic_coverage.get('classes_covered'),
				'class_coverage': self.dynamic_coverage.get('class_coverage')
			},
			'code_fragments': sorted_cf_dict
		}
		with open(os.path.join(s.DATA_PATH, s.program_name + '_code_fragments_dynamics.json'), 'w') as f: 
			json.dump(final_dict, f, indent=4)

	def get_logging_coverage(self):
		log_path = os.path.join(s.DATA_PATH, self.name + '_dynamics.log')
		if os.path.isfile(log_path):
			print('Logging found and analyzed..')
			log_code_fragments = []
			with open(log_path) as f:
				for i, line in enumerate(f.readlines()):
					log = line.split('::') # split on first comma occurence
					name = re.sub(r'<locals>\.', '', log[13]) # sometimes the log contains <locals> tokens that we want to remove
					# file = log[17].replace(self.runnable_src_path + os.sep, '') # C:\Users\larsv\SynologyDrive\Repos\Test\test\__init__.py gets test\__init__.py
					# file = file[:-3].replace(os.sep, '.') # \test\__init__.py -> test.__init__
					# Full name is file_name + qual_name
					file_path = log[17][:-3].replace(os.sep, '.')
					for i, elem in enumerate(reversed(file_path.split('.'))):
						if i == 0:
							file_name = elem
						else:
							file_name = elem + '.' + file_name
						if elem == os.path.basename(self.src_path):
							break

					# search_pool = [] 
					# for i, mod in enumerate(name.split('.')): # ['beets', 'util', 'as_string']
					# 	if i == 0:
					# 		temp = mod
					# 	else:
					# 		temp = temp + '.' + mod
					# 	search_pool.append(temp) # ['beets', 'beet.util', 'beets.util.as_string']

					# is_found = False
					# # Only get the unique part of the function name obtained from logging
					# # If the qualname already includes the root path, don't add this anymore
					# for m in reversed(search_pool):
					# 	if file_name.startswith(m):
					# 		full_name = file_name + name.replace(m, '')
					# 		is_found = True
					# 		break
					
					# if not is_found:
					# 	full_name = file_name + name

					# print('full', full_name)
					# print('name', name)
					# print('file', file_name)
					# print('new name', file_name + name)
					# name = re.match(r'(^twitter.)(.+?(?=\())', name).group(2) # search for the name of the code fragment
					log_code_fragments.append(file_name + name)
			log_code_fragments = set(log_code_fragments)
			print('All unique code fragments obtained from logging:', log_code_fragments)
			all_code_fragments = set(src.util.get_all_code_fragments(self))
			print('Length of log_code_fragments:', len(log_code_fragments))
			print('Code fragments that are missing:', all_code_fragments.difference(log_code_fragments))
			print('THIS SHOULD BE NONE', log_code_fragments.difference(all_code_fragments))
			both = log_code_fragments & all_code_fragments
			print('Intersection:', log_code_fragments & all_code_fragments)
			log_coverage = len(both)/len(all_code_fragments)
			print('Coverage of logging:', round(log_coverage, 2))

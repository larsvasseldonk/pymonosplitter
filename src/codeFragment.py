'''
Everything that I want to cluster is a code fragment.

Code fragment have the following characteristics:
- Highest level: module (contains all the code in the module)
- Lowest level: nested functions (the deepest function)
'''

import src.util

from src.semanticManager import semanticManager
# from src.trace_manager import traceManager
from src.dependencyManager import dependencyManager

from collections import Counter

import ast
import re
import json
import os

from src.settings import s

class CodeFragment:
	# Code Fragment ID counter
	id_counter = 0
	def __init__(self, name, code: list, ast, parent, start_lineno, end_lineno, input_params, output_params, filepath, is_first=False):
		self.id = 'cf' + str(CodeFragment.id_counter)
		self.name = name # Name of the CodeFragment: module_name + function_name (separated by a dot)
		self.code = code # List of lines of code
		self.ast = ast # Abstract Syntax Tree of code element. This way we can recognize a class or a function
		self.parent = parent # The parent CodeFragments, None value for modules
		self.filepath = filepath # The file in which the CF is implemented
		self.start_lineno = start_lineno
		self.end_lineno = end_lineno
		self.input_params = input_params
		self.output_params = output_params
		self.type = self.get_type()
		self.defined_in = None
		self.dynamic_dependencies = []
		self.semantic_edges = []
		self.dependency_edges = []
		self.dynamic_edges = []
		self.dependencyManager = None
		self.semanticManager = None
		self.trace_manager = None
		self.code_fragments = set()
		self.dynamics = [] # A vector representation of the dynamics, e.g. [0,1,0,0,1,1,0,0]: each row represents a trace
		self.bag_of_traces = None # variant1, variant2, etc.
		self.service_id = None
		self.is_operation = False
		self.is_first = is_first # If the code fragments is the first order in the module: this means nested functions are False
		CodeFragment.id_counter += 1

	def code_fragment_to_dict(self):
		x = lambda cf : cf.name if cf else cf
		cf_dict = {
				'id': self.id,
				'name': self.name,
				'type': self.type,
				'filepath': self.filepath,
				'start_lineno': self.start_lineno,
				'end_lineno': self.end_lineno,
				'input_params': self.input_params,
				'output_params': self.output_params,
				'defined_in': x(self.defined_in),
				'nested_functions': [cf.name for cf in self.get_all_nested_code_fragments([])]
		}
		return cf_dict

	def code_fragment_dependencies_to_dict(self):
		c = Counter(dependencyManager.depend_to_list(self.dependencyManager.depend))
		if self.dependencyManager:
			dep_dict = {
				'id': self.id,
				'name': self.name,
				'type': self.type,
				'destinations': [(elem, c[elem]) for elem in c],
				'internal_destinations': self.dependencyManager.internal_dependencies,
				'outgoing_edges': self.dependency_edges
			}
			return dep_dict

	def code_fragment_vocab_to_dict(self):
		c = Counter(self.semanticManager.vocab)
		if self.semanticManager:
			vocab_dict = {
				'id': self.id,
				'name': self.name,
				'type': self.type,
				'raw_vocab': [(elem, c[elem]) for elem in c],
				'tfidf_vocab': self.semanticManager.tfidf_dict,
				'outgoing_edges': self.semantic_edges
			}
			return vocab_dict

	def code_fragment_dynamics_to_dict(self):
		c = Counter(self.dynamic_dependencies)
		dynamics_dict = {
				'id': self.id,
				'name': self.name,
				'type': self.type,
				'dynamic_destinations': [(elem, c[elem]) for elem in c],
				'outgoing_edges': self.dynamic_edges
		}
		return dynamics_dict

	def get_type(self):
		cf_type = ''
		if isinstance(self.ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
			if isinstance(self.parent.ast, ast.ClassDef):
					cf_type = 'method'
			elif self.input_params:
				if self.input_params[0] == 'self':
					cf_type = 'method'
				else:
					cf_type = 'function'
			else:
				cf_type = 'function'
		elif isinstance(self.ast, ast.ClassDef):
			cf_type = 'class'
		elif isinstance(self.ast, ast.Module):
			cf_type = 'module'
		else:
			cf_type = self.ast
		return cf_type

	def get_list_of_name_terms(self):
		# Change Capitals to underscores
		name = self.name.split('.')[-1] # Last element in the name list
		name = re.sub(r'([A-Z])', r'_\1', name) # Split words that start with a capital
		names = name.split('_')
		return [name for name in names if name] # remove empty ones

	def nicely_print_code_fragments(self):
		cf_dict = self.code_fragment_to_dict()
		# Serializing and print JSON
		print(json.dumps(cf_dict, indent=4)) 

	def add_dependencyManager(self, dependencies, edges):
		self.dependencyManager = dependencyManager(dependencies, edges)

	def add_semanticManager(self):
		self.semanticManager = semanticManager(self.get_terms())

	# def add_trace_manager(self, traces):
		# self.trace_manager = traceManager(traces)

	def add_code_fragment(self, code_fragment):
		self.code_fragments.add(code_fragment)

	def get_terms(self):
		terms = []
		# The code of the classes already incorporates all its nested methods
		# However, the code of the module only includes code that is not defined in a function
		# Since functions are bounded to modules, we also incorporate this data
		if self.type == 'module':
			# We don't need all nested functions, since the first level already
			# take all the code
			nested_cfs = [cf for cf in self.code_fragments]
			for cf in nested_cfs:
				for term in cf.get_terms():
					terms.append(term)
				# The module name is not written inside the code so thus we 
				# add it manually.
				mod_name = self.name.split('.')[-1]
				terms.append(mod_name)
		if s.USE_ALL_CODE_SEMANTICS:
			for line in self.code:
				program_name = self.name.split('.')[0]
				if re.search(rf'^from {program_name}.external_logger', line.strip()):
					continue
				if re.search(r'^@method_logger|@function_logger', line.strip()):
					continue
				# Remove all special characters and number
				# Only keep text
				line = re.sub(r'[^A-Za-z]+', r' ', line.strip())
				# If line is not null
				if line:
					for item in line.split():
						terms.append(item)
		elif s.USE_ALL_NAMES_DOCSTRINGS:
			for node in ast.walk(self.ast):
				if isinstance(node, ast.Module):
					mod_docstring = ast.get_docstring(node, clean=True)
					if mod_docstring:
						terms.append(mod_docstring)
				if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
					# If the function/class has a docstring
					docstring = ast.get_docstring(node, clean=True)
					if docstring:
						terms.append(docstring)
				# This also adds the input and output parameters
				if isinstance(node, ast.Name):
					terms.append(node.id)
		elif s.USE_ONLY_IDENTIFIER_NAMES:
			docstring = ast.get_docstring(self.ast, clean=True)
			if docstring:
				terms.append(docstring)
			if self.input_params:
				for input in self.input_params:
					terms.append(input)
			if self.output_params:
				for output in self.output_params:
					terms.append(output)

		# Remove special characters
		terms = [re.sub(r'[^A-Za-z]+', r' ', term.strip()).lower() for term in terms]
		final_terms = []
		for term in terms:
			for i in term.split(sep=' '):
				if len(i) > 1:
					final_terms.append(i)
		del terms
		return final_terms

	def show_code(self):
		'''Display code of an module'''
		for code in self.code:
			# (name, identation, code)
			print(code)

	def get_all_nested_code_fragments(self, nested_cfs):
		# This means also nested nested functions
		if self.code_fragments:
			for cf in self.code_fragments:
				nested_cfs.append(cf)
				cf.get_all_nested_code_fragments(nested_cfs)
		return nested_cfs

	def get_cluster_code_fragment(self):
		if self.type == 'class' and self.is_first:
			cluster_cf = self
		elif self.type in ['function', 'method', 'class']:
			cluster_cf = self.defined_in
		elif self.type == 'module':
			cluster_cf = self
		return cluster_cf

	def search_defined_in(self):
		'''This searches for the code fragment in which the method or function is defined in.'''
		if self.type == 'method':
			parent = self.search_class()
		elif self.type in ['function', 'class']:
			parent = self.search_module()
		else:
			parent = None
		self.defined_in = parent

	def search_class(self):
		# The class code fragment has the property is_first = True
		# This can be used for methods
		if not self.is_first:
			return self.parent.search_class()
		else:
			return self

	def search_module(self):
		# The module code fragment does not have a module
		if self.parent:
			return self.parent.search_module()
		else:
			return self

	def has_no_behaviour(self):
		'''Return true if the code fragment has no behaviour'''
		# Classes and modules only having member variables but no member methods/functions. - Jin et al.
		if self.code_fragments:
			return False
		else:
			return True

	def has_3rd_party_service(self):
		if self.dependencyManager.depend_to_list(self.dependencyManager.depend):
			if not self.dependencyManager.internal_dependencies:
				# has only dependencies to external parts of the system
				return True
			else:
				return False
		else:
			False

	@staticmethod
	def search_code_fragment(program, filepath, function_name=None, lineno=None):
		'''Search for the code fragment based on the function name and the file it is 
		implemented in.'''
		code_fragments = program.get_all_code_fragments()
		for code_fragment in code_fragments:
			# If the file is the same and the name ends the same
			if code_fragment.filepath == filepath:
				if lineno:
					# If the line number falls within the range of the code fragment
					if lineno >= code_fragment.start_lineno and lineno <= code_fragment.end_lineno:
						return code_fragment
				if function_name:
					if code_fragment.name.endswith(function_name):
						return code_fragment

	@staticmethod
	def search_code_fragment_with_name(program, name):
		code_fragments = program.get_all_code_fragments()
		for code_fragment in code_fragments:
			if code_fragment.name == name:
				return code_fragment

	@staticmethod
	def search_code_fragment_with_id(program, id):
		code_fragments = program.get_all_code_fragments()
		for code_fragment in code_fragments:
			if code_fragment.id == id:
				return code_fragment		

	@staticmethod
	def get_all_possible_code_fragment_names(code_fragments_names):
		all_names = list()
		for code_fragment_name in code_fragments_names:
			names = src.util.get_name_combinations(code_fragment_name)
			for name in names:
				all_names.append(name)
		return all_names

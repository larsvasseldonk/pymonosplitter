
import ast

import src.util

from src.codeFragment import CodeFragment
from src.dependencyManager import dependencyManager
from src.loggingManager import LoggingManager



class Module:
	def __init__(self, name, path):
		self.name = name # name of the module e.g. temp (wihtout Python extension), e.g. beets.importer
		self.path = path # openable e.g. path/temp.py
		self.code = self.add_code() # A list of code lines (OLD: Represented in a tuple to also show identation: [(name, identation, code)])
		self.ast = self.add_ast() # Add abstract syntax tree
		self.code_fragments = set() # Set of unique code_fragment objects
		self.dependencyManager = None # None means dependencies are not set, {} means there are no dependencies found
		self.loggingManager = None
		self.semanticManager = None

	def add_code_fragment(self, code_fragment): # Single trailing underscore as variable is already taken by a keyword
		self.code_fragments.add(code_fragment)

	def show_code_fragments(self):
		for code_fragment in self.code_fragments:
			print('Name:', code_fragment.name)
			print('Class:', code_fragment.ast)

	def remove_code_fragment(self, code_fragment):
		self.code_fragment.discard(code_fragment)

	def add_dependencyManager(self, depend, edges):
		self.dependencyManager = dependencyManager(depend, edges)

	def add_loggingManager(self):
		self.loggingManager = LoggingManager(self.path, self.ast)

	def add_ast(self):
		with open(self.path, 'r', encoding='utf-8') as f:
			return ast.parse(f.read())
	
	def add_code(self):
		with open(self.path, 'r', encoding='utf-8') as f:
			return f.readlines()

	def show_code(self):
		'''Display code of an module'''
		for code in self.code:
			print(code)

	def search_code(self, start_lineno, end_lineno):
		return self.code[start_lineno-1:end_lineno]

	def search_nested_functions_classes(self, node, parent, is_first):
		'''Recursive function that searches for nested classes and functions. The function
		returns the indices of the code fragment.'''
		indices = []
		name = parent.name + '.' + node.name
		code = self.search_code(node.lineno, node.end_lineno)
		start_lineno = node.lineno
		end_lineno = node.end_lineno
		for index in list(range(node.lineno-1, node.end_lineno)):
			indices.append(index)
		# Search for function's input and output parameters. Classes cannot have them.
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
			input_params = search_input_params(node)
			output_params = search_output_params(node)
			cf = CodeFragment(name, code, node, parent, start_lineno, end_lineno, input_params, output_params, self.path, is_first)
		else:
			cf = CodeFragment(name, code, node, parent, start_lineno, end_lineno, [], [], self.path, is_first)
		# Add nested code_fragment 
		# Only add functions to modules
		if parent.type == 'module':
			if cf.type == 'function':
				parent.add_code_fragment(cf)
		else:
			parent.add_code_fragment(cf)
		# Add code fragment to module
		self.add_code_fragment(cf)
		# Search for nested functions
		for elem in node.body:
			if isinstance(elem, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
				indices += self.search_nested_functions_classes(elem, cf, is_first=False)
		return indices

	def search_code_fragments(self):
		res_code = self.code
		indices = []
		if len(self.ast.body) > 0:
			start_lineno = self.ast.body[0].lineno
			end_lineno = self.ast.body[-1].end_lineno
		else:
			start_lineno = 0
			end_lineno = 0
		cf_module = CodeFragment(self.name, res_code, self.ast, None, start_lineno, end_lineno, [], [], self.path, is_first=True)
		for node in self.ast.body:
			start_lineno = node.lineno
			if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
				# Save indices that belong to a code fragment
				indices += self.search_nested_functions_classes(node, cf_module, is_first=True)
		# Make set to only take unique indices
		# Reverse order since after removing an element the index decreases with 1
		for index in sorted(set(indices), reverse=True):
			if index < len(res_code):
				res_code.pop(index)
		# Change values of parent module
		cf_module.res_code = res_code
		self.add_code_fragment(cf_module)

	def search_first_order_code_fragments(self):
		for node in self.ast.body:
			if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
				pass
			if isinstance(node, (ast.ClassDef)):
				pass

	def add_dependencyManager_to_code_fragments(self):
		'''Add dependencyManagers to the code fragments'''
		added_keys = set()
		for code_fragment in self.code_fragments:
			depend = dict()
			classes = dict()
			functions = list()
			edges = list()
			# TWO CONFIGURATIONS
			# This regex captures the name of the code fragment and is necessary to make sure that
			# test2.test_redirect does not also includes test2.test_redirect_loop
			#            ^ code_fragment.name 			the start of the string has to match the code_fragment.name
			#								  (\.|$)	followed by either a point or the end of the string
			# pattern = rf'^{code_fragment.name}(\.|$)' # Parent does contain child dependencies
			# temp = re.match(r'[^.]*$', self.name).group() # get module name beets.util.hidden = hidden
			mod_name = self.name.split('.')[-1] # beets.util.hidden = hidden (hidden.py)
			mod_name = self.name.replace(mod_name, '') # beets.util.hidden = beets.util.
			cf_name = code_fragment.name.replace(mod_name, '')
			# print('Code fragment:', code_fragment.name, 'dependencies', self.dependencyManager.depend)
			# dependencies of the PyCG do not always have the full name
			# both options mean the same: beets.util.hidden.is_hidden becomes hidden.is_hidden
			# missing_fragments.append([key for key in self.dependencyManager.depend if not key == code_fragment.name or not key == cf_name])
			for key in self.dependencyManager.depend:
				if key == code_fragment.name or key == cf_name:
					added_keys.add(key)
					depend[key] = self.dependencyManager.depend[key]
			for key in self.dependencyManager.classes:
				if key == code_fragment.name or key == cf_name:
					classes[key] = self.dependencyManager.classes[key]
			for key in self.dependencyManager.functions:
				if key == code_fragment.name or key == cf_name:
					functions.append(key)
			for key in self.dependencyManager.edges:
				if key == code_fragment.name or key == cf_name:
					edges.append(key)
			code_fragment.add_dependencyManager(depend, classes, functions, edges)
			# print(code_fragment.name, code_fragment.dependencyManager.counter)
		all_keys = set([key for key in self.dependencyManager.depend])
		print(f'Warning: The following dependencies for {self.name} are detected by PyCG but not appended to a code fragment: {all_keys.difference(added_keys)}')

def search_output_params(node):
	'''Node should be an ast.FunctionDef or ast.AsyncFunctionDef variable.'''
	output_params = []
	# Search for output parameters in the body of the node
	for elem in node.body:
		# Search for return statements in the body of the function/method
		if isinstance(elem, ast.Return):
			for i in ast.walk(elem):
				# We can only get the name and not the type(s) since Python is dynamically typed
				if isinstance(i, ast.Name):
					output_params.append(i.id)
	return output_params

def search_input_params(node):
	'''Node should be an ast.FunctionDef or ast.AsyncFunctionDef variable.'''
	input_params = []
	# Search for input parameters in the args attribute
	for elem in ast.walk(node.args):
		if isinstance(elem, ast.arg):
			input_params.append(elem.arg)
	return input_params
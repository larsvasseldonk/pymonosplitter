
import os
from pycg.pycg import CallGraphGenerator

from src.module import Module
from src.dependencyManager import dependencyManager

class Package:
	def __init__(self, name, path):
		self.name = name # e.g. beets, beets.autotag
		self.path = path 
		self.modules = set()
		self.dependencyManager = None

	def add_module(self, module):
		self.modules.add(module)

	def remove_module(self, module):
		self.modules.discard(module)

	def search_modules(self):
		'''
		Search for all modules (Python files) that are included in
		the package. 
		'''
		print('Search for modules..')
		# Search for all modules of the package
		for root, _, files in os.walk(self.path, topdown=True):
			# Only add the modules that are part of this package
			if root == self.path:
				for file in files:
					if file.endswith('.py') and not file.endswith('external_logger.py'):
						name = self.name + '.' + file[:-3]
						print('Added:', name)
						self.add_module(Module(name, os.path.join(root, file)))

	def get_pycg_callgraph(self):
		module_paths = [module.path for module in self.modules]
		pycg_callgraph = CallGraphGenerator(module_paths, self.path)
		return pycg_callgraph

	def add_dependencyManager(self):
		'''Takes all dependencies of the package and creates a dependencyManager object'''
		print('Extract static dependencies for:', self.name)
		cg = self.get_pycg_callgraph()
		cg.analyze()
		self.dependencyManager = dependencyManager(cg.output(), cg.output_edges())

	def add_dependencyManager_to_modules(self):
		'''Add dependencyManagers to the modules'''
		# print('Package:', self.name, 'dependencies:', self.dependencyManager.depend)
		for module in self.modules:
			print('module:', module)
			depend = dict()
			classes = dict()
			functions = list()
			edges = list()
			mod_name = module.name.replace(self.name + '.', '') # beets.autotag.hooks becomes hooks
			for key in self.dependencyManager.depend:
				if key.startswith(mod_name or module.name):
					depend[key] = self.dependencyManager.depend[key]
			for key in self.dependencyManager.classes:
				if key.startswith(mod_name or module.name):
					classes[key] = self.dependencyManager.classes[key]
			for function in self.dependencyManager.functions:
				if function.startswith(mod_name or module.name):
					functions.append(function)
			for edge in self.dependencyManager.edges:
				if edge[0].startswith(mod_name or module.name):
					edges.append(edge)
			module.add_dependencyManager(depend, classes, functions, edges)

from collections import Counter

import src.util
import src.codeFragment

import re
import pandas as pd
import numpy as np
import os

from sklearn.decomposition import TruncatedSVD

from src.settings import s

class dependencyManager:
    def __init__(self, dependencies, edges):
        self.depend = dependencies # dict
        self.internal_dependencies = [] # dependencies to other code elements in the program.
        self.edges = edges # list of lists e.g. [[A, B],[B, C]]
        self.counter = Counter(self.depend_to_list(self.depend))
        self.destinations = []

    def show_dependencies(self):
        for key in self.depend:
            print(key, self.depend[key])

    def search_internal_dependencies(self, code_fragments):
        '''Internal dependencies are dependencies to other code elements in the program.'''
        for values in self.depend.values():
            if values:
                for value in values:
                    for code_fragment in code_fragments:
                        if code_fragment.name.endswith(value):
                            self.internal_dependencies.append(code_fragment.name)

    @staticmethod
    def get_dependency_edges(program):
        list_of_code_fragments = program.get_all_code_fragments()
        covered_classes = set()
        covered_modules = set()
        for code_fragment_A in list_of_code_fragments:
            # For each internal dependency
            # We only want relations to other functions and classes
            encapsulator_code_fragment_A = code_fragment_A.get_cluster_code_fragment()
            if not encapsulator_code_fragment_A is code_fragment_A:
                continue
            for each in code_fragment_A.dependencyManager.internal_dependencies:
                # CODE FRAGMENT B CAN ONLY BE A FUNCTION OR METHOD, NOT A MODULE OR CLASS
                code_fragment_B = src.codeFragment.CodeFragment.search_code_fragment_with_name(program, each)
                # Sanity check
                if code_fragment_A is code_fragment_B:
                    continue
                encapsulator_code_fragment_B = code_fragment_B.get_cluster_code_fragment()
                # Only outgoing edges
                # If the two are both defined in different fragments
                if encapsulator_code_fragment_A is encapsulator_code_fragment_B:
                    continue
                edge = [encapsulator_code_fragment_A.id, encapsulator_code_fragment_B.id, 1.0]
                # Calculating class and module coverage
                for cf in [encapsulator_code_fragment_A, encapsulator_code_fragment_B]:
                    if cf.type == 'class':
                        covered_classes.add(cf)
                    if cf.type == 'module':
                        covered_modules.add(cf)
                # if not encapsulator_code_fragment_A is code_fragment_A:
                #     # Check if the edge already exists
                #     edges_without_weight = [i[:2] for i in code_fragment_A.dependency_edges]
                #     if edge[:2] in edges_without_weight:
                #         for i, cf_edge in enumerate(code_fragment_A.dependency_edges):
                #             if edge[:2] == cf_edge[:2]:
                #                 code_fragment_A.dependency_edges[i][2] += 1
                #                 code_fragment_A.dependency_edges[i][3] += 1
                #     else:
                #         code_fragment_A.dependency_edges.append(edge)
                # Check if the edge already exists for the encapsulator
                # IMPORTANT BECAUSE THIS ARE THE ONES THAT WE WILL CLUSTER
                if edge[:2] in [i[:2] for i in encapsulator_code_fragment_A.dependency_edges]:
                    for i, cf_edge in enumerate(encapsulator_code_fragment_A.dependency_edges):
                        if edge[:2] == cf_edge[:2]:
                            encapsulator_code_fragment_A.dependency_edges[i][2] += 1
                            break
                else:
                    encapsulator_code_fragment_A.dependency_edges.append(edge)              
        # dependencyManager.normalise_dependency_weights(program)
        all_cfs = program.get_all_code_fragments_to_be_clustered()
        module_cfs = [cf for cf in all_cfs if cf.type == 'module']
        class_cfs = [cf for cf in all_cfs if cf.type == 'class']
        # program.static_coverage['total_code_fragments'] = len(all_cfs)
        # program.static_coverage['total_coverage'] = round((len(covered_classes) + len(covered_modules))/len(all_cfs),3)
        program.static_coverage['modules_covered'] = len(covered_modules)
        program.static_coverage['module_coverage'] = round(len(covered_modules)/len(module_cfs), 3)
        program.static_coverage['classes_covered'] = len(covered_classes)
        program.static_coverage['class_coverage'] = round(len(covered_classes)/len(class_cfs),3)


    @staticmethod
    def normalise_dependency_weights(program):
        print('Normalising weights...')
        list_of_code_fragments = program.get_all_code_fragments()
        weights = [edge[2] for cf in list_of_code_fragments for edge in cf.dependency_edges]
        max_weight = max(weights)
        for code_fragment in list_of_code_fragments:
            for edge in code_fragment.dependency_edges:
                updated_weight = edge[2]/max_weight
                edge[2] = round(updated_weight, 2)

    @staticmethod
    def depend_to_list(depend):
        depend_list = []
        for k in depend:
            for i in depend[k]:
                depend_list.append(str(i))
        return depend_list
    
    @staticmethod
    def search_depends_dictionary(names, dependencies):
        '''
        input:
            names: list
            dependencies: dict e.g. {'beets.hooks' : ['depend1', 'depend2']}
        output:
            dependencies: dict e.g. {'beets.hooks' : ['depend1', 'depend2']}
        '''
        depends = dict()
        for key, values in dependencies.items():
            # True if the name either matches one of the names or is followed by a dot
            # This way, we won't get orderdetail dependencies in order (they start the same)
            # if it starts with the name followed by a dot, or only the name
            if any(re.match(f'^{name}\.|^{name}$', key) for name in names):
                depends[key] = dependencies[key]
        return depends
    
    @staticmethod
    def search_edges_list(names, edges, code_fragments_names):
        '''
        input: names is a list, edges is a list
        output: list of lists e.g. [[A, B],[B, C]]
        '''
        new_edges = list()
        for edge in edges:
            if any(re.match(f'^{name}\.|^{name}$', edge[0]) for name in names):
                if edge[1] in code_fragments_names:
                    new_edges.append(edge)
        return new_edges

    @staticmethod
    def allocate_dependencies_to_modules(program):
        '''Add dependencyManagers to the modules'''
        code_fragments = program.get_all_code_fragments()
        code_fragments_names = [cf.name for cf in code_fragments]
        all_names = src.codeFragment.CodeFragment.get_all_possible_code_fragment_names(code_fragments_names)
        for package in program.packages:
            for module in package.modules:
                mod_names = src.util.get_name_combinations(module.name)
                depend = dependencyManager.search_depends_dictionary(mod_names, package.dependencyManager.depend)
                edges = dependencyManager.search_edges_list(mod_names, package.dependencyManager.edges, all_names)
                module.add_dependencyManager(depend, edges)

    @staticmethod
    def allocate_dependencies_to_code_fragments(program):
        '''Add dependencyManagers to the code fragments'''
        # code_fragments = program.get_all_code_fragments()
        code_fragments = program.get_all_code_fragments()
        code_fragments_names = [cf.name for cf in code_fragments]
        all_names = src.codeFragment.CodeFragment.get_all_possible_code_fragment_names(code_fragments_names)
        for package in program.packages:
            for module in package.modules:
                for code_fragment in module.code_fragments:
                    if code_fragment.type == 'module':
                        continue
                    cf_names = src.util.get_name_combinations(code_fragment.name)
                    depend = dependencyManager.search_depends_dictionary(cf_names, package.dependencyManager.depend)
                    edges = dependencyManager.search_edges_list(cf_names, package.dependencyManager.edges, all_names)
                    code_fragment.add_dependencyManager(depend, edges)
                    code_fragment.dependencyManager.search_internal_dependencies(code_fragments)
    
    @staticmethod
    def dependencies_to_module_code_fragments(program):
        code_fragments = [cf for cf in program.get_all_code_fragments() if cf.type == 'module']
        for code_fragment_module in code_fragments:
            module_dependencies = dict()
            for code_fragment_nested in code_fragment_module.code_fragments:
                for key, value in code_fragment_nested.dependencyManager.depend.items():
                    module_dependencies[key] = value
            code_fragment_module.add_dependencyManager(module_dependencies, [])
            code_fragment_module.dependencyManager.search_internal_dependencies(program.get_all_code_fragments())
    
    @staticmethod
    def show_missing_dependencies(program):
        pass

    @staticmethod
    def get_all_dependencies(code_fragments):
        return [list(cf.dependencyManager.depend.values()) for cf in code_fragments] 

    @staticmethod
    def export_dependencies(program):
        '''
        Export the dependencies in the form of a bag of dependencies
        '''
        all_dependencies = list()
        for code_fragment in program.get_all_code_fragments():
            all_dependencies.append(code_fragment.dependencyManager.depend.values())

    @staticmethod
    def make_bod(code_fragments):
        '''Make a bag of dependencies from a set of vocabs'''
        headers = list()
        bod = list()  # Bag of words
        dependencies = list()
        for cf in code_fragments:
            dependencies = dependencies + dependencyManager.depend_to_list(cf.dependencyManager.depend)
        dependencies = set(dependencies)
        for cf in code_fragments:
            headers.append(cf.name)
            bod.append(Counter(dependencyManager.depend_to_list(cf.dependencyManager.depend)))
        dependencies = list(dependencies)
        df = pd.DataFrame(bod, columns=dependencies, index=headers)
        print(df)
        df = df.to_numpy()
        print(df)
        dimensionality_reduction(df, dependencies, headers)
        return bod

def dimensionality_reduction(bod, dependencies, headers):
    # Change nan's to 0
    bod_no_nas = np.where(np.isnan(bod), 0, bod)
    svd = TruncatedSVD(n_components=3)
    svd.fit(bod_no_nas)
    for idx, topic in enumerate(svd.components_):
        # Get indices of highest lda components
        print("Topic ", idx, " ".join(dependencies[i] for i in topic.argsort()[:-5 - 1:-1]))
    feature_distr = svd.transform(bod_no_nas) # each row represents a code fragment
    df = pd.DataFrame(feature_distr, columns=['component1', 'component2', 'component3'], index=headers)
    df.to_excel(os.path.join(s.DATA_PATH, s.program_name + '_depend_distribution.xlsx'))
    for i, each in enumerate(feature_distr):
        print(headers[i], each)
        

    
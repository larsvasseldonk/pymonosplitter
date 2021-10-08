
import ast
import os
import re

import src.codeFragment

from src.settings import s

from src.codeFragment import CodeFragment

class LoggingManager:
    def __init__(self, path, ast):
        self.path = path
        self.ast = ast

    def instrument_logging(self):
        # First declare the objects that we want to instrument in the program
        # Create the function decorator as it is represented in the AST
        FunctionLoggerDec = ast.Name(id='function_logger', ctx=ast.Load())
        MethodLoggerDec = ast.Name(id='method_logger', ctx=ast.Load())
        ImportFromStat = ast.ImportFrom(module=s.PROGRAM_IMPORT_MOD_NAME, names=[ast.alias(name='function_logger'), ast.alias(name='method_logger')], level=0)
        imports_from = [elem for elem in self.ast.body if isinstance(elem, ast.ImportFrom)]
        if not any([import_from.module == s.PROGRAM_IMPORT_MOD_NAME for import_from in imports_from]):
            # The import statement always has to be after the __future__ import
            if any([import_from.module == '__future__' for import_from in imports_from]):
                if ast.get_docstring(self.ast):
                    self.ast.body.insert(2, ImportFromStat)
                else:
                    self.ast.body.insert(1, ImportFromStat)
            else:
                if ast.get_docstring(self.ast):
                    self.ast.body.insert(1, ImportFromStat)
                else:
                    self.ast.body.insert(0, ImportFromStat)            
        for node in ast.walk(self.ast):
            if isinstance(node, ast.FunctionDef):
                # Check if the function has a 'self'
                # If the function has no arguments, it cannot have self
                if not node.args.args:
                    if not any([decorator.id == 'function_logger' for decorator in node.decorator_list if isinstance(decorator, ast.Name)]):
                        node.decorator_list.append(FunctionLoggerDec)
                # When it has arguments, search for 'self'
                else:
                    if node.args.args[0].arg == 'self':
                        if not any([decorator.id == 'method_logger' for decorator in node.decorator_list if isinstance(decorator, ast.Name)]):
                            node.decorator_list.append(MethodLoggerDec)
                    else:
                        if not any([decorator.id == 'function_logger' for decorator in node.decorator_list if isinstance(decorator, ast.Name)]):
                            node.decorator_list.append(FunctionLoggerDec)
        self.update_code(ast.unparse(self.ast))

    def update_code(self, code):
        with open(self.path, 'w', encoding='utf-8', errors='ignore') as f:
            f.writelines(code)

#2021-07-16 15:46:28.381500;print_orders_menu;C:\Users\larsv\SynologyDrive\Repos\salie-backup\salie\util.py;<module>;20;C:\Users\larsv\SynologyDrive\Repos\salie-backup\salie\main.py
    @staticmethod
    def load_logging(program):
        log_path = os.path.join(s.DATA_PATH, program.name + '_dynamics.log')
        logs = list() # List of all consecutive logs
        if os.path.isfile(log_path):
            covered_classes = []
            covered_modules = []
            with open(log_path) as f:
                for line in f.readlines():
                    # Remove end of line token
                    line = line.rstrip('\n')
                    log = line.split(';')
                    if not len(log) == 6:
                        print('Wrong log found:', line)
                        continue
                    qualname_callee = re.sub(r'<locals>\.', '', log[1]) # sometimes the log contains <locals> tokens that we want to remove
                    # The callee is being called by the caller
                    # Thus, dependency: caller->callee
                    callee = CodeFragment.search_code_fragment(program, log[2], function_name=qualname_callee)
                    caller = CodeFragment.search_code_fragment(program, log[5], lineno=int(log[4]))
                    if caller and callee:
                        caller.dynamic_dependencies.append(callee.name)
                        encapsulator = caller.get_cluster_code_fragment()
                        if not encapsulator is caller:
                            encapsulator.dynamic_dependencies.append(callee.name)
                        for cf in [caller, callee]:
                            cf_type = cf.get_cluster_code_fragment().type
                            if cf_type == 'class':
                                covered_classes.append(cf)
                            elif cf_type == 'module':
                                covered_modules.append(cf)
                        logs.append((log[0], callee.name, caller.name))
                    else:
                        print('Error. Either caller or callee not found.', log)
        else:
            print('Error. We did not find any logs at:', log_path)
        return logs

    @staticmethod
    def get_dynamic_edges(program):
        list_of_code_fragments = program.get_all_code_fragments()
        covered_classes = set()
        covered_modules = set()
        for code_fragment_A in list_of_code_fragments:
            # For each internal dependency
            # We only want relations to other functions and classes
            encapsulator_code_fragment_A = code_fragment_A.get_cluster_code_fragment()
            if not encapsulator_code_fragment_A is code_fragment_A:
                continue
            for each in code_fragment_A.dynamic_dependencies:
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
                # Check if the edge already exists
                # edges_without_weight = [i[:2] for i in code_fragment_A.dynamic_edges]
                # if edge[:2] in edges_without_weight:
                #     for i, cf_edge in enumerate(code_fragment_A.dynamic_edges):
                #         if edge[:2] == cf_edge[:2]:
                #             code_fragment_A.dynamic_edges[i][2] += 1
                # else:
                #     code_fragment_A.dynamic_edges.append(edge)
                # Check if the edge already exists for the encapsulator
                # IMPORTANT BECAUSE THIS ARE THE ONES THAT WE WILL CLUSTER
                if edge[:2] in [i[:2] for i in encapsulator_code_fragment_A.dynamic_edges]:
                    for i, encap_edge in enumerate(encapsulator_code_fragment_A.dynamic_edges):
                        if edge[:2] == encap_edge[:2]:
                            # TODO: The value also changes for code_fragment_A.dynamic_edges[i][2]!!
                            encapsulator_code_fragment_A.dynamic_edges[i][2] += 1
                else:
                    encapsulator_code_fragment_A.dynamic_edges.append(edge)
        # LoggingManager.normalise_dynamic_weights(program)
        all_cfs = program.get_all_code_fragments_to_be_clustered()
        module_cfs = [cf for cf in all_cfs if cf.type == 'module']
        class_cfs = [cf for cf in all_cfs if cf.type == 'class']
        # uncovered_mods = module_cfs - covered_modules
        # uncovered_mods_no_behavior = []
        # for mod in uncovered_mods:
        #     if mod.has_no_behaviour:
        #         uncovered_mods_no_behavior.append(mod)
        # program.dynamic_coverage['modules_uncovered'] = len(uncovered_mods)
        # program.dynamic_coverage['modules_uncovered_no_behavior'] = len(uncovered_mods)
        program.dynamic_coverage['modules_covered'] = len(covered_modules)
        program.dynamic_coverage['module_coverage'] = round(len(covered_modules)/len(module_cfs), 3)
        program.dynamic_coverage['classes_covered'] = len(covered_classes)
        program.dynamic_coverage['class_coverage'] = round(len(covered_classes)/len(class_cfs), 3)

    @staticmethod
    def normalise_dynamic_weights(program):
        print('Normalising weights...')
        list_of_code_fragments = program.get_all_code_fragments()
        weights = [edge[2] for cf in list_of_code_fragments for edge in cf.dynamic_edges]
        max_weight = max(weights)
        for code_fragment in list_of_code_fragments:
            for edge in code_fragment.dynamic_edges:
                updated_weight = edge[2]/max_weight
                edge[2] = round(updated_weight, 2)

    @staticmethod
    def get_module_name(file_path, qualname, program):
        '''This is necessary because sometimes the __init__ module is not considered a module. 
        Therefore, we reconstruct it using the filepath.'''
        file_path = file_path.replace(os.sep, '.')
        for i, elem in enumerate(reversed(file_path.split('.'))):
            if i == 0:
                file_name = elem
            else:
                file_name = elem + '.' + file_name
            if elem == os.path.basename(program.src_path):
                break
        return file_name + qualname

    @staticmethod
    def get_caller(log):
        # example log: (2021-06-29 09:00:40.534, 11128, MainProcess, 6880, MainThread, (('salie.orderdetail', 'test1', 'C:\\Users\\larsv\\SynologyDrive\\Repos\\salie-backup\\salie\\orderdetail.py', [(1, <class 'int'>), (2, <class 'int'>)], (3, <class 'int'>)), ('main', 22, 'C:\\Users\\larsv\\SynologyDrive\\Repos\\salie-backup\\salie\\main.py')))
        caller_qualname = re.sub(r'<locals>\.', '', log[4])
        caller = (log[4]) # qualname
        print('callee', log[0])
        print('caller', log[5])

    def get_ast(self):
        # First declare the objects that we want to instrument in the program
        # Create the metaclass that will be added to the classes
        MethodLoggerMeta = ast.keyword(arg='metaclass', value=ast.Name(id='MethodLoggerMeta', ctx=ast.Load()))
        # Create the function decorator as it is represented in the AST
        FunctionLoggerDec = ast.Name(id='FunctionLoggerDec', ctx=ast.Load())
        # The import from statement that has to be instrumented
        ImportFromStat = ast.ImportFrom(module='externalLogger', names=[ast.alias(name='FunctionLoggerDec'), ast.alias(name='MethodLoggerMeta')], level=0)
        for file in self.files:
            print(file)
            # If you do w+ it deletes the content first
            with open(file, 'r+', encoding='utf-8', errors='ignore') as f:
                code = f.read()
                tree = ast.parse(code)
                print('code', ast.dump(tree))
                
                # ClassDefs = [elem for elem in tree.body if isinstance(elem, ast.ClassDef)]
                # FunctionDefs = [elem for elem in tree.body if isinstance(elem, ast.FunctionDef)]
                ImportFroms = [elem for elem in tree.body if isinstance(elem, ast.ImportFrom)]
                # Module = [elem for elem in tree.body if not isinstance(elem, (ast.ClassDef, ast.FunctionDef))]
                imports_from = list()
                module_level = list()
                class_level = list()
                function_level = list()
                for node in tree.body:
                    if isinstance(node, ast.ImportFrom):
                        imports_from.append(node)
                    if isinstance(node, ast.ClassDef):
                        class_level.append(node)
                        # Only add the metaclass MethodLoggerMeta if it does not already exists
                        if not any([keyword.value.id == 'MethodLoggerMeta' for keyword in node.keywords]):
                            node.keywords.append(MethodLoggerMeta)
                        continue          
                    if isinstance(node, ast.FunctionDef):
                        print('FUNCTION:', node.name)
                        print('LINENO START:', node.lineno)
                        print('LINENO ENDS:', node.end_lineno)
                        function_level.append(node)
                        if not any([decorator.id == 'FunctionLoggerDec' for decorator in node.decorator_list]):
                            node.decorator_list.append(FunctionLoggerDec)
                        continue
                    if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                        # Everything outside classes and functions
                        module_level.append(node)
                print('new import froms', imports_from)
                if not any([import_from.module == 'externalLogger' for import_from in imports_from]):
                    tree.body.append(ImportFromStat)

                # for node in ast.walk(tree):
                #     # Only take the functions that are not inside classes
                #     # This is necessary because otherwise the log @decorator does not work
                #     if isinstance(node, ast.FunctionDef) and (node not in class_functions):
                #         print('This is a function outside a class')
                #         fragment_code = ast.get_source_segment(code, node)
                #         print('fragment_code:\n', fragment_code)
                #         print('node type:', type(node))
                #         print('name', node.name)
                #         print('decorator', node.decorator_list)
                #         # If the function contains a decorator with id func_logger
                #         if not any([decorator.id == 'FunctionLoggerDec' for decorator in node.decorator_list]):
                #             node.decorator_list.append(FunctionLoggerDec)
                        # if FunctionLoggerDec not in node.decorator_list:
                        #     print('Does not have FunctionLoggerDec')
                        #     node.decorator_list.append(FunctionLoggerDec)
                        # if [decorator for decorator in node.decorator_list if decorator.id == 'func_logger']:
                        #     print('has func_logger')
                        #     print(node)
                        #     continue
                        # else:
                        #     node.decorator_list.append(ast.Name(id='func_logger', ctx=ast.Load()))

                new_code = ast.unparse(tree)
                f.seek(0) # To set the pointer in the file back to the beginning
                f.write(new_code)
                f.truncate() # In order to delete the overwritten text. It removes all of the file content after the specified number of bytes.
                print(new_code)
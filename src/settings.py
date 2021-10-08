
import os

class Settings:
	ROOT_PATH: str = f'C:{os.sep}Users{os.sep}larsv{os.sep}SynologyDrive{os.sep}Repos'
	SRC_PATH = os.path.join(ROOT_PATH, 'Thesis') + os.sep + 'src' + os.sep
	DATA_PATH: str = os.path.join(ROOT_PATH, 'Thesis') + os.sep + 'data' + os.sep
	# TARGET_PATH: str = os.path.join(ROOT_PATH, 'Twitter') + os.sep + 'twitter'
	LOG_PATH: str = os.path.join(ROOT_PATH, 'Thesis') + os.sep
	static_output_filename = '_static_raw.json'

	'''
	The kind data that will be extracted from the target application.
	'''
	USE_DYNAMIC: bool = True
	USE_STATIC: bool = True
	USE_SEMANTIC: bool = True

	global PROGRAM_IMPORT_MOD_NAME 
	PROGRAM_IMPORT_MOD_NAME = None # The name of the source folder. This to refer to it in the dynamic logging.

	''' TRACE SETTINGS '''
	LOG_WINDOW: int = 10 # The number of consecutive logs

	# Select one of the three settings
	USE_K_FUNCTIONS = False # K unique functions, where K is the LOG_WINDOW
	USE_K_FUNCTION_PERMUTATIONS = False
	USE_K_FUNCTION_PERMUTATIONS_STARTOVER = True

	# Select one of the three settings
	USE_ALL_CODE_SEMANTICS = True # Take the whole source code and transform it to a vocab #s1
	USE_ALL_NAMES_DOCSTRINGS = False # Take the all variable names and function/class names and docstrings #s2
	USE_ONLY_IDENTIFIER_NAMES = False # Only take the function/class names and docstrings #s3 (setting3)

	BOUND_FUNCTIONS_TO_MODULES = True

	# important when creating outgoing edges
	CLUSTER_SETTING_1 = True # THIS IS CLUSTERING CLASSES AND FUNCTIONS
	CLUSTER_SETTING_2 = False # THIS IS CLUSTERING CLASSES AND MODULES

	FREQUENCY_THRESHOLD = 0.03

	'''
	Do we need to inject the target application to capture logging?
	'''
	INSTRUMENT_LOGGING: bool = True

	program_name = None

s = Settings()
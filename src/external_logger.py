

"""
	To add logging functionality, we have to install the Twitter tool again:

	Go to cmd and type:
	python setup.py install
	
	DON'T CALL THE FILE logging.py, this causes errors. 

	Don't run python logger.py, but go to the parent folder and run: python -m twitter.logger.py
"""

import logging
import logging.config
import os
import inspect
import datetime
import re

log_conf_path = os.path.join("C:/Users/larsv/SynologyDrive/Repos/Thesis/src", 'external_logger.conf')
# The external_logger.config file is in the same directory as the file running this code:
logging.config.fileConfig(log_conf_path)
logger_dynamic = logging.getLogger('root')

# WHAT WE NEED:
# datetime, qualname_callee, filename_callee, name_caller, lineno_caller, filepath_caller
# 2021-06-30 10:10:22.360::print_products_menu::C:\Users\larsv\SynologyDrive\Repos\salie-backup\salie\util.py::[]::[]::71::C:\Users\larsv\SynologyDrive\Repos\salie-backup\salie\main.py
# This is equivalent to func = 
SPOS = 1
def function_logger(func):
	"""
		A decorator designed to automatically report 
		functions calls to a log file.

		CAPTURES:
			- Functions that may contain zero or more
			  than one arguments.
			- Methods with @staticmethod decorator
		DOES NOT CAPTURE:
			- Methods (contain 'self' argument)
	"""
	def inner(*args, **kwargs):
		ret = func(*args, **kwargs)
		logger_dynamic.info(f'{datetime.datetime.now()};{func.__qualname__};{func.__globals__["__file__"]};{inspect.stack()[SPOS][3]};{inspect.stack()[SPOS][2]};{inspect.stack()[SPOS][1]}')
		return ret

	return inner

def method_logger(method):
	"""
		A decorator designed to automatically report 
		method calls to a log file. In a method call, 
		we need to take care of the 'self'.

		CAPTURES:
			- Methods (functions that are associated 
			  with objects/classes) that are called
			  on an object.
			- Magic methods that start with __
			  (e.g. __call__, __init__)
		DOES NOT CAPTURE
			- Functions (do not contain 'self' argument)
	"""
	def inner(self, *args, **kwargs):
		ret = method(self, *args, **kwargs)
		logger_dynamic.info(f'{datetime.datetime.now()};{method.__qualname__};{method.__globals__["__file__"]};{inspect.stack()[SPOS][3]};{inspect.stack()[SPOS][2]};{inspect.stack()[SPOS][1]}')
		return ret

	return inner
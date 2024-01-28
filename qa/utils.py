from functools import wraps

import json
import os
import time

import sys

import pickle



def timed(func):
    """This decorator prints the execution time for the decorated function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Started: {}".format(func.__name__))
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Finished: {} in {}s".format(func.__name__, round(end - start, 2)))
        return result
    return wrapper
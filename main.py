import errno
import json
import sys

from lp.entity import ProblemInstance
from test import test

if __name__ == '__main__':
    test_name = str(sys.argv[1])
    try:
        with open('test/' + test_name + '.json') as json_file:
            data = json.load(json_file,
                             object_hook=lambda d: ProblemInstance(obj=d['obj'], c=d['c'], A=d['A'],
                                                                   b=d['b'], sense=d['sense']))
        test.run(data)
    except IOError:
        print('No test file with this name!')

import sys

from test import test_01, test_02

if __name__ == '__main__':
    test = str(sys.argv[1])
    if test == 'test_01':
        test_01.run()
    elif test == 'test_02':
        test_02.run()
    else:
        print('No tests to run with name: {0}'.format(test))


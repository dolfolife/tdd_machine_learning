#!/usr/bin/python
''' run all tests for project '''

import unittest

if __name__ == "__main__":
    ALL_TESTS = unittest.TestLoader().discover('linear_regression', pattern='**/*test.py')
    unittest.TextTestRunner().run(ALL_TESTS)

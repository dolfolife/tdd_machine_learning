''' Linear Regression Tests '''
import unittest

from squared_error import SquaredError

class SquaredErrorTest(unittest.TestCase):
    ''' Test for Squared Error cost function '''

    def setUp(self):
        self.training_set = {"1": 1, "2": 2, "3": 3}
        self.subject = SquaredError

    def test_thetas_0(self):
        '''
           it calculates the squared difference of the training set
           and its hypotesis
        '''
        squared_error = self.subject.calculate_error(self.training_set, 0, 0)
        self.assertEqual(squared_error, 14.0/6.0)

    def test_trains(self):
        '''
        Trains returns new model with trained thetas
        '''
        goal = self.subject.goal(self.training_set,
                                 range(-20, 20),
                                 range(-20, 20))
        self.assertEqual(goal, [0, 1])

    def test_hypotesis(self):
        '''
        Hypotesis evaluates a value given the thetas
        '''
        self.assertEqual(5, self.subject.hypotesis(5, 0, 1))

if __name__ == '__main__':
    unittest.main()

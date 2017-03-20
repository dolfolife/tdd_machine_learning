''' Linear Regression Tests '''
import unittest

from linear_regression import LinearRegression

class SquaredErrorTest(unittest.TestCase):
    ''' Test for Squared Error cost function for linear regression'''

    def setUp(self):
        self.training_set = {"1": 1, "2": 2, "3": 3}
        self.squared_error_model = LinearRegression('SquaredError')

    def test_error_for_wrong_cost(self):
        ''' it raise an error for cost functions that are not implemented '''
        self.assertRaises(ValueError, LinearRegression, 'FooCostType')

    def test_thetas_0(self):
        '''
           it calculates the squared difference of the training set
           and its hypotesis
        '''
        cost_function = self.squared_error_model.cost_function
        squared_error = cost_function.calculate_error(self.training_set, 0, 0)
        self.assertEqual(squared_error, 14.0/6.0)

    def test_trains(self):
        '''
        Trains returns new model with trained thetas
        '''
        model = LinearRegression('SquaredError')
        model_trained = model.train(self.training_set,
                                    range(-20, 20),
                                    range(-20, 20))
        self.assertEqual(model_trained.theta_0, 0)
        self.assertEqual(model_trained.theta_1, 1)

    def test_evaluate(self):
        '''
        Evaluate predict value for new input after trained
        '''
        model = LinearRegression('SquaredError', 0, 1)
        self.assertEqual(5, model.evaluate(5))

    def test_evaluate_error(self):
        '''
        Evaluate raise an error when model has not been trained
        '''
        self.assertRaises(ValueError, self.squared_error_model.evaluate, 5)

if __name__ == '__main__':
    unittest.main()

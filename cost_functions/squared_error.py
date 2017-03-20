import math
from functools import reduce

class SquaredError(object):
    ''' implementation of the Squared Error cost function '''

    @staticmethod
    def hypotesis(input_value, theta_0, theta_1):
        ''' hypotesis to evaluate input_value with the passed thetas '''
        return theta_0 + (theta_1 * input_value)

    @staticmethod
    def calculate_error(training_set, theta_0, theta_1):
        ''' calculates the error of the hypotesis with current values '''
        errors = \
        [math.pow(SquaredError.hypotesis(int(key), theta_0, theta_1) - value, 2)
         for key, value in training_set.items()]
        return reduce((lambda x, y: x + y), errors) * 1.0/(2*len(errors))

    @staticmethod
    def goal(training_set, range_theta_0, range_theta_1):
        '''
        returns best combination of [theta_0, theta_1] given a range of
        values for theta_0 and theta_1
        '''
        return min([[[theta_0, theta_1],
                    SquaredError.calculate_error(training_set,
                                                 theta_0,
                                                 theta_1)]
                    for theta_0 in range_theta_0
                    for theta_1 in range_theta_1], key=lambda x: x[1])[0]


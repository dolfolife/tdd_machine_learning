''' Linear regression algorithms '''
from cost_functions.squared_error import SquaredError

class LinearRegression(object):
    ''' Linear Regression Model '''

    def __init__(self, cost_function='SquaredError', theta_0=None, theta_1=None):
        ''' create a base class cost function '''
        if cost_function == 'SquaredError':
            self.cost_function = SquaredError
        else:
            raise ValueError('Cost Function not implemented')
        self.theta_0 = theta_0
        self.theta_1 = theta_1

    def evaluate(self, value):
        ''' returns the prediction of the new value '''
        if self.theta_0 is None or self.theta_1 is None:
            raise ValueError('Model needs to be trained before evaluating')
        return self.cost_function.hypotesis(value, self.theta_0, self.theta_1)

    def train(self, training_set, range_theta_0, range_theta_1):
        '''
        Train Linear Regression model
        @return a new Linear Regression with thetas assigned
        '''
        goal = self.cost_function.goal(
            training_set,
            range_theta_0,
            range_theta_1)
        return LinearRegression(self.cost_function.__name__,
                                goal[0],
                                goal[1])

"""Module to implement exercise 3 functionality"""
import math
import unittest
import random
import numpy as np
import matplotlib.pyplot as plt

def run():
    """Runs the tasks in order to solve exercise 3"""

    # Exercise parameters
    lower = 0
    upper = 4 * math.pi
    number_of_points = 100

    print(f"Splitting [{lower}, {upper}] interval in {number_of_points} equidistant points")
    values = split_interval_equidistant_points(lower, upper, number_of_points)

    print(f"Mapping values to the three functions")
    sin_values, cos_values, complex_function_values = map_values_to_functions(values)

    print(f"Values are: {values}")
    print(f"Sin values are: {sin_values}\n")
    print(f"Cos values are: {cos_values}\n")
    print(f"tanh(sin + cos) values are: {complex_function_values}\n")

    print("Plotting values")
    plot_three_functions(values, sin_values, cos_values, complex_function_values)

def split_interval_equidistant_points(lower: float, upper: float, number_of_points: int) -> list[float]:
    """
    Splits [lower, upper] in number_of_points points which are equidistant
    lower and upper are included in the returned list
    upper has to be greater than lower
    number_of_points has to be greater or equal 2
    """

    # Safety checks
    if number_of_points <= 1:
        raise Exception("Number of points has to be a positive non-cero integer")
    if upper < lower:
        raise Exception("upper has to be greater than lower")

    values = []
    step = (upper - lower) / (number_of_points - 1)
    for i in range(0, number_of_points):
        current_value = lower + step * i
        values.append(current_value)

    return values


def map_values_to_functions(values):
    """Map values (which are x_values) to the three given functions in exercise
    specification"""

    sin_values = [np.sin(x) for x in values]
    cos_values = [np.cos(x) for x in values]
    complex_function_values = [np.tanh(np.sin(x) + np.cos(x)) for x in values]

    return sin_values, cos_values, complex_function_values

def plot_three_functions(values, sin_values, cos_values, complex_function_values):
    """
    Plot the three functions as specified

    I read on matplotlib's documentation how to change colors and line style
    https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    """

    # Change x axis values to trig values
    # Docstring says from where I copied the function
    set_x_axis_scale_to_pi()

    # En verde, con lineas discontinuas
    plt.plot(values, sin_values, "--g")

    # En negro, con lineas discontinuas
    plt.plot(values, cos_values, "--k")

    # En rojo, con lineas discontinuas
    plt.plot(values, complex_function_values, "--r")
    plt.show()

def wait_for_user_input():
    input("Press ENTER to continue...")


def set_x_axis_scale_to_pi():
    """
    Sets x axis to display 1/2Pi, 1Pi, 3/2PI, 2Pi, ... instead of 1, 2, 3...
    Useful for plotting trig functions
    Code copied from: https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
    """

    # To get the formatting
    def format_func(value, tick_number):
        # find number of multiples of pi/2
        N = int(np.round(2 * value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi/2$"
        elif N == 2:
            return r"$\pi$"
        elif N % 2 > 0:
            return r"${0}\pi/2$".format(N)
        else:
            return r"${0}\pi$".format(N // 2)

    _, ax = plt.subplots()

    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))



# Unit testing
#===============================================================================

class TestSplitInterval(unittest.TestCase):
    def random_lower_upper_and_number_of_points(self):
        num_vals = random.randint(2, 400)
        lower = random.uniform(-10.0, 10.0)
        upper = lower + random.uniform(0.0, 5.0) # Upper has to be greater than lower

        return lower, upper, num_vals

    def test_len_of_returned_list(self):
        self.number_of_tests = 1000
        for _ in range(self.number_of_tests):
            lower, upper, num_vals = self.random_lower_upper_and_number_of_points()
            values = split_interval_equidistant_points(lower, upper, num_vals)
            self.assertEquals(num_vals, len(values))

    def test_values_are_equidistant(self):
        self.number_of_tests = 1000
        for _ in range(self.number_of_tests):
            lower, upper, num_vals = self.random_lower_upper_and_number_of_points()
            values = split_interval_equidistant_points(lower, upper, num_vals)

            common_distance = values[1] - values[0]
            for index in range(1, len(values)):
                curr_distance = values[index] - values[index-1]
                self.assertAlmostEqual(common_distance, curr_distance)


    # For this test I read the answer of this stackoverflow question:
    # https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
    def test_raise_when_bad_num_of_points(self):
        lower = 0
        upper = 1

        number_of_points = 0
        with self.assertRaises(Exception) as context:
            split_interval_equidistant_points(lower, upper, number_of_points)
        self.assertEqual("Number of points has to be a positive non-cero integer", str(context.exception))

        number_of_points = 1
        with self.assertRaises(Exception) as context:
            split_interval_equidistant_points(lower, upper, number_of_points)
        self.assertEqual("Number of points has to be a positive non-cero integer", str(context.exception))

        # Here should not fail
        number_of_points = 2
        try:
            split_interval_equidistant_points(lower, upper, number_of_points)
        except Exception as e:
            self.fail(f"ERROR! This function should not raise exception with {number_of_points} number of points")

    def test_raise_when_upper_not_greater_than_lower(self):
        lower = 1
        upper = 0
        number_of_points = 100
        with self.assertRaises(Exception) as context:
            split_interval_equidistant_points(lower, upper, number_of_points)
        self.assertEqual("upper has to be greater than lower", str(context.exception))

        lower = 1
        upper = -1
        number_of_points = 100
        with self.assertRaises(Exception) as context:
            split_interval_equidistant_points(lower, upper, number_of_points)
        self.assertEqual("upper has to be greater than lower", str(context.exception))



    def test_upper_and_lower_in_result(self):
        self.number_of_tests = 1000
        for _ in range(self.number_of_tests):
            lower, upper, num_vals = self.random_lower_upper_and_number_of_points()
            values = split_interval_equidistant_points(lower, upper, num_vals)

            self.assertAlmostEqual(lower, values[0])
            self.assertAlmostEqual(upper, values[-1])

    def test_some_defined_intervals(self):
        lower = 0
        upper = 1
        number_of_points = 4
        expected = [0, 1/3, 2/3, 1]
        actual = split_interval_equidistant_points(lower, upper, number_of_points)
        self.assertAlmostEqual(actual, expected)

        lower = 0
        upper = 1
        number_of_points = 5
        expected = [0, 1/4, 2/4, 3/4, 1]
        actual = split_interval_equidistant_points(lower, upper, number_of_points)
        self.assertAlmostEqual(actual, expected)

        lower = 5
        upper = 7
        number_of_points = 3
        expected = [5, 6, 7]
        actual = split_interval_equidistant_points(lower, upper, number_of_points)
        self.assertAlmostEqual(actual, expected)


# In order to run exercise3, run python3 main.py which calls run function
# If you run python3 ejercicio3.py you will be running the test suit of this module
if __name__ == "__main__":
    unittest.main()

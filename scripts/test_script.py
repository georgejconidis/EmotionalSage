# This is a test scrip to see how to import a function the author has written into the Flask web app

import numpy as np


def test_round(number):
    """
    This function will round the number given using the function np.round
    """
    return np.round(number)

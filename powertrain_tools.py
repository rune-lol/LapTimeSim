import numpy as np
import matplotlib.pyplot as plt

def get_coefficients(x_rpm, t_norm):
    return np.polyfit(x_rpm, t_norm, deg=len(x_rpm))
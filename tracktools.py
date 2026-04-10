import math
import numpy as np
import pandas as pd # type: ignore

import matplotlib.pyplot as plt

from carModel import CarState

def get_track(trackFileLocation):
    """Track is a list of segments, a tuple with arc length and kappa curvature (ds, kappa)"""
    track = []
    
    data = pd.read_csv(trackFileLocation)

    x = data["# x_m"].to_numpy()
    y = data["y_m"].to_numpy()
    
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)

    # First derivatives
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)

    # Second derivatives
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    # Curvature
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5

    for dS, kappa in zip(ds, curvature):
        track.append((dS, kappa))

    return track

def calcTotalTime(track, car_states):
    t = 0
    for i, seg in enumerate(track):
        if car_states[i].v != 0:
            t += seg[0]/car_states[i].v
    return t

def getTotalDistance(track):
    s = 0
    for seg in track:
        s += seg[0]
    return s
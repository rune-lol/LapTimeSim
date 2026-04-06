import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def get_track(trackFileLocation):
    """Track is a list of segments, a tuple with arc length and kappa curvature (ds, kappa)"""
    track = []
    
    data = pd.read_csv(trackFileLocation)

    print(data.head)

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

def draw_track(track, speed=None):
    if speed is None:
        speed = [0]*len(track)

    heading = [0]*len(track)
    prev_heading = 0
    for i, seg in enumerate(track):
        heading[i] = seg[1]*seg[0] + prev_heading
        prev_heading = heading[i]

    x = []
    y = []
    lastCoord = (0, 0)
    for i, _ in enumerate(track):
        x_n = lastCoord[0] + track[i][0]*math.cos(heading[i])
        y_n = lastCoord[1] + track[i][0]*math.sin(heading[i])
        x.append(x_n)
        y.append(y_n)
        new_coord = (x_n, y_n)
        lastCoord = new_coord

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(x, y, c=speed, cmap='plasma')
    plt.colorbar(label="Speed")
    plt.axis('equal')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Track Speed")
    plt.subplot(122)
    plt.scatter(x, y, c=[seg[1] for seg in track], cmap='viridis')
    plt.colorbar(label="Curvature κ")
    plt.axis('equal')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Track layout")
    plt.show()

def calcTotalTime(track, speedmap):
    t = 0
    for i, seg in enumerate(track):
        if speedmap[i] != 0:
            t += seg[0]/speedmap[i]
    return t

def getTotalDistance(track):
    s = 0
    for seg in track:
        s += seg[0]
    return s
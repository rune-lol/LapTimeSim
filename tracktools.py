import math

import matplotlib.pyplot as plt

def get_track():
    track = []
    for _ in range(0, 100):
        track.append((1, 0)) # ds (1m), kappa (1/r)
    for _ in range(0, 100):
        track.append((1, 0.01)) # ds (1m), kappa (1/r) (1/100)
    for i in range(0, 50):
        track.append((1, -0.001 + i*0.001)) # ds (1m), kappa (1/r) (1/100 - 1/9999999)
    for i in range(0, 50):
        track.append((1, 0.001 - i*0.001)) # ds (1m), kappa (1/r) (1/100 - 1/9999999)
    for _ in range(0, 100):
        track.append((1, 0.01)) # ds (1m), kappa (1/r) (1/100)
    for _ in range(0, 100):
        track.append((1, 0)) # ds (1m), kappa (1/r)
    
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

    plt.figure()
    plt.scatter(x, y, c=speed, cmap='plasma')
    plt.colorbar(label="Speed")
    plt.axis('equal')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Track layout")
    plt.show()

def calcTotalTime(track, speedmap):
    t = 0
    for i, seg in enumerate(track):
        try:
            t += seg[0]/speedmap[i]
        except ZeroDivisionError:
            print('Passing ZeroDivisionError in the total time calculator')
    return t

def getTotalDistance(track):
    s = 0
    for seg in track:
        s += seg[0]
    return s
import math

from tracktools import get_track

G = 9.81 #m/s^2

car = {
    'mu': 0.8, #mu
    'P_max': 100_000, #W
    'm': 1000, #kg
}

def get_max_corner_speeds(track):
    """a_lat = v^2 * kappa"""
    v_max = []*len(track)
    for i in track:
        v_max[i] = math.sqrt((G*car['mu'])/track[1])

    return v_max


def main():
    track = get_track()

    v_max = get_max_corner_speeds(track)
    print(v_max)

    pass
import math

from tracktools import get_track, draw_track, calcTotalTime, getTotalDistance

G = 9.81 #m/s^2
rho_air = 1.225 #kg/m^3

car = {
    'mu': 0.8, #mu
    'P_max': 100_000, #W
    'm': 1000, #kg
    'ClA': -2.5 #m^2 * Cl (high downforce)
}

def get_local_vmax(kappa):
    return math.sqrt((car['mu']*G)/abs(kappa))

def get_max_corner_speeds(track):
    """a_lat = v^2 * kappa"""
    v_max = [0]*len(track)
    print(len(track), len(v_max))
    for i, segment in enumerate(track):
        try:
            v_max[i] = get_local_vmax(segment[1])
        except ZeroDivisionError:
            v_max[i] = 999_999_999 # absurdly high speed to cap, no car will ever reach

    return v_max


def main():
    track = get_track()

    v_max = get_max_corner_speeds(track)

    # actual solver
    v = v_max.copy()

    v[0] = 0 #start speed (from a dig)
    epoch = 1
    while epoch <= 100:
        forward_propagation(track, v)
        backward_propagation(track, v)
        
        print('Ran',epoch,'epochs')
        epoch += 1

    print('t:',calcTotalTime(track, v),'s')
    print('s:',getTotalDistance(track),'m')
    print('v_avg:',(getTotalDistance(track)/calcTotalTime(track, v))*3.6,'km/h')
    draw_track(track, v_max)
    draw_track(track, v)

def backward_propagation(track, v):
    for j, v_seg in enumerate(reversed(v)):
        i = len(v)-j-1
        F_lat = car['m']*(v_seg**2)*track[i][1]
        F_total_available = car['mu']*car['m']*G
        F_longLeftover = math.sqrt(max(F_total_available**2 - F_lat**2, 0))

        accel_brake = -F_longLeftover/car['m']
        v_prev = math.sqrt(v_seg**2 - 2*accel_brake*track[-i][0])
        if i-1 >= 0:
            v[i-1] = min(v[i-1], v_prev)

def forward_propagation(track, v):
    for i, v_seg in enumerate(v):
        F_lat = car['m']*(v_seg**2)*track[i][1]
        F_total_available = car['mu']*car['m']*G
        F_longLeftover = math.sqrt(max(F_total_available**2 - F_lat**2, 0))
        
        accel_grip_limited = F_longLeftover/car['m']
        try:
            accel_power_limited = car['P_max']/(abs(v_seg)*car['m'])
        except ZeroDivisionError:
            accel_power_limited = accel_grip_limited
        accel = min(accel_grip_limited, accel_power_limited)
        v_next = math.sqrt(v_seg**2 + 2*accel*track[i][0])
        try:
            v[i+1] = min(v[i+1], v_next)
        except IndexError:
            pass

if __name__ == "__main__":
    main()
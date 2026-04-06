import math

from tracktools import get_track, draw_track, calcTotalTime, getTotalDistance

G = 9.81 #m/s^2
RHO_AIR = 1.225 #kg/m^3
STRAIGHT_LINE_CAP = 1_000

car = {
    'mu': 1.7, #mu
    'P_max': 600_000, #W
    'm': 750, #kg
    'ClA': -9.1, #m^2 * Cl (high downforce)
    'CdA': 1.4 #m^2 * Cl (high downforce)
}

def get_fz(v):
    return car['m']*G-0.5*RHO_AIR*(v**2)*car['ClA']

def get_drag(v):
    return 0.5*RHO_AIR*(v**2)*car['CdA']

def get_local_vmax(kappa):
    if kappa < 1e-6:
        return STRAIGHT_LINE_CAP
    
    denom = (car['m']*abs(kappa)+car['mu']*0.5*RHO_AIR*car['ClA'])
    if denom <= 0:
        # aero dominant corner
        return STRAIGHT_LINE_CAP

    v_max = math.sqrt((car['mu']*car['m']*G)/denom)
    return v_max
    

def get_max_corner_speeds(track):
    """a_lat = v^2 * kappa"""
    v_max = [0]*len(track)
    print(len(track), len(v_max))
    for i, segment in enumerate(track):
        v_max[i] = get_local_vmax(segment[1])

    return v_max


def main():
    track = get_track('.\\racelines\\Monza.csv')

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
        F_total_available = get_fz(track[i][1])*car['mu']
        F_longLeftover = math.sqrt(max(F_total_available**2 - F_lat**2, 0))

        accel_brake = -F_longLeftover/car['m'] - get_drag(v_seg)/car['m']
        v_prev = math.sqrt(v_seg**2 - 2*accel_brake*track[-i][0])
        if i-1 >= 0:
            v[i-1] = min(v[i-1], v_prev)

def forward_propagation(track, v):
    for i, v_seg in enumerate(v):
        F_lat = car['m']*(v_seg**2)*track[i][1]
        F_total_available = get_fz(track[i][1])*car['mu']
        F_longLeftover = math.sqrt(max(F_total_available**2 - F_lat**2, 0))
        
        accel_grip_limited = F_longLeftover/car['m']
        try:
            accel_power_limited = car['P_max']/(abs(v_seg)*car['m'])
        except ZeroDivisionError:
            accel_power_limited = accel_grip_limited
        accel = min(accel_grip_limited, accel_power_limited) - get_drag(v_seg)/car['m']
        v_next = math.sqrt(v_seg**2 + 2*accel*track[i][0])
        try:
            v[i+1] = min(v[i+1], v_next)
        except IndexError:
            pass

if __name__ == "__main__":
    main()
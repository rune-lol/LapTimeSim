import math

from carModel import CarModel, CarState
from tracktools import get_track, draw_track, calcTotalTime, getTotalDistance
from physical_constants import G, RHO_AIR

car = CarModel(# Based on values from LMP2 Car
    mu= 1.7,
    P_max= 375_000,
    m= 950,
    ClA= -3.75,
    CdA= 1.2 
)

def main():
    track = get_track('.\\LapTimeSim\\racelines\\Spa.csv')

    v_max = car.get_max_corner_speeds(track)

    # actual solver
    car_states = [CarState()] * len(v_max)
    for i, v in enumerate(v_max):
        car_states[i] = CarState(v=v)

    car_states[0] = CarState(v=0, forces=car_states[0].forces) #start speed (from a dig)
    epoch = 1
    while epoch <= 100:
        forward_propagation(track, car_states)
        backward_propagation(track, car_states)
        
        if epoch % 10 == 0:
            print('Ran',epoch,'epochs')
        epoch += 1

    print('t:',calcTotalTime(track, car_states),'s')
    print('s:',getTotalDistance(track),'m')
    print('v_avg:',(getTotalDistance(track)/calcTotalTime(track, car_states))*3.6,'km/h')
    draw_track(track, v_max)
    draw_track(track, car_states)

def backward_propagation(track, car_states):
    for j, state in enumerate(reversed(car_states)):
        v_seg = state.v
        i = len(car_states)-j-1
        F_lat = car.get_Flat(v_seg, track[i][1])
        F_total_available = car.get_fz(track[i][1])*car.mu
        F_longLeftover = math.sqrt(max(F_total_available**2 - F_lat**2, 0))

        accel_brake = -F_longLeftover/car.m - car.get_drag(v_seg)/car.m
        v_prev = math.sqrt(v_seg**2 - 2*accel_brake*track[-i][0])
        if i-1 >= 0:
            car_states[i-1] = CarState(v=min(car_states[i-1].v, v_prev), forces=(F_lat, F_longLeftover + accel_brake*G))

def forward_propagation(track, car_states):
    for i, state in enumerate(car_states):
        v_seg = state.v
        F_lat = car.get_Flat(v_seg, track[i][1])
        F_total_available = car.get_fz(track[i][1])*car.mu
        F_longLeftover = math.sqrt(max(F_total_available**2 - F_lat**2, 0))
        
        accel_grip_limited = F_longLeftover/car.m
        try:
            accel_power_limited = car.P_max/(abs(v_seg)*car.m)
        except ZeroDivisionError:
            accel_power_limited = accel_grip_limited
        accel = min(accel_grip_limited, accel_power_limited) - car.get_drag(v_seg)/car.m
        v_next = math.sqrt(v_seg**2 + 2*accel*track[i][0])
        try:
            car_states[i+1] = CarState(v=min(car_states[i+1].v, v_next), forces=(F_lat, F_longLeftover + accel*G))
        except IndexError:
            pass

if __name__ == "__main__":
    main()
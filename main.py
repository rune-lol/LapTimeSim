import math

from pyparsing import col

from carModel import CarModel, CarState
from tracktools import get_track, calcTotalTime, getTotalDistance
from visualisation_tools import draw_distancetrace, draw_track
from physical_constants import G, RHO_AIR
from powertrain_tools import get_coefficients

car = CarModel(# Based on values from LMP2 Car
    mu= 1.7,
    P_max= 375_000,
    m= 950,
    ClA= -5,
    CdA= 1.75,

    # Gibson GK428 V8
    # Educated guess;
    # P_max = 450_000 @ 8500
    # T_max = 556Nm @ 6000
    # Idle @ 3500

    # in all fairness this is all just an educated guess based on trends, google searches and AI halucinations. Should be close enough tho
    torque_coefficients=get_coefficients(
        [3000, 4500, 5500, 6500, 8500, 9000],
        [0.6, 0.9, 1, 0.98, 0.90, 0.85]
    ),
    T_max = 555, # 555 normally
    idlerpm=3000,
    redlinerpm=9000,

    # Gearbox
    # https://www.xtrac.com/product/p1159-transverse-lmp-gearbox/
    # Ratios are from iracing https://s100.iracing.com/wp-content/uploads/2023/10/UM-Dallara-P217-LMP2-Manual.pdf
    gear_ratios=[2.78, 2.15, 1.83, 1.59, 1.39, 1.24],
    final_drive=47/15, # 47/15 / 42/16 / 43/18
    trans_efficiency=0.95,
    driven_wheel_radius=0.3575, # lmp2 rear wheel
    shift_point=8900
)

print(car)

def main():
    track = get_track('.\\LapTimeSim\\racelines\\Spa.csv')

    v_max = car.get_max_corner_speeds(track)

    # actual solver
    car_states = [CarState()] * len(v_max)
    for i, v in enumerate(v_max):
        car_states[i] = CarState(v=v)

    car_states[0] = CarState(v=0, forces=car_states[0].forces) #start speed (from a dig)
    epoch = 1
    while epoch <= 30:
        forward_propagation(track, car_states)
        backward_propagation(track, car_states)
        
        if epoch % 10 == 0:
            print('Ran',epoch,'epochs')
        epoch += 1

    print('t:',calcTotalTime(track, car_states),'s')
    print('s:',getTotalDistance(track),'m')
    print('v_avg:',(getTotalDistance(track)/calcTotalTime(track, car_states))*3.6,'km/h')

    gear = []
    for state in car_states:
        gear.append(state.gear)

    rpm = []
    for state in car_states:
        rpm.append(state.rpm)

    speeds = []
    for state in car_states:
        speeds.append(state.v)

    draw_track(track, speeds)

    draw_distancetrace(track, [(gear, 'gear'), (rpm, 'rpm'), (speeds, 'speed')])

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
            car_states[i-1] = CarState(v=min(car_states[i-1].v, v_prev), forces=(F_lat, F_longLeftover + accel_brake*G), gear=car.get_optimal_gear(v_seg), rpm=car.get_optimal_rpm(v_seg))

def forward_propagation(track, car_states):
    for i, state in enumerate(car_states):
        v_seg = state.v
        F_lat = car.get_Flat(v_seg, track[i][1])
        F_total_available = car.get_fz(track[i][1])*car.mu
        F_longLeftover = math.sqrt(max(F_total_available**2 - F_lat**2, 0))
        
        accel_grip_limited = F_longLeftover/car.m
        try:
            accel_power_limited = car.get_max_usable_engine_wheel_force(v_seg)/car.m
        except ZeroDivisionError:
            accel_power_limited = accel_grip_limited
        accel = min(accel_grip_limited, accel_power_limited) - car.get_drag(v_seg)/car.m
        v_next = math.sqrt(v_seg**2 + 2*accel*track[i][0])
        try:
            car_states[i+1] = CarState(v=min(car_states[i+1].v, v_next), forces=(F_lat, F_longLeftover + accel*G), gear=car.get_optimal_gear(v_seg), rpm=car.get_optimal_rpm(v_seg))
        except IndexError:
            pass

if __name__ == "__main__":
    main()
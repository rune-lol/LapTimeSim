import math
import numpy as np

from pyparsing import col

from carModel import CarModel, CarState
from tracktools import get_track, calcTotalTime, getTotalDistance
from visualisation_tools import draw_distancetrace, draw_track, draw_curve, draw_GGV
from physical_constants import G
from powertrain_tools import get_coefficients

EPOCHS_TO_RUN = 30

car = CarModel(# Based on values from LMP2 Car
    mu0= 1.7,
    load_sensitivity_factor= -0.15,
    m= 950,
    ClA= -3.5,
    CdA= 1.65,

    # Gibson GK428 V8
    # Educated guess;
    # P_max = 450_000 @ 8500
    # T_max = 556Nm @ 6000
    # Idle @ 3500

    # in all fairness this is all just an educated guess based on trends, google searches and AI halucinations. Should be close enough tho
    torque_coefficients=get_coefficients(
        [3000, 4500, 5500, 6500, 8500, 9000],
        [0.6, 0.9, 1, 0.95, 0.90, 0.85]
    ),
    T_max = 555, # 555 normally
    idlerpm=3000,
    redlinerpm=9000,

    # Gearbox
    # https://www.xtrac.com/product/p1159-transverse-lmp-gearbox/
    # Ratios are from iracing https://s100.iracing.com/wp-content/uploads/2023/10/UM-Dallara-P217-LMP2-Manual.pdf
    gear_ratios=[3.31834921, 2.56634921, 2.18438095, 1.89790476, 1.6591746, 1.48012698],
    final_drive=2.9, # 47/15 / 42/16 / 43/18
    trans_efficiency=0.92,
    driven_wheel_radius=0.3575, # lmp2 rear wheel
    shift_point=8900
)

def main():
    # print the car
    print(car)

    # show torque and power curve
    draw_curve(np.linspace(0, 9500, 1000), np.asarray([car.get_torque(x) for x in np.linspace(0, 9500, 1000)]))
    draw_curve(np.linspace(0, 9500, 1000), np.asarray([car.get_power(x) for x in np.linspace(0, 9500, 1000)]))

    # import the track
    track = get_track('.\\racelines\\Spa.csv')

    # initialise a list with the maximum speeds everywhere
    v_max = car.get_max_corner_speeds(track)

    # initialise a list of car states, which the solver will modify.
    car_states = [CarState()] * len(v_max)
    for i, v in enumerate(v_max):
        car_states[i] = CarState(v=v)

    # start the solver
    car_states[0] = CarState(v=0) #start speed (from a dig)
    epoch = 1
    while epoch <= EPOCHS_TO_RUN:
        # propagate forward and backward
        forward_propagation(track, car_states)
        backward_propagation(track, car_states)
        
        # Dont spam the output lmao
        if epoch % 10 == 0:
            print('Ran',epoch,'epochs')
        epoch += 1

    # calculate accurate acceleration data based off of the kinematics equation
    prev_velo = 0
    for i, state in enumerate(car_states):
        velo = state.v
        a_long = (velo**2-prev_velo**2)/(2*track[i][0])
        a_lat = car.get_Flat(velo, track[i][1])/car.m
        car_states[i] = CarState(
            gear=state.gear,
            rpm=state.rpm,
            forces=state.forces,
            v=state.v,
            accel=(a_lat, a_long)
        )

        prev_velo = velo

    # print the important values
    print('t:',calcTotalTime(track, car_states),'s')
    print('s:',getTotalDistance(track),'m')
    print('v_avg:',(getTotalDistance(track)/calcTotalTime(track, car_states))*3.6,'km/h')

    # get arrays to make the nececary visualisations
    gear = []
    for state in car_states:
        gear.append(state.gear)

    rpm = []
    for state in car_states:
        rpm.append(state.rpm)

    speeds = []
    for state in car_states:
        speeds.append(state.v)

    accels = []
    for state in car_states:
        accels.append(math.sqrt(state.accel[0]**2 + state.accel[1]**2))
    
    # draw the track with the local max  speeds
    draw_track(track, v_max)
    # draw the track with the simulated stpeeds
    draw_track(track, speeds)

    # draw distancetrace with the previous visualisations
    draw_distancetrace(track, [(gear, 'gear'), (rpm, 'rpm'), (speeds, 'speed')])

    # draw the G-G-V diagram
    draw_GGV(car_states)

def backward_propagation(track, car_states: list[CarState]):
    for j, state in enumerate(reversed(car_states)):
        v_seg = state.v
        i = len(car_states)-j-1

        # compute traction limits
        F_lat = car.get_Flat(v_seg, track[i][1])
        F_total_available = car.get_fz(v_seg)*car.get_mu(car.get_fz(v_seg))
        F_longLeftover = 0
        # check for if the lateral speed limit is not exceeded
        if abs(F_lat) <= F_total_available:
            F_longLeftover = math.sqrt(max(F_total_available**2 - F_lat**2, 0))
            # set the current speed to the max steady state cornering speed
            limited_speed = car.get_local_vmax(track[i][1])

            car_states[i] = CarState(v=min(car_states[i].v, limited_speed), forces=car_states[i].forces, gear=car.get_optimal_gear(v_seg), rpm=car.get_optimal_rpm(v_seg), accel=car_states[i].accel)

        # calculate the forces
        F_net = -F_longLeftover - car.get_drag(v_seg)
        accel_brake = F_net / car.m

        # calculate and set the maximum allowed previous speed given our set accel_brake and the kinematics equation
        v_prev = math.sqrt(v_seg**2 - 2*accel_brake*track[-i][0])
        if i-1 >= 0:
            car_states[i-1] = CarState(v=min(car_states[i-1].v, v_prev), forces=(F_lat, F_longLeftover + accel_brake*G), gear=car.get_optimal_gear(v_seg), rpm=car.get_optimal_rpm(v_seg), accel=(0, 0))

def forward_propagation(track, car_states: list[CarState]):
    for i, state in enumerate(car_states):
        v_seg = state.v

        # compute traction limits
        F_lat = car.get_Flat(v_seg, track[i][1])
        F_total_available = car.get_fz(v_seg)*car.get_mu(car.get_fz(v_seg))
        F_longLeftover = 0
        # check for if the lateral speed limit is not exceeded
        if abs(F_lat) <= F_total_available:
            F_longLeftover = math.sqrt(max(F_total_available**2 - F_lat**2, 0))
            # set the current speed to the max steady state cornering speed
            limited_speed = car.get_local_vmax(track[i][1])

            car_states[i] = CarState(v=min(car_states[i].v, limited_speed), forces=car_states[i].forces, gear=car.get_optimal_gear(v_seg), rpm=car.get_optimal_rpm(v_seg), accel=car_states[i].accel)

        # calculate the forces
        force_grip_limited = F_longLeftover
        force_power_limited = car.get_max_usable_engine_wheel_force(v_seg)

        # decide what force goes from the wheels to the ground, either power or traction limited
        F_drive = min(
            force_grip_limited,
            force_power_limited
        )
        
        # get net forces
        F_net = F_drive - car.get_drag(v_seg)
        accel =  F_net/car.m

        # calculate and set the maximum allowed next speed given our set accel_brake and the kinematics equation
        v_next = math.sqrt(v_seg**2 + 2*accel*track[i][0])
        try:
            car_states[i+1] = CarState(v=min(car_states[i+1].v, v_next), forces=(F_lat, F_drive), gear=car.get_optimal_gear(v_seg), rpm=car.get_optimal_rpm(v_seg), accel=(0, 0))
        except IndexError:
            pass

if __name__ == "__main__":
    main()
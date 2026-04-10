import math
from dataclasses import dataclass
import numpy as np

from physical_constants import RHO_AIR, G
from powertrain_tools import get_coefficients

STRAIGHT_LINE_CAP = 1_000.0
AERO_DOMINANT_CAP = STRAIGHT_LINE_CAP/2

@dataclass
class CarModel:
    mu0: float # tyre coefficient of friction
    load_sensitivity_factor: float
    m: float # mass in kilograms
    ClA: float # Cl * A, m^2, negative = downforce
    CdA: float # Cd * A, m^2

    # engine specs
    idlerpm: int
    redlinerpm: int
    torque_coefficients: np.ndarray
    T_max: float

    # gearbox specs
    gear_ratios: list[float]
    final_drive: float
    trans_efficiency: float
    driven_wheel_radius: float
    shift_point: int

    def get_mu(self, fz):
        return self.mu0 * ((fz/(self.m*G))**self.load_sensitivity_factor)

    def get_optimal_gear(self, speed):
        # Assume shifting at the same speed for every gear
        gear = 1
        for ratio in self.gear_ratios:
            rpm = (speed / (self.driven_wheel_radius * 2 * math.pi))*60*ratio*self.final_drive
            if rpm < self.shift_point:
                break
            else:
                gear += 1

        gear = np.clip(gear, 1, len(self.gear_ratios))
        return gear

    def get_optimal_rpm(self, speed):
        return (speed / (self.driven_wheel_radius * 2 * math.pi))*60*self.gear_ratios[self.get_optimal_gear(speed)-1]*self.final_drive

    def get_max_usable_engine_wheel_force(self, speed):
        rpm = self.get_optimal_rpm(speed)
        engTorque = self.get_torque(rpm)
        wheelTorque = engTorque * self.final_drive * self.gear_ratios[self.get_optimal_gear(speed)-1]
        return wheelTorque / self.driven_wheel_radius

    def get_torque(self, rpm):
        if rpm < self.idlerpm:
            return 0.6*self.T_max
        if rpm >= self.redlinerpm:
            return 0

        poly = np.poly1d(self.torque_coefficients)

        rpm = np.asarray(rpm)  # ensures array operations work
        rpm_norm = rpm # normalize RPM to match polynomial fitting
        torque = poly(rpm_norm) * self.T_max  # scale to peak torque
        torque = np.clip(torque, 0, self.T_max)     # can't go below 0 or above peak
        return torque

    def get_power(self, rpm):
        torque = self.get_torque(rpm)
        omega = (rpm/60)*2*np.pi
        power = torque * omega
        return power

    def get_local_vmax(self, kappa):
        """Assumes lineair tyres, for initialising"""
        if abs(kappa) < 1e-6:
            return STRAIGHT_LINE_CAP

        denom = (self.m*abs(kappa)+self.mu0*0.5*RHO_AIR*self.ClA)
        if denom <= 0:
            # aero dominant corner
            return AERO_DOMINANT_CAP

        v_max = math.sqrt((self.mu0*self.m*G)/denom)

        for _ in range(3):
            Fz = self.get_fz(v_max)
            mu = self.get_mu(Fz)
            v_max = math.sqrt((mu * Fz) / (self.m * abs(kappa)))
        
        return v_max
    
    def get_max_corner_speeds(self, track):
        """
        a_lat = v^2 * kappa
        """
        v_max = [0]*len(track)
        for i, segment in enumerate(track):
            v_max[i] = self.get_local_vmax(segment[1])

        return v_max
    
    def get_fz(self, v):
        return self.m*G-0.5*RHO_AIR*(v**2)*self.ClA

    def get_drag(self, v):
        return 0.5*RHO_AIR*(v**2)*self.CdA

    def get_Flat(self, v, kappa):
        return self.m*(v**2)*kappa

@dataclass
class CarState:
    gear: int = 0
    rpm: int = 0
    forces: tuple[float, float] = (0, 0) # Lateral, Longtitudinal, in Newton
    accel: tuple[float, float] = (0, 0) # Lateral, Longtitudinal, in m/s²
    v: float = 0 # speed in m/s

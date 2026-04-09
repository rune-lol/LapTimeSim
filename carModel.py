import math
from dataclasses import dataclass

from physical_constants import RHO_AIR, G

STRAIGHT_LINE_CAP = 1_000.0

@dataclass
class CarModel:
    mu: float = 0 # tyre coefficient of friction
    P_max: float = 0 # maximum power in watts
    m: float = 0 # mass in kilograms
    ClA: float = 0 # Cl * A, m^2, negative = downforce
    CdA: float = 0 # Cd * A, m^2

    #def fromDict(self, inputDict: dict):
    #    """Legacy code"""
    #    self.mu = inputDict['mu']
    #    self.P_max = inputDict['P_max']
    #    self.m = inputDict['m']
    #    self.ClA = inputDict['ClA']
    #    self.CdA = inputDict['CdA']

    def get_local_vmax(self, kappa):
        if kappa < 1e-6:
            return STRAIGHT_LINE_CAP

        denom = (self.m*abs(kappa)+self.mu*0.5*RHO_AIR*self.ClA)
        if denom <= 0:
            # aero dominant corner
            return STRAIGHT_LINE_CAP

        v_max = math.sqrt((self.mu*self.m*G)/denom)
        return v_max
    
    
    def get_max_corner_speeds(self, track):
        """a_lat = v^2 * kappa"""
        v_max = [0]*len(track)
        print(len(track), len(v_max))
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
    forces: tuple[float, float] = (0, 0) # Lateral, Longtitudinal, in Newton
    v: float = 0 # speed in m/s

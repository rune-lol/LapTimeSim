# Quasi steady state point mass lap time sim
## Supports:
- Custom Tracks
- Tracks imported from the TUM dataset (attribution below)
- Car model:
    - mu
    - mass
    - max power
    - support for downforce/lift
    - support for drag

## How does it work?
### 1. Import Track:
Uses TUMFTM's dataset which is cloned from their github repository. We need to have the `ds` (arc lenght of the segment) and `κ`, the local curvature of the segment. `ds` is calculated easily enough using the formula: `ds = np.sqrt(dx**2 + dy**2)`. Local curvature `κ` is a little harder requiring this formula: `curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5`, noting that dt is not time but the paramter for the parametrised curve.
  
Track is a list of tuples, each tuple represents a track segment with the following properties: element 0 is the arc length of the track segment, and the 1 element is the curvature `κ`. (Example `(1m, 1/100 m^-1)` would be a segment of arclength 1, with radius 1m to the right)

### 2. Calculate Maximum Speeds
Create a list, `v_max`, containing the maximum steady state cornering speed at that track segment given that curvature `κ` and the aerodynamic characteristics of the car. Formula: `v_max^2 = a_lat / κ`, `sqrt((μ*m*G)/(m*abs(κ)+μ*0.5*ρ*Cl*A)`

### 3. Set Entry Conditions
Create a new list v containing the physical speeds, by copying the v_max list. set the `v[0] = 0`.

### 4. Run the solver
The solver is run for 100 epochs. It contains 2 parts: forward and backward propagation.

#### 4.1 Forward Propagation
Forward propagation checks that for each segment, it is physically possible to accelerate to the max cornering speed as initialised. Realistically speaking the cornering speed v_max is higher than the physical speed.
  
We check the lateral required cornering force: `F_lat = m*v_seg^2*κ`  
We check total available grip: `F_total_available = μ*(m*g-0.5*ρ*v^2*Cl*A)`  
Then we can calculate the longtitudinal force to reach the grip limit: `F_long = sqrt(F_total_available^2-F_lat^2)`  
  
After we have the total longtitudinal grip left we calculate how hard our car can accelerate. This can be grip or power limited:  
`a_grip_limited = F_longLeftover/m` (we assume that we have perfect braking)  
`a_power_limited = P_max / (v*m)`  
We then of course pick the lower limit of acceleration we can apply and calculate the highest possible speed in the next cell. Do not forget to subtract the deceleration from drag.  
`v[i] = sqrt(v^2+2*a*κ)`.
We then check if the next cell is higher than this maximum attainable speed, and if so we set it to that max speed.  
  
This has the effect that at each moment acceleration is optimised.

#### 4.2 Backward Propagation
We iterate through every segment of the track in reverse and then we calculate the maximum longtitudinal grip left in the same way as in forward propagation.
We again assume perfect braking and thus `a_braking = -F_longLeftover/m - 0.5*ρ*v^2*Cd*A`.
We can then determine the absolute maximum speed in the segment before the current segment to ensure the car has braked to the current cells speed. `v[i-1] = min(v[i-1], v_prev)`  
  
This has the effect that at any point the car optimally uses the grip limit regarding to braking.

#### 4.3 Combining them
When running the forward and backward propagation multiple times behind eachother we optimise the speed list until it converges to a point.

### 5. Showing
At last we show the data to the user using matplotlib
---
## Dataset:
Respect the licenses for the dataset (LGPL-3.0 license)  
https://github.com/TUMFTM/racetrack-database/tree/master
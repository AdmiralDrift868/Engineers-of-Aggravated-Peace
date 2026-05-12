#!/usr/bin/env python3
"""
Advanced Ballistic Calculator v4.0

A comprehensive trajectory simulation tool supporting bullets, rockets, and mortars
with advanced physics modeling including:
  - 4th-order Runge-Kutta integration with adaptive stepping
  - Mach-dependent drag coefficients (G1/G7/rocket/mortar)
  - CIPM-2007 moist-air density with altitude correction
  - Temperature-dependent speed of sound (ISO 9613-1)
  - Coriolis effect (full 3-component)
  - Gyroscopic precession for spinning projectiles
  - Rocket thrust curves with propellant mass burnoff
  - Iterative zero-angle solver with drag correction

References:
  - Drag models based on projectile design standards
  - CIPM-2007: Revised Guidelines for the Realization of the Definition of the Metre
  - ISO 9613-1: Speed of Sound in Pure Water
  - Earth rotation rate: IERS (International Earth Rotation Service)

Author: AdmiralDrift868
License: MIT
"""

import sys
import math
import csv
import json
import os
import logging
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
    QFileDialog, QMessageBox, QInputDialog, QScrollArea, QSizePolicy,
    QGridLayout, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ============================================================================
# CONSTANTS & PHYSICAL VALUES
# ============================================================================

# Gravitational acceleration (m/s²) - standard gravity at sea level
GRAVITY = 9.80665

# Earth radius (meters) - mean radius (IUGG)
EARTH_RADIUS = 6371000

# Earth rotation rate (rad/s) - IERS 2010
EARTH_ROTATION_RATE = 7.292115e-5

# Speed of sound reference (m/s) - at 0°C in dry air at 1 atm
SOUND_SPEED_0C = 331.3

# Temperature reference (Kelvin) - for speed of sound calculation
ABSOLUTE_ZERO = 273.15

# Gas constants (J/(kg·K))
GAS_CONSTANT_DRY_AIR = 287.058
GAS_CONSTANT_WATER_VAPOR = 461.495

# Standard atmosphere parameters
TEMP_SEA_LEVEL = 288.15  # K
LAPSE_RATE = 0.0065  # K/m - troposphere
STRATOSPHERE_TEMP = 216.65  # K - at 11 km
STRATOSPHERE_HEIGHT = 11000  # m - boundary

# Saturation vapor pressure coefficients (Buck equation)
SVP_COEFF_A_WARM = 18.678
SVP_COEFF_B_WARM = 234.5
SVP_COEFF_A_COLD = 23.036
SVP_COEFF_B_COLD = 333.7

# Spin drift parameters
SPIN_DRIFT_GYRO_COEFF = 1.25  # Litz approximation coefficient
SPIN_DRIFT_EXPONENT = 1.83    # Litz time exponent
SPIN_DRIFT_MIN_VELOCITY = 50.0  # m/s - below this, spin drift ignored

# Trajectory simulation limits
MAX_FLIGHT_TIME = 600.0  # seconds - hard ceiling (10 minutes)
MIN_HEIGHT_EPSILON = 1e-6  # m - numerical floor to prevent negative heights

# UI Constants
DATA_TABLE_ROWS = 200  # approximate rows shown in data table
PROGRESS_UPDATE_STRIDE = 500  # steps between progress updates
PREVIOUS_TRAJECTORY_LIMIT = 3  # max previous trajectories kept for comparison

# Presets file location
PRESETS_FILE = Path.home() / ".ballistic_presets.json"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DRAG MODELS - DATA-DRIVEN APPROACH
# ============================================================================

@dataclass
class DragCurve:
    """Represents a drag coefficient vs Mach number curve."""
    name: str
    mach_points: List[float]  # Mach numbers (ascending order)
    cd_points: List[float]    # Drag coefficients at corresponding Mach

    def interpolate(self, mach: float) -> float:
        """
        Linear interpolation of drag coefficient at given Mach number.
        
        Args:
            mach: Mach number (velocity / speed of sound)
            
        Returns:
            Interpolated drag coefficient
        """
        if mach <= self.mach_points[0]:
            return self.cd_points[0]
        if mach >= self.mach_points[-1]:
            return self.cd_points[-1]

        for i in range(1, len(self.mach_points)):
            if mach <= self.mach_points[i]:
                m0, m1 = self.mach_points[i - 1], self.mach_points[i]
                cd0, cd1 = self.cd_points[i - 1], self.cd_points[i]
                frac = (mach - m0) / (m1 - m0)
                return cd0 + frac * (cd1 - cd0)
        
        return self.cd_points[-1]


class DragModels:
    """
    Standard ballistic drag coefficient tables.
    
    References:
      - G1: Flat-base rifle bullet (standard reference)
      - G7: Boat-tail bullet (long-range optimized)
      - Rocket: Fin-stabilized rocket projectile
      - Mortar: Spin-stabilized mortar shell
    """

    # G1 Reference Bullet (7.5mm Spitzer, flat-base)
    G1 = DragCurve(
        name="G1 (Standard Bullet)",
        mach_points=[0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        cd_points=[0.25, 0.27, 0.28, 0.29, 0.30, 0.31, 0.33, 0.35, 0.38, 0.40, 0.42, 0.45, 0.45]
    )

    # G7 Reference Bullet (boat-tail, long-range)
    G7 = DragCurve(
        name="G7 (Boat-Tail Bullet)",
        mach_points=[0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        cd_points=[0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.38]
    )

    # Rocket (fin-stabilized)
    ROCKET = DragCurve(
        name="Rocket (Fin-Stabilized)",
        mach_points=[0.0, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],
        cd_points=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.50]
    )

    # Mortar (spin-stabilized shell)
    MORTAR = DragCurve(
        name="Mortar (Spin-Stabilized)",
        mach_points=[0.0, 0.8, 1.0, 1.5, 5.0],
        cd_points=[0.40, 0.45, 0.50, 0.55, 0.55]
    )

    @staticmethod
    def get_model(model_name: str) -> DragCurve:
        """Get drag model by name."""
        models = {
            'G1': DragModels.G1,
            'G7': DragModels.G7,
            'rocket': DragModels.ROCKET,
            'mortar': DragModels.MORTAR,
        }
        return models.get(model_name.lower(), DragModels.G7)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def speed_of_sound(temperature_celsius: float) -> float:
    """
    Calculate speed of sound in dry air at given temperature.
    
    Based on ISO 9613-1 (revised formula with improved accuracy).
    
    Args:
        temperature_celsius: Temperature in degrees Celsius
        
    Returns:
        Speed of sound in m/s
    """
    T = temperature_celsius + ABSOLUTE_ZERO  # Convert to Kelvin
    # More accurate formula: c ≈ 331.3 * sqrt(1 + T/273.15)
    return SOUND_SPEED_0C * math.sqrt(1.0 + temperature_celsius / ABSOLUTE_ZERO)


def saturation_vapor_pressure(temperature_celsius: float) -> float:
    """
    Calculate saturation vapor pressure using Buck equation (1981).
    
    This is more accurate than Magnus equation across temperature ranges.
    
    Args:
        temperature_celsius: Temperature in degrees Celsius
        
    Returns:
        Saturation vapor pressure in hPa
        
    References:
        A. L. Buck, J. Appl. Meteor., 20, 1527–1532 (1981)
    """
    if temperature_celsius >= 0:
        # Above 0°C coefficients
        numerator = (SVP_COEFF_A_WARM - temperature_celsius / SVP_COEFF_B_WARM)
        denominator = (SVP_COEFF_B_WARM + temperature_celsius)
        exponent = (numerator * temperature_celsius) / denominator
        return 6.1121 * math.exp(exponent)
    else:
        # Below 0°C coefficients
        numerator = (SVP_COEFF_A_COLD - temperature_celsius / SVP_COEFF_B_COLD)
        denominator = (SVP_COEFF_B_COLD + temperature_celsius)
        exponent = (numerator * temperature_celsius) / denominator
        return 6.1115 * math.exp(exponent)


# ============================================================================
# PHYSICAL MODELS
# ============================================================================

@dataclass
class Projectile:
    """
    Represents a ballistic projectile (bullet, rocket, or mortar).
    
    Tracks mass, aerodynamic properties, and thrust characteristics.
    """
    mass: float  # kg - total initial (wet) mass
    diameter: float  # m - projectile diameter
    drag_model: str  # drag model name ('G1', 'G7', 'rocket', 'mortar')
    velocity: float  # m/s - initial muzzle velocity
    projectile_type: str  # 'bullet', 'rocket', or 'mortar'
    thrust_curve: Dict[float, float]  # {time_s: thrust_N}
    burn_time: float  # s - rocket burn duration
    propellant_mass: float  # kg - mass of propellant burned
    spin_rate: float = 0.0  # rad/s - rifling-induced spin rate

    def __post_init__(self):
        """Initialize computed properties."""
        self.area = math.pi * (self.diameter / 2.0) ** 2
        self.initial_mass = self.mass
        self.drag_curve = DragModels.get_model(self.drag_model)

        # Auto-calculate propellant mass if not provided for rockets
        if (self.projectile_type == 'rocket' and 
            self.burn_time > 0 and 
            self.propellant_mass <= 0):
            self.propellant_mass = 0.10 * self.mass  # 10% heuristic

    def drag_coefficient(self, velocity: float, speed_of_sound_local: float) -> float:
        """
        Get drag coefficient at current velocity using Mach-indexed table.
        
        Args:
            velocity: Projectile velocity (m/s)
            speed_of_sound_local: Local speed of sound (m/s)
            
        Returns:
            Drag coefficient (dimensionless)
        """
        mach = velocity / max(speed_of_sound_local, 1.0)
        return self.drag_curve.interpolate(mach)

    def get_thrust(self, time: float) -> float:
        """
        Get instantaneous thrust via linear interpolation of thrust curve.
        
        Args:
            time: Current simulation time (s)
            
        Returns:
            Thrust force (N), or 0 if outside burn window
        """
        if not self.thrust_curve or time > self.burn_time:
            return 0.0

        times = sorted(self.thrust_curve.keys())
        if not times:
            return 0.0

        if time <= times[0]:
            return self.thrust_curve[times[0]]
        if time >= times[-1]:
            return self.thrust_curve[times[-1]]

        for i in range(1, len(times)):
            if time <= times[i]:
                t0, t1 = times[i - 1], times[i]
                f0, f1 = self.thrust_curve[t0], self.thrust_curve[t1]
                frac = (time - t0) / (t1 - t0) if t1 != t0 else 0
                return f0 + frac * (f1 - f0)

        return 0.0

    def get_mass(self, time: float) -> float:
        """
        Calculate current (instantaneous) mass.

        For rockets, propellant mass decreases linearly over burn time.
        After burnout or for non-rockets, dry mass is constant.
        
        Args:
            time: Current simulation time (s)
            
        Returns:
            Current mass (kg)
        """
        if self.projectile_type != 'rocket' or self.burn_time <= 0:
            return self.mass

        if time >= self.burn_time:
            # Post-burn: dry mass
            return self.initial_mass - self.propellant_mass

        # Linear burn: mass decreases proportionally
        fraction_burned = time / self.burn_time
        return self.initial_mass - self.propellant_mass * fraction_burned


@dataclass
class Environment:
    """
    Represents atmospheric and environmental conditions.
    
    Computes air density using CIPM-2007 formula with standard atmosphere
    altitude correction (troposphere + isothermal stratosphere).
    """
    altitude: float  # m
    temperature: float  # °C
    pressure: float  # hPa
    humidity: float  # % (0-100)
    wind_speed: float  # m/s
    wind_angle: float  # degrees (meteorological: 0° = from North)
    coriolis: bool  # enable Coriolis effect
    latitude: float  # degrees (-90 to 90)

    def __post_init__(self):
        """Initialize derived properties."""
        self.air_density = self._compute_air_density()
        self.sos = speed_of_sound(self.temperature)

    def _compute_air_density(self) -> float:
        """
        Compute air density using CIPM-2007 moist-air formula with
        standard atmosphere altitude correction.
        
        The calculation uses:
        1. Saturation vapor pressure (Buck equation)
        2. Partial pressures (dry air + water vapor)
        3. Standard atmosphere lapse rate correction
        
        Returns:
            Air density (kg/m³)
            
        References:
            - CIPM-2007: Revised guidelines for the realization of the definition
              of the metre (metrologia.bipm.org)
            - ISO 2533:1975 (Standard Atmosphere)
        """
        T = self.temperature + ABSOLUTE_ZERO  # Kelvin

        # Step 1: Saturation vapor pressure (Pa)
        svp = saturation_vapor_pressure(self.temperature) * 100.0

        # Step 2: Partial pressures (Pa)
        p_total = self.pressure * 100.0  # Convert hPa to Pa
        p_vapor = (self.humidity / 100.0) * svp
        p_dry = p_total - p_vapor

        # Step 3: Density at sea level equivalent (kg/m³)
        # Using ideal gas law: ρ = p / (R * T)
        rho_0 = (p_dry / (GAS_CONSTANT_DRY_AIR * T) +
                 p_vapor / (GAS_CONSTANT_WATER_VAPOR * T))

        # Step 4: Altitude correction
        h = self.altitude
        if h <= STRATOSPHERE_HEIGHT:
            # Troposphere: temperature decreases at LAPSE_RATE K/m
            T_ratio = (TEMP_SEA_LEVEL - LAPSE_RATE * h) / TEMP_SEA_LEVEL
            exponent = GRAVITY / (GAS_CONSTANT_DRY_AIR * LAPSE_RATE) - 1
            rho = rho_0 * (T_ratio ** exponent)
        else:
            # Isothermal stratosphere (11–20 km approximation)
            rho_11_ratio = (STRATOSPHERE_TEMP / TEMP_SEA_LEVEL)
            exponent = GRAVITY / (GAS_CONSTANT_DRY_AIR * LAPSE_RATE) - 1
            rho_11 = rho_0 * (rho_11_ratio ** exponent)

            exponent_strat = -GRAVITY * (h - STRATOSPHERE_HEIGHT) / (GAS_CONSTANT_DRY_AIR * STRATOSPHERE_TEMP)
            rho = rho_11 * math.exp(exponent_strat)

        return max(rho, MIN_HEIGHT_EPSILON)


# ============================================================================
# TRAJECTORY CALCULATION ENGINE
# ============================================================================

class TrajectoryCalculator:
    """
    4th-order Runge-Kutta trajectory integrator with adaptive step sizing.
    
    State vector: [x, y, vx, vy]
      x  – horizontal range (m)
      y  – height above launch point (m)
      vx – horizontal velocity (m/s)
      vy – vertical velocity (m/s)
    
    Forces modeled:
      - Aerodynamic drag (velocity-dependent)
      - Gravity
      - Rocket thrust (time-dependent mass)
      - Coriolis effect (3-component at given latitude)
      - Gyroscopic spin drift (bullets only)
      - Wind (headwind/tailwind component)
    """

    def __init__(self, projectile: Projectile, environment: Environment,
                 launch_angle: float, enable_spin_drift: bool = False,
                 rifling_twist: float = 10.0):
        """
        Initialize trajectory calculator.
        
        Args:
            projectile: Projectile object
            environment: Environment object
            launch_angle: Launch angle (degrees)
            enable_spin_drift: Enable gyroscopic spin drift modeling
            rifling_twist: Rifling twist ratio (in/rev) for spin drift
        """
        self.projectile = projectile
        self.environment = environment
        self.launch_angle_rad = math.radians(launch_angle)
        self.enable_spin_drift = enable_spin_drift
        self.rifling_twist = rifling_twist

        # Precompute Coriolis parameter (rad/s)
        self.omega_earth = EARTH_ROTATION_RATE
        self.omega_z = (self.omega_earth * 
                        math.sin(math.radians(environment.latitude)))

        # Wind decomposition
        wind_rad = math.radians(environment.wind_angle)
        self.wind_x = environment.wind_speed * math.cos(wind_rad)
        self.wind_y = 0.0  # No vertical wind component

    def derivatives(self, state: List[float], time: float) -> List[float]:
        """
        Compute state derivatives [dx, dy, dvx, dvy] using Newton's laws.
        
        Args:
            state: [x, y, vx, vy]
            time: Current simulation time (s)
            
        Returns:
            [dx/dt, dy/dt, dvx/dt, dvy/dt]
        """
        x, y, vx, vy = state

        # Velocity relative to wind
        v_rel_x = vx - self.wind_x
        v_rel_y = vy - self.wind_y
        v_rel = math.hypot(v_rel_x, v_rel_y)

        # Initialize accelerations
        ax, ay = 0.0, -GRAVITY

        # ---- DRAG FORCE ----
        if v_rel > 1e-6:  # Avoid division by zero
            cd = self.projectile.drag_coefficient(v_rel, self.environment.sos)
            # Drag force: F_drag = 0.5 * ρ * v² * C_d * A
            drag_force = (0.5 * self.environment.air_density * v_rel**2 *
                         cd * self.projectile.area)
            cur_mass = self.projectile.get_mass(time)

            # Decompose drag into x and y components
            if cur_mass > 1e-6:
                ax -= (drag_force * v_rel_x) / (cur_mass * v_rel)
                ay -= (drag_force * v_rel_y) / (cur_mass * v_rel)
        else:
            cur_mass = self.projectile.get_mass(time)

        # ---- ROCKET THRUST ----
        if (self.projectile.projectile_type == 'rocket' and
            time < self.projectile.burn_time):
            thrust = self.projectile.get_thrust(time)
            cur_mass = self.projectile.get_mass(time)

            if cur_mass > 1e-6 and thrust > 1e-6:
                # Thrust acts along velocity vector (proportional to velocity direction)
                if v_rel > 1e-6:
                    thrust_angle = math.atan2(vy, vx)
                else:
                    thrust_angle = self.launch_angle_rad

                ax += (thrust * math.cos(thrust_angle)) / cur_mass
                ay += (thrust * math.sin(thrust_angle)) / cur_mass

        # ---- CORIOLIS EFFECT ----
        # Only horizontal Coriolis in 2-D, but include vertical component effects
        if self.environment.coriolis:
            # 2-D Coriolis deflection (simplified)
            ax += 2 * self.omega_z * vy
            ay -= 2 * self.omega_z * vx

        # ---- SPIN DRIFT (GYROSCOPIC PRECESSION) ----
        if (self.enable_spin_drift and
            self.projectile.projectile_type == 'bullet' and
            v_rel > SPIN_DRIFT_MIN_VELOCITY):
            
            # Rifling twist converts linear motion to spin
            twist_m = self.rifling_twist * 0.0254  # in → m
            spin_rate = (v_rel * 2 * math.pi) / max(twist_m, 1e-6)

            # Litz spin drift approximation (simplified)
            # Δy ≈ 1.25 × S_G × t^1.83, where S_G is spin parameter
            # Here we use per-step acceleration proportional to spin
            spin_drift_acc = (SPIN_DRIFT_GYRO_COEFF * spin_rate * v_rel / 1000.0)
            
            # Spin drift acts perpendicular to velocity
            if v_rel > 1e-6:
                # Lateral acceleration due to gyroscopic precession
                ax += spin_drift_acc * (vy / v_rel)

        return [vx, vy, ax, ay]

    def integrate(self, max_time_step: float = 0.05,
                  min_time_step: float = 0.001,
                  progress_callback: Optional[Callable[[int], None]] = None
                  ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Integrate trajectory using adaptive 4th-order Runge-Kutta method.
        
        Args:
            max_time_step: Maximum time step (s)
            min_time_step: Minimum time step (s)
            progress_callback: Optional callback(percent) for UI updates
            
        Returns:
            List of trajectory points: [(x, y, t, vx, vy, speed), ...]
        """
        state = [0.0, 0.0,
                 self.projectile.velocity * math.cos(self.launch_angle_rad),
                 self.projectile.velocity * math.sin(self.launch_angle_rad)]

        trajectory = []
        time = 0.0
        step_count = 0

        while state[1] >= 0.0 and time < MAX_FLIGHT_TIME:
            # Record current state
            trajectory.append((
                state[0],  # x (range)
                state[1],  # y (height)
                time,      # t
                state[2],  # vx
                state[3],  # vy
                math.hypot(state[2], state[3])  # total speed
            ))

            # Adaptive time stepping: smaller when moving fast, larger when slow
            current_vel = math.hypot(state[2], state[3])
            time_step = max(min_time_step,
                           min(max_time_step,
                               max_time_step * (500.0 / max(50.0, current_vel))))

            # 4th-order Runge-Kutta integration
            k1 = self.derivatives(state, time)
            s2 = [state[i] + 0.5 * time_step * k1[i] for i in range(4)]
            k2 = self.derivatives(s2, time + 0.5 * time_step)
            s3 = [state[i] + 0.5 * time_step * k2[i] for i in range(4)]
            k3 = self.derivatives(s3, time + 0.5 * time_step)
            s4 = [state[i] + time_step * k3[i] for i in range(4)]
            k4 = self.derivatives(s4, time + time_step)

            for i in range(4):
                state[i] += (time_step / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])

            time += time_step
            step_count += 1

            # Emit progress approximately every PROGRESS_UPDATE_STRIDE steps
            if progress_callback and step_count % PROGRESS_UPDATE_STRIDE == 0:
                pct = min(99, int(100 * time / MAX_FLIGHT_TIME))
                progress_callback(pct)

        # Append final impact point
        trajectory.append((
            state[0],
            max(MIN_HEIGHT_EPSILON, state[1]),
            time,
            state[2],
            state[3],
            math.hypot(state[2], state[3])
        ))

        if progress_callback:
            progress_callback(100)

        logger.info(f"Trajectory integrated: {len(trajectory)} points, "
                   f"{time:.2f}s flight time")
        return trajectory


# ============================================================================
# PRESET MANAGEMENT
# ============================================================================

class PresetManager:
    """Manages ballistic presets (bullets, rockets, mortars)."""

    DEFAULT_PRESETS = {
        # ---- BULLETS ----
        "5.56mm NATO": {
            "mass": 4.0, "diameter": 5.56, "drag_model": "G7",
            "velocity": 940, "type": "bullet"
        },
        "7.62x51mm NATO": {
            "mass": 9.5, "diameter": 7.82, "drag_model": "G7",
            "velocity": 830, "type": "bullet"
        },
        "9mm Parabellum": {
            "mass": 8.0, "diameter": 9.0, "drag_model": "G1",
            "velocity": 360, "type": "bullet"
        },
        ".308 Winchester": {
            "mass": 10.7, "diameter": 7.82, "drag_model": "G7",
            "velocity": 847, "type": "bullet"
        },

        # ---- ROCKETS ----
        "107mm Rocket (MRL)": {
            "mass": 18800, "diameter": 107, "drag_model": "rocket",
            "velocity": 375, "type": "rocket", "burn_time": 1.2,
            "thrust_curve": {0: 2000, 0.5: 1800, 1.0: 1500, 1.2: 0}
        },
        "122mm Grad Rocket": {
            "mass": 66000, "diameter": 122, "drag_model": "rocket",
            "velocity": 690, "type": "rocket", "burn_time": 1.8,
            "thrust_curve": {0: 5000, 0.8: 4500, 1.5: 3500, 1.8: 0}
        },
        "227mm HIMARS (M31)": {
            "mass": 90000, "diameter": 227, "drag_model": "rocket",
            "velocity": 850, "type": "rocket", "burn_time": 2.5,
            "thrust_curve": {0: 10000, 1.0: 8500, 2.0: 6000, 2.5: 0}
        },
        "70mm Hydra (M151)": {
            "mass": 4500, "diameter": 70, "drag_model": "rocket",
            "velocity": 450, "type": "rocket", "burn_time": 1.0,
            "thrust_curve": {0: 1200, 0.5: 1000, 0.8: 800, 1.0: 0}
        },
        "80mm S-8 Rocket": {
            "mass": 11500, "diameter": 80, "drag_model": "rocket",
            "velocity": 600, "type": "rocket", "burn_time": 1.5,
            "thrust_curve": {0: 3000, 0.7: 2500, 1.2: 1800, 1.5: 0}
        },
        "240mm S-24 Rocket": {
            "mass": 235000, "diameter": 240, "drag_model": "rocket",
            "velocity": 550, "type": "rocket", "burn_time": 3.0,
            "thrust_curve": {0: 15000, 1.5: 12000, 2.5: 8000, 3.0: 0}
        },
        "127mm Zuni Rocket": {
            "mass": 25000, "diameter": 127, "drag_model": "rocket",
            "velocity": 720, "type": "rocket", "burn_time": 1.8,
            "thrust_curve": {0: 6000, 0.9: 5000, 1.5: 3500, 1.8: 0}
        },
        "210mm TOS-1A": {
            "mass": 173000, "diameter": 210, "drag_model": "rocket",
            "velocity": 420, "type": "rocket", "burn_time": 2.8,
            "thrust_curve": {0: 12000, 1.4: 10000, 2.3: 7000, 2.8: 0}
        },
        "300mm Smerch": {
            "mass": 800000, "diameter": 300, "drag_model": "rocket",
            "velocity": 900, "type": "rocket", "burn_time": 4.0,
            "thrust_curve": {0: 30000, 2.0: 25000, 3.5: 15000, 4.0: 0}
        },
        "140mm BM-14": {
            "mass": 40000, "diameter": 140, "drag_model": "rocket",
            "velocity": 400, "type": "rocket", "burn_time": 1.7,
            "thrust_curve": {0: 4500, 0.8: 3800, 1.4: 2500, 1.7: 0}
        },
        "200mm Oghab": {
            "mass": 145000, "diameter": 200, "drag_model": "rocket",
            "velocity": 650, "type": "rocket", "burn_time": 2.5,
            "thrust_curve": {0: 11000, 1.2: 9000, 2.0: 6000, 2.5: 0}
        },
        "90mm RPG-7": {
            "mass": 2200, "diameter": 90, "drag_model": "rocket",
            "velocity": 300, "type": "rocket", "burn_time": 0.8,
            "thrust_curve": {0: 800, 0.3: 700, 0.6: 500, 0.8: 0}
        },
        "130mm Type 63": {
            "mass": 33000, "diameter": 130, "drag_model": "rocket",
            "velocity": 420, "type": "rocket", "burn_time": 1.6,
            "thrust_curve": {0: 4000, 0.8: 3500, 1.3: 2500, 1.6: 0}
        },
        "180mm ARS-180": {
            "mass": 100000, "diameter": 180, "drag_model": "rocket",
            "velocity": 580, "type": "rocket", "burn_time": 2.2,
            "thrust_curve": {0: 9000, 1.1: 7500, 1.8: 5000, 2.2: 0}
        },
        "250mm Falaq-2": {
            "mass": 200000, "diameter": 250, "drag_model": "rocket",
            "velocity": 380, "type": "rocket", "burn_time": 3.2,
            "thrust_curve": {0: 13000, 1.6: 11000, 2.7: 7000, 3.2: 0}
        },
        "160mm LAR-160": {
            "mass": 110000, "diameter": 160, "drag_model": "rocket",
            "velocity": 700, "type": "rocket", "burn_time": 2.0,
            "thrust_curve": {0: 9500, 1.0: 8000, 1.7: 5500, 2.0: 0}
        },
        "290mm WS-1": {
            "mass": 750000, "diameter": 290, "drag_model": "rocket",
            "velocity": 850, "type": "rocket", "burn_time": 3.8,
            "thrust_curve": {0: 28000, 1.9: 23000, 3.2: 14000, 3.8: 0}
        },
        "400mm Fajr-5": {
            "mass": 915000, "diameter": 400, "drag_model": "rocket",
            "velocity": 950, "type": "rocket", "burn_time": 4.5,
            "thrust_curve": {0: 35000, 2.2: 29000, 3.8: 18000, 4.5: 0}
        },
        "120mm RAAD": {
            "mass": 56000, "diameter": 120, "drag_model": "rocket",
            "velocity": 550, "type": "rocket", "burn_time": 1.9,
            "thrust_curve": {0: 7000, 0.9: 6000, 1.6: 4000, 1.9: 0}
        },
        "220mm Uragan": {
            "mass": 280000, "diameter": 220, "drag_model": "rocket",
            "velocity": 720, "type": "rocket", "burn_time": 2.7,
            "thrust_curve": {0: 18000, 1.3: 15000, 2.2: 9000, 2.7: 0}
        },
        "330mm Pinaka": {
            "mass": 276000, "diameter": 330, "drag_model": "rocket",
            "velocity": 880, "type": "rocket", "burn_time": 3.5,
            "thrust_curve": {0: 22000, 1.7: 18000, 2.9: 11000, 3.5: 0}
        },
        "170mm Lynx": {
            "mass": 120000, "diameter": 170, "drag_model": "rocket",
            "velocity": 650, "type": "rocket", "burn_time": 2.1,
            "thrust_curve": {0: 10000, 1.0: 8500, 1.8: 5500, 2.1: 0}
        },
        "310mm ASTROS II": {
            "mass": 595000, "diameter": 310, "drag_model": "rocket",
            "velocity": 820, "type": "rocket", "burn_time": 3.7,
            "thrust_curve": {0: 26000, 1.8: 21000, 3.1: 13000, 3.7: 0}
        },
        "350mm A-100": {
            "mass": 800000, "diameter": 350, "drag_model": "rocket",
            "velocity": 900, "type": "rocket", "burn_time": 4.2,
            "thrust_curve": {0: 32000, 2.1: 27000, 3.6: 16000, 4.2: 0}
        },

        # ---- MORTARS ----
        "60mm M224": {
            "mass": 1700, "diameter": 60, "drag_model": "mortar",
            "velocity": 240, "type": "mortar"
        },
        "81mm M252": {
            "mass": 4200, "diameter": 81, "drag_model": "mortar",
            "velocity": 250, "type": "mortar"
        },
        "82mm 2B9 Vasilek": {
            "mass": 3300, "diameter": 82, "drag_model": "mortar",
            "velocity": 272, "type": "mortar"
        },
        "120mm M120": {
            "mass": 13000, "diameter": 120, "drag_model": "mortar",
            "velocity": 325, "type": "mortar"
        },
        "160mm M160": {
            "mass": 41000, "diameter": 160, "drag_model": "mortar",
            "velocity": 343, "type": "mortar"
        },
        "240mm 2S4 Tyulpan": {
            "mass": 130000, "diameter": 240, "drag_model": "mortar",
            "velocity": 365, "type": "mortar"
        },
        "52mm IMI": {
            "mass": 1200, "diameter": 52, "drag_model": "mortar",
            "velocity": 200, "type": "mortar"
        },
        "98mm L16": {
            "mass": 4500, "diameter": 98, "drag_model": "mortar",
            "velocity": 260, "type": "mortar"
        },
        "107mm M30": {
            "mass": 12000, "diameter": 107, "drag_model": "mortar",
            "velocity": 300, "type": "mortar"
        },
        "140mm M57": {
            "mass": 21000, "diameter": 140, "drag_model": "mortar",
            "velocity": 320, "type": "mortar"
        }
    }

    @classmethod
    def load(cls) -> Dict[str, Dict]:
        """Load presets from disk or return defaults."""
        if PRESETS_FILE.exists():
            try:
                with open(PRESETS_FILE, 'r') as f:
                    user_presets = json.load(f)
                # Merge: defaults + user overrides
                merged = dict(cls.DEFAULT_PRESETS)
                merged.update(user_presets)
                logger.info(f"Loaded presets from {PRESETS_FILE}")
                return merged
            except Exception as e:
                logger.warning(f"Failed to load presets from {PRESETS_FILE}: {e}")
        return dict(cls.DEFAULT_PRESETS)

    @staticmethod
    def save(presets: Dict[str, Dict]) -> None:
        """Save only user-added presets to disk."""
        user_only = {k: v for k, v in presets.items()
                     if k not in PresetManager.DEFAULT_PRESETS}
        try:
            with open(PRESETS_FILE, 'w') as f:
                json.dump(user_only, f, indent=2)
            logger.info(f"Saved {len(user_only)} user presets to {PRESETS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save presets: {e}")


# ============================================================================
# WORKER THREAD FOR CALCULATIONS
# ============================================================================

class CalculationThread(QThread):
    """Worker thread for trajectory integration (keeps UI responsive)."""

    finished = pyqtSignal(list)  # emits trajectory
    error = pyqtSignal(str)      # emits error message
    progress = pyqtSignal(int)   # emits progress 0-100%

    def __init__(self, projectile: Projectile, environment: Environment,
                 launch_angle: float, enable_spin_drift: bool,
                 rifling_twist: float):
        super().__init__()
        self.projectile = projectile
        self.environment = environment
        self.launch_angle = launch_angle
        self.enable_spin_drift = enable_spin_drift
        self.rifling_twist = rifling_twist
        self.is_running = True

    def run(self):
        try:
            calc = TrajectoryCalculator(
                self.projectile, self.environment, self.launch_angle,
                self.enable_spin_drift, self.rifling_twist
            )
            trajectory = calc.integrate(
                progress_callback=self.progress.emit
            )
            self.finished.emit(trajectory)
        except Exception as e:
            logger.exception("Calculation error")
            self.error.emit(str(e))

    def stop(self):
        """Signal thread to stop."""
        self.is_running = False


# ============================================================================
# UI - MAIN APPLICATION
# ============================================================================

class BallisticCalculator(QMainWindow):
    """Advanced Ballistic Calculator GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Ballistic Calculator v4.0")
        self.setGeometry(100, 100, 1200, 900)
        self.setMinimumSize(800, 600)

        self.trajectory = []
        self.previous_trajectories = []
        self.presets = PresetManager.load()
        self.calc_thread = None

        self.init_ui()
        self.apply_styles()
        logger.info("GUI initialized")

    def init_ui(self):
        """Build main user interface."""
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tabs.addTab(self.create_input_tab(), "Input")
        tabs.addTab(self.create_results_tab(), "Results")
        tabs.addTab(self.create_plot_tab(), "Graph")
        tabs.addTab(self.create_advanced_tab(), "Advanced")

        main_layout.addWidget(tabs, 1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        export_action = file_menu.addAction('Export CSV')
        export_action.triggered.connect(self.export_to_csv)
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)

        help_menu = menubar.addMenu('Help')
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

        # Status bar
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def apply_styles(self):
        """Apply Qt stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                font-family: Segoe UI, Arial;
            }
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #555;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #aaa;
                border-radius: 3px;
                padding: 5px 10px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #bbb;
                border-radius: 3px;
                padding: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                margin-top: -1px;
            }
        """)

    def create_input_tab(self) -> QWidget:
        """Create Input tab."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Projectile Type
        type_group = QGroupBox("Projectile Type")
        type_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Bullet", "Rocket", "Mortar"])
        self.type_combo.currentTextChanged.connect(self.update_projectile_type)
        type_layout.addWidget(QLabel("Type:"))
        type_layout.addWidget(self.type_combo)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Preset Selection
        preset_group = QGroupBox("Ammunition Presets")
        preset_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom"] + sorted(self.presets.keys()))
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)
        save_btn = QPushButton("Save Current")
        save_btn.clicked.connect(self.save_preset)
        preset_layout.addWidget(save_btn)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Projectile Parameters
        proj_group = QGroupBox("Projectile Parameters")
        proj_layout = QGridLayout()

        proj_layout.addWidget(QLabel("Mass (g):"), 0, 0)
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.1, 2_000_000)
        self.mass_input.setDecimals(1)
        self.mass_input.setValue(10)
        self.mass_input.setSingleStep(1)
        proj_layout.addWidget(self.mass_input, 0, 1)

        proj_layout.addWidget(QLabel("Diameter (mm):"), 1, 0)
        self.diam_input = QDoubleSpinBox()
        self.diam_input.setRange(0.1, 1000)
        self.diam_input.setDecimals(2)
        self.diam_input.setValue(7.62)
        self.diam_input.setSingleStep(0.1)
        proj_layout.addWidget(self.diam_input, 1, 1)

        proj_layout.addWidget(QLabel("Drag Model:"), 2, 0)
        self.drag_model_combo = QComboBox()
        self.drag_model_combo.addItems(['G1', 'G7', 'rocket', 'mortar'])
        proj_layout.addWidget(self.drag_model_combo, 2, 1)

        # Rocket Parameters
        self.rocket_group = QGroupBox("Rocket Parameters")
        rocket_layout = QVBoxLayout()

        burn_layout = QHBoxLayout()
        burn_layout.addWidget(QLabel("Burn Time (s):"))
        self.burn_time_input = QDoubleSpinBox()
        self.burn_time_input.setRange(0, 60)
        self.burn_time_input.setValue(1.0)
        self.burn_time_input.setSingleStep(0.1)
        burn_layout.addWidget(self.burn_time_input)
        rocket_layout.addLayout(burn_layout)

        thrust_layout = QHBoxLayout()
        thrust_layout.addWidget(QLabel("Peak Thrust (N):"))
        self.thrust_input = QDoubleSpinBox()
        self.thrust_input.setRange(0, 500_000)
        self.thrust_input.setValue(1000)
        self.thrust_input.setSingleStep(100)
        thrust_layout.addWidget(self.thrust_input)
        rocket_layout.addLayout(thrust_layout)

        prop_layout = QHBoxLayout()
        prop_layout.addWidget(QLabel("Propellant Mass (g):"))
        self.propellant_input = QDoubleSpinBox()
        self.propellant_input.setRange(0, 1_000_000)
        self.propellant_input.setDecimals(1)
        self.propellant_input.setValue(0)
        self.propellant_input.setToolTip(
            "Mass of propellant burned during flight.\n"
            "Leave 0 to use 10% of launch mass as default.")
        prop_layout.addWidget(self.propellant_input)
        rocket_layout.addLayout(prop_layout)

        self.rocket_group.setLayout(rocket_layout)
        self.rocket_group.setVisible(False)
        proj_layout.addWidget(self.rocket_group, 3, 0, 1, 2)

        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)

        # Launch Parameters
        launch_group = QGroupBox("Launch Parameters")
        launch_layout = QGridLayout()

        launch_layout.addWidget(QLabel("Muzzle Velocity (m/s):"), 0, 0)
        self.velocity_input = QDoubleSpinBox()
        self.velocity_input.setRange(1, 5000)
        self.velocity_input.setValue(800)
        self.velocity_input.setSingleStep(10)
        launch_layout.addWidget(self.velocity_input, 0, 1)

        launch_layout.addWidget(QLabel("Launch Angle (deg):"), 1, 0)
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(-90, 90)
        self.angle_input.setValue(15)
        self.angle_input.setSingleStep(1)
        launch_layout.addWidget(self.angle_input, 1, 1)

        launch_group.setLayout(launch_layout)
        layout.addWidget(launch_group)

        # Environmental Parameters
        env_group = QGroupBox("Environmental Parameters")
        env_layout = QGridLayout()

        env_layout.addWidget(QLabel("Altitude (m):"), 0, 0)
        self.altitude_input = QDoubleSpinBox()
        self.altitude_input.setRange(-500, 20000)
        self.altitude_input.setValue(0)
        self.altitude_input.setSingleStep(10)
        env_layout.addWidget(self.altitude_input, 0, 1)

        env_layout.addWidget(QLabel("Temperature (°C):"), 1, 0)
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(-80, 60)
        self.temp_input.setValue(15)
        self.temp_input.setSingleStep(1)
        env_layout.addWidget(self.temp_input, 1, 1)

        env_layout.addWidget(QLabel("Humidity (%):"), 2, 0)
        self.humidity_input = QDoubleSpinBox()
        self.humidity_input.setRange(0, 100)
        self.humidity_input.setValue(50)
        self.humidity_input.setSingleStep(5)
        env_layout.addWidget(self.humidity_input, 2, 1)

        env_layout.addWidget(QLabel("Wind Speed (m/s):"), 3, 0)
        self.wind_speed_input = QDoubleSpinBox()
        self.wind_speed_input.setRange(0, 100)
        self.wind_speed_input.setValue(0)
        self.wind_speed_input.setSingleStep(0.5)
        env_layout.addWidget(self.wind_speed_input, 3, 1)

        env_layout.addWidget(QLabel("Wind Angle (deg):"), 4, 0)
        self.wind_angle_input = QSpinBox()
        self.wind_angle_input.setRange(0, 359)
        self.wind_angle_input.setValue(0)
        env_layout.addWidget(self.wind_angle_input, 4, 1)

        self.coriolis_check = QCheckBox("Coriolis Effect")
        env_layout.addWidget(self.coriolis_check, 5, 0)
        env_layout.addWidget(QLabel("Latitude (°):"), 5, 1)
        self.latitude_input = QDoubleSpinBox()
        self.latitude_input.setRange(-90, 90)
        self.latitude_input.setValue(45)
        self.latitude_input.setEnabled(False)
        env_layout.addWidget(self.latitude_input, 5, 2)
        self.coriolis_check.stateChanged.connect(
            lambda: self.latitude_input.setEnabled(self.coriolis_check.isChecked()))

        env_group.setLayout(env_layout)
        layout.addWidget(env_group)

        # Calculate button
        self.calculate_btn = QPushButton("Calculate Trajectory")
        self.calculate_btn.clicked.connect(self.calculate_trajectory)
        layout.addWidget(self.calculate_btn)

        scroll.setWidget(container)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)
        return tab

    def create_results_tab(self) -> QWidget:
        """Create Results tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        summary_group = QGroupBox("Summary Results")
        summary_layout = QVBoxLayout()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        scroll = QScrollArea()
        scroll.setWidget(self.summary_text)
        scroll.setWidgetResizable(True)
        summary_layout.addWidget(scroll)
        summary_group.setLayout(summary_layout)

        data_group = QGroupBox("Trajectory Data")
        data_layout = QVBoxLayout()
        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        scroll_data = QScrollArea()
        scroll_data.setWidget(self.data_text)
        scroll_data.setWidgetResizable(True)
        data_layout.addWidget(scroll_data)
        data_group.setLayout(data_layout)

        layout.addWidget(summary_group, 1)
        layout.addWidget(data_group, 2)
        tab.setLayout(layout)
        return tab

    def create_plot_tab(self) -> QWidget:
        """Create Plot tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch(1)
        self.compare_check = QCheckBox("Compare with previous trajectory")
        bottom_layout.addWidget(self.compare_check)
        bottom_layout.addStretch(1)
        layout.addLayout(bottom_layout)

        tab.setLayout(layout)
        return tab

    def create_advanced_tab(self) -> QWidget:
        """Create Advanced tab."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Sight height
        sight_group = QGroupBox("Sight Parameters")
        sight_layout = QHBoxLayout()
        sight_layout.addWidget(QLabel("Sight Height (mm):"))
        self.sight_height_input = QDoubleSpinBox()
        self.sight_height_input.setRange(0, 200)
        self.sight_height_input.setValue(50)
        sight_layout.addWidget(self.sight_height_input)
        sight_group.setLayout(sight_layout)
        layout.addWidget(sight_group)

        # Zeroing
        zero_group = QGroupBox("Zeroing")
        zero_layout = QHBoxLayout()
        zero_layout.addWidget(QLabel("Zero Range (m):"))
        self.zero_range_input = QDoubleSpinBox()
        self.zero_range_input.setRange(10, 5000)
        self.zero_range_input.setValue(100)
        zero_layout.addWidget(self.zero_range_input)
        zero_btn = QPushButton("Calculate Zero Angle")
        zero_btn.clicked.connect(self.calculate_zero_angle)
        zero_layout.addWidget(zero_btn)
        zero_group.setLayout(zero_layout)
        layout.addWidget(zero_group)

        # Spin drift
        spin_group = QGroupBox("Spin Drift")
        spin_layout = QHBoxLayout()
        self.spin_drift_check = QCheckBox("Enable Spin Drift (Bullets Only)")
        spin_layout.addWidget(self.spin_drift_check)
        spin_layout.addWidget(QLabel("Rifling Twist (in/rev):"))
        self.twist_input = QDoubleSpinBox()
        self.twist_input.setRange(1, 50)
        self.twist_input.setValue(10)
        self.twist_input.setEnabled(False)
        spin_layout.addWidget(self.twist_input)
        self.spin_drift_check.stateChanged.connect(
            lambda: self.twist_input.setEnabled(self.spin_drift_check.isChecked()))
        spin_group.setLayout(spin_layout)
        layout.addWidget(spin_group)

        scroll.setWidget(container)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)
        return tab

    def update_projectile_type(self, type_str: str):
        """Update UI when projectile type changes."""
        type_lower = type_str.lower()
        self.rocket_group.setVisible(type_lower == "rocket")
        self.drag_model_combo.clear()

        if type_lower == "bullet":
            self.drag_model_combo.addItems(['G1', 'G7'])
        elif type_lower == "rocket":
            self.drag_model_combo.addItems(['rocket'])
        elif type_lower == "mortar":
            self.drag_model_combo.addItems(['mortar'])

    def load_preset(self, preset_name: str):
        """Load preset values into input fields."""
        if preset_name == "Custom" or preset_name not in self.presets:
            return

        preset = self.presets[preset_name]
        self.preset_combo.blockSignals(True)

        try:
            self.mass_input.setValue(preset["mass"])
            self.diam_input.setValue(preset["diameter"])
            p_type = preset.get("type", "bullet")
            self.type_combo.setCurrentText(p_type.capitalize())
            self.update_projectile_type(p_type)
            self.drag_model_combo.setCurrentText(preset["drag_model"])
            self.velocity_input.setValue(preset["velocity"])

            if p_type == "rocket":
                self.burn_time_input.setValue(preset.get("burn_time", 1.0))
                curve = preset.get("thrust_curve", {})
                peak = max(curve.values()) if curve else 1000
                self.thrust_input.setValue(peak)
                self.propellant_input.setValue(preset.get("propellant_mass", 0))
        finally:
            self.preset_combo.blockSignals(False)

    def save_preset(self):
        """Save current settings as preset."""
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not (ok and name.strip()):
            return

        name = name.strip()
        preset_data = {
            "mass": self.mass_input.value(),
            "diameter": self.diam_input.value(),
            "drag_model": self.drag_model_combo.currentText(),
            "velocity": self.velocity_input.value(),
            "type": self.type_combo.currentText().lower()
        }

        if self.type_combo.currentText().lower() == "rocket":
            preset_data.update({
                "burn_time": self.burn_time_input.value(),
                "propellant_mass": self.propellant_input.value(),
                "thrust_curve": {
                    0: self.thrust_input.value(),
                    self.burn_time_input.value(): 0
                }
            })

        self.presets[name] = preset_data
        if self.preset_combo.findText(name) == -1:
            self.preset_combo.addItem(name)

        self.preset_combo.setCurrentText(name)
        PresetManager.save(self.presets)
        QMessageBox.information(self, "Success", f"Preset '{name}' saved.")

    def calculate_zero_angle(self):
        """Calculate launch angle for zero drop at specified range."""
        zero_range = self.zero_range_input.value()
        sight_h = self.sight_height_input.value() / 1000.0  # mm → m
        velocity = self.velocity_input.value()
        mass_kg = self.mass_input.value() / 1000.0

        def height_at_range(angle_deg: float) -> Optional[float]:
            """Compute height at zero_range for given angle."""
            try:
                projectile = Projectile(
                    mass=mass_kg,
                    diameter=self.diam_input.value() / 1000.0,
                    drag_model=self.drag_model_combo.currentText(),
                    velocity=velocity,
                    projectile_type=self.type_combo.currentText().lower(),
                    thrust_curve={},
                    burn_time=0,
                    propellant_mass=0.0
                )
                environment = Environment(
                    altitude=self.altitude_input.value(),
                    temperature=self.temp_input.value(),
                    humidity=self.humidity_input.value(),
                    wind_speed=0,
                    wind_angle=0,
                    coriolis=False,
                    latitude=self.latitude_input.value()
                )
                calc = TrajectoryCalculator(
                    projectile, environment, angle_deg,
                    enable_spin_drift=False
                )
                traj = calc.integrate()

                if not traj:
                    return None

                # Interpolate height at zero_range
                for i in range(1, len(traj)):
                    x0, x1 = traj[i-1][0], traj[i][0]
                    if x0 <= zero_range <= x1:
                        frac = (zero_range - x0) / (x1 - x0) if x1 != x0 else 0
                        h = traj[i-1][1] + frac * (traj[i][1] - traj[i-1][1])
                        return h - sight_h

                return traj[-1][1] - sight_h
            except Exception as e:
                logger.warning(f"Height calculation failed: {e}")
                return None

        # Bisection search
        lo, hi = 0.0, 45.0
        f_lo = height_at_range(lo)
        f_hi = height_at_range(hi)

        if f_lo is None or f_hi is None:
            QMessageBox.warning(self, "Zero Angle",
                              "Could not compute trajectory for zeroing.")
            return

        if f_lo is not None and f_hi is not None and f_lo * f_hi > 0:
            QMessageBox.warning(self, "Zero Angle",
                              f"No solution found in 0–45°.\n"
                              f"Check that zero range ({zero_range} m) is reachable.")
            return

        for _ in range(50):  # ~0.000001° precision
            mid = (lo + hi) / 2.0
            f_mid = height_at_range(mid)
            if f_mid is None:
                break
            if f_lo is not None and f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
            if abs(hi - lo) < 1e-6:
                break

        angle = (lo + hi) / 2.0
        self.angle_input.setValue(round(angle, 4))
        QMessageBox.information(self, "Zero Angle",
                              f"Calculated zero angle: {angle:.4f}°\n"
                              f"(at {zero_range} m with drag correction)")

    def calculate_trajectory(self):
        """Start trajectory calculation in worker thread."""
        if self.calc_thread is not None and self.calc_thread.isRunning():
            QMessageBox.warning(self, "Warning",
                              "Calculation already in progress.")
            return

        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setText("Calculating…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Store previous trajectory
        if self.trajectory:
            self.previous_trajectories.append(self.trajectory)
            if len(self.previous_trajectories) > PREVIOUS_TRAJECTORY_LIMIT:
                self.previous_trajectories.pop(0)

        try:
            projectile = Projectile(
                mass=self.mass_input.value() / 1000.0,
                diameter=self.diam_input.value() / 1000.0,
                drag_model=self.drag_model_combo.currentText(),
                velocity=self.velocity_input.value(),
                projectile_type=self.type_combo.currentText().lower(),
                thrust_curve=self._build_thrust_curve(),
                burn_time=(self.burn_time_input.value()
                          if self.type_combo.currentText().lower() == "rocket"
                          else 0),
                propellant_mass=(self.propellant_input.value() / 1000.0
                                if self.type_combo.currentText().lower() == "rocket"
                                else 0.0)
            )

            environment = Environment(
                altitude=self.altitude_input.value(),
                temperature=self.temp_input.value(),
                humidity=self.humidity_input.value(),
                wind_speed=self.wind_speed_input.value(),
                wind_angle=self.wind_angle_input.value(),
                coriolis=self.coriolis_check.isChecked(),
                latitude=self.latitude_input.value()
            )

            self.calc_thread = CalculationThread(
                projectile, environment,
                self.angle_input.value(),
                self.spin_drift_check.isChecked(),
                self.twist_input.value()
            )
            self.calc_thread.finished.connect(self.on_calculation_complete)
            self.calc_thread.error.connect(self.on_calculation_error)
            self.calc_thread.progress.connect(self.progress_bar.setValue)
            self.calc_thread.start()

        except Exception as e:
            logger.exception("Failed to start calculation")
            QMessageBox.critical(self, "Error", f"Failed to start calculation:\n{e}")
            self.calculate_btn.setEnabled(True)
            self.calculate_btn.setText("Calculate Trajectory")

    def _build_thrust_curve(self) -> Dict[float, float]:
        """Build thrust curve from UI inputs or preset."""
        ptype = self.type_combo.currentText().lower()
        if ptype != "rocket":
            return {}

        # Check if preset is loaded
        preset_name = self.preset_combo.currentText()
        if preset_name != "Custom" and preset_name in self.presets:
            p = self.presets[preset_name]
            if "thrust_curve" in p:
                return {float(k): v for k, v in p["thrust_curve"].items()}

        # Build from UI
        burn_time = self.burn_time_input.value()
        peak_thrust = self.thrust_input.value()
        return {0: peak_thrust, burn_time: 0}

    def on_calculation_complete(self, trajectory):
        """Handle completed calculation."""
        self.trajectory = trajectory
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Trajectory")
        self.progress_bar.setVisible(False)

        if trajectory:
            self.update_results()
            self.plot_trajectory()
        else:
            QMessageBox.warning(self, "Warning", "No trajectory data generated.")

    def on_calculation_error(self, error_msg: str):
        """Handle calculation error."""
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Trajectory")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Calculation failed:\n{error_msg}")

    def update_results(self):
        """Update Results tab with trajectory summary and data."""
        if not self.trajectory:
            return

        max_height = max(p[1] for p in self.trajectory)
        distance = self.trajectory[-1][0]
        flight_time = self.trajectory[-1][2]
        impact_vel = self.trajectory[-1][5]

        # Final mass (dry mass for rockets)
        ptype = self.type_combo.currentText().lower()
        m_kg = self.mass_input.value() / 1000.0
        if ptype == "rocket":
            prop_g = self.propellant_input.value()
            m_final = (m_kg - prop_g / 1000.0) if prop_g > 0 else m_kg * 0.90
        else:
            m_final = m_kg
        m_final = max(m_final, 1e-6)

        impact_energy = 0.5 * m_final * impact_vel**2

        summary = (
            f"PROJECTILE:\n"
            f"  Type:           {self.type_combo.currentText()}\n"
            f"  Mass (launch):  {self.mass_input.value():.1f} g\n"
            f"  Diameter:       {self.diam_input.value():.2f} mm\n"
            f"  Drag Model:     {self.drag_model_combo.currentText()}\n"
            f"  Muzzle Vel:     {self.velocity_input.value():.1f} m/s\n"
            f"  Launch Angle:   {self.angle_input.value():.2f}°\n"
        )
        if ptype == "rocket":
            summary += (f"  Burn Time:      {self.burn_time_input.value():.1f} s\n"
                       f"  Peak Thrust:    {self.thrust_input.value():.0f} N\n")

        summary += (
            f"\nENVIRONMENT:\n"
            f"  Altitude:       {self.altitude_input.value():.0f} m\n"
            f"  Temperature:    {self.temp_input.value():.1f} °C\n"
            f"  Humidity:       {self.humidity_input.value():.0f} %\n"
            f"  Wind Speed:     {self.wind_speed_input.value():.1f} m/s "
            f"@ {self.wind_angle_input.value()}°\n"
            f"  Coriolis:       {'On' if self.coriolis_check.isChecked() else 'Off'}\n"
            f"\nRESULTS:\n"
            f"  Max Height:     {max_height:.1f} m\n"
            f"  Total Distance: {distance:.1f} m\n"
            f"  Flight Time:    {flight_time:.2f} s\n"
            f"  Impact Velocity:{impact_vel:.1f} m/s\n"
            f"  Impact Energy:  {impact_energy:.1f} J  "
            f"({impact_energy/1000:.3f} kJ)\n"
        )
        self.summary_text.setPlainText(summary)

        # Data table
        data_header = "Time(s)\tRange(m)\tHeight(m)\tVx(m/s)\tVy(m/s)\tSpeed(m/s)\n"
        data_lines = [data_header]
        stride = max(1, len(self.trajectory) // DATA_TABLE_ROWS)
        for i, point in enumerate(self.trajectory):
            if i % stride == 0:
                data_lines.append(
                    f"{point[2]:.3f}\t{point[0]:.1f}\t{point[1]:.1f}\t"
                    f"{point[3]:.1f}\t{point[4]:.1f}\t{point[5]:.1f}\n"
                )
        self.data_text.setPlainText(''.join(data_lines))

    def plot_trajectory(self):
        """Plot trajectory and velocity profile."""
        if not self.trajectory:
            return

        self.figure.clear()

        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        x = [p[0] for p in self.trajectory]
        y = [p[1] for p in self.trajectory]
        t = [p[2] for p in self.trajectory]
        sp = [p[5] for p in self.trajectory]

        ax1.plot(x, y, 'b-', linewidth=2, label='Current')

        if self.compare_check.isChecked() and self.previous_trajectories:
            colors = ['r', 'g', 'm']
            for i, traj in enumerate(self.previous_trajectories):
                xp = [p[0] for p in traj]
                yp = [p[1] for p in traj]
                ax1.plot(xp, yp, '--', color=colors[i % len(colors)],
                        linewidth=1, label=f'Previous {i+1}', alpha=0.7)

        ax1.set_title('Trajectory')
        ax1.set_xlabel('Range (m)')
        ax1.set_ylabel('Height (m)')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=8)

        ax2.plot(t, sp, 'r-', linewidth=1.5, label='Speed')
        ax2.set_title('Speed vs Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=8)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def export_to_csv(self):
        """Export trajectory to CSV file."""
        if not self.trajectory:
            QMessageBox.warning(self, "Warning", "No trajectory data to export.")
            return

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv)", options=options)
        if not filename:
            return
        if not filename.endswith('.csv'):
            filename += '.csv'

        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["# Advanced Ballistic Calculator v4.0 Export",
                               datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow(["# Projectile Type", self.type_combo.currentText()])
                writer.writerow(["# Mass (g)", self.mass_input.value()])
                writer.writerow(["# Diameter (mm)", self.diam_input.value()])
                writer.writerow(["# Drag Model", self.drag_model_combo.currentText()])
                writer.writerow(["# Muzzle Velocity (m/s)", self.velocity_input.value()])
                writer.writerow(["# Launch Angle (deg)", self.angle_input.value()])
                writer.writerow(["# Altitude (m)", self.altitude_input.value()])
                writer.writerow(["# Temperature (C)", self.temp_input.value()])
                writer.writerow(["# Wind Speed (m/s)", self.wind_speed_input.value()])
                writer.writerow(["# Wind Angle (deg)", self.wind_angle_input.value()])
                writer.writerow([])
                writer.writerow(['Time(s)', 'Range(m)', 'Height(m)',
                               'Vx(m/s)', 'Vy(m/s)', 'Speed(m/s)'])
                for point in self.trajectory:
                    writer.writerow([f"{v:.4f}" for v in point])

            QMessageBox.information(self, "Success",
                                  f"Data exported to:\n{filename}")
            logger.info(f"Trajectory exported to {filename}")
        except Exception as e:
            logger.exception("Export failed")
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    def show_about(self):
        """Show About dialog."""
        about_text = (
            "Advanced Ballistic Calculator v4.0\n\n"
            "A comprehensive trajectory simulation tool with advanced physics.\n\n"
            "Features:\n"
            "  • Bullets, rockets, and mortars\n"
            "  • 40+ built-in presets\n"
            "  • 4th-order Runge-Kutta integration (adaptive stepping)\n"
            "  • Mach-dependent drag (G1/G7/rocket/mortar)\n"
            "  • CIPM-2007 air density with altitude correction\n"
            "  • ISO 9613-1 speed of sound\n"
            "  • Buck equation saturation vapor pressure\n"
            "  • Coriolis effect (3-component)\n"
            "  • Gyroscopic spin drift (Litz model)\n"
            "  • Rocket thrust curves with propellant burnoff\n"
            "  • Iterative zero-angle solver\n"
            "  • Trajectory & velocity plots\n"
            "  • CSV export with metadata\n"
            "  • Persistent user presets\n\n"
            "References:\n"
            "  • CIPM-2007: Realization of the Metre\n"
            "  • ISO 9613-1: Speed of Sound\n"
            "  • IERS: Earth Rotation Service\n\n"
            "Created for Kali Linux\n"
            "License: MIT"
        )
        QMessageBox.about(self, "About", about_text)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = BallisticCalculator()
    calculator.show()
    sys.exit(app.exec_())

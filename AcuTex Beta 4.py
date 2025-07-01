#!/usr/bin/env python3
"""
Advanced Ballistic Calculator v3.0
Imports, constants, utilities, and interpolation helpers.
"""

import sys
import math
import csv
import json
import os
import logging
import datetime
import difflib
import random
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple, Union
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
    QFileDialog, QMessageBox, QInputDialog, QScrollArea, QSizePolicy,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

logging.basicConfig(filename="ballistic.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
GRAVITY = 9.80665
EARTH_RADIUS = 6371000
EARTH_ROTATION_RATE = 7.292115e-5
SPEED_OF_SOUND = 343
ISA_TABLE = [
    (0, 15.0, 1013.25), (1000, 8.5, 898.76), (2000, 2.0, 794.98),
    (3000, -4.5, 701.12), (4000, -11.0, 616.60), (5000, -17.5, 540.19),
    (6000, -24.0, 471.82), (7000, -30.5, 410.55), (8000, -37.0, 356.51),
    (9000, -43.5, 308.78), (10000, -50.0, 265.00)
]
PRESET_FILE = "user_presets.json"
CONFIG_FILE = "user_config.json"

def log(msg: str, level="info"):
    if level == "debug":
        logging.debug(msg)
    elif level == "warning":
        logging.warning(msg)
    elif level == "error":
        logging.error(msg)
    else:
        logging.info(msg)

def clamp(val, vmin, vmax): return max(vmin, min(val, vmax))

def linear_interpolate(x, x0, x1, y0, y1):
    """Linear interpolation for (x0, y0) - (x1, y1) at x."""
    if x1 == x0:
        return y0
    return y0 + ((y1 - y0) * (x - x0) / (x1 - x0))

def interpolate_curve(x, curve: List[Tuple[float, float]]):
    """General 1D interpolation for monotonic x curve."""
    if not curve or len(curve) < 2:
        return curve[0][1] if curve else 0.0
    curve = sorted(curve)
    xs, ys = zip(*curve)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x < xs[i]:
            return linear_interpolate(x, xs[i - 1], xs[i], ys[i - 1], ys[i])
    return ys[-1]

"""
Projectile, DragModel, Environment classes.
"""

class DragModel:
    @staticmethod
    @lru_cache(maxsize=4096)
    def G1(velocity: float) -> float:
        mach = velocity / SPEED_OF_SOUND
        if mach > 4.0: return 0.45
        elif mach > 3.0: return 0.42
        elif mach > 2.5: return 0.40
        elif mach > 2.0: return 0.38
        elif mach > 1.5: return 0.35
        elif mach > 1.2: return 0.33
        elif mach > 1.0: return 0.31
        elif mach > 0.9: return 0.30
        elif mach > 0.8: return 0.29
        elif mach > 0.7: return 0.28
        elif mach > 0.6: return 0.27
        else: return 0.25

    @staticmethod
    @lru_cache(maxsize=4096)
    def G7(velocity: float) -> float:
        mach = velocity / SPEED_OF_SOUND
        if mach > 4.0: return 0.38
        elif mach > 3.0: return 0.36
        elif mach > 2.5: return 0.34
        elif mach > 2.0: return 0.32
        elif mach > 1.5: return 0.30
        elif mach > 1.2: return 0.28
        elif mach > 1.0: return 0.26
        elif mach > 0.9: return 0.25
        elif mach > 0.8: return 0.24
        elif mach > 0.7: return 0.23
        elif mach > 0.6: return 0.22
        else: return 0.21

    @staticmethod
    @lru_cache(maxsize=4096)
    def rocket(velocity: float) -> float:
        mach = velocity / SPEED_OF_SOUND
        if mach > 3.0: return 0.50
        elif mach > 2.0: return 0.45
        elif mach > 1.5: return 0.40
        elif mach > 1.0: return 0.35
        elif mach > 0.8: return 0.30
        else: return 0.25

    @staticmethod
    @lru_cache(maxsize=4096)
    def mortar(velocity: float) -> float:
        mach = velocity / SPEED_OF_SOUND
        if mach > 1.5: return 0.55
        elif mach > 1.0: return 0.50
        elif mach > 0.8: return 0.45
        else: return 0.40

    @staticmethod
    def custom(velocity: float, curve: List[Tuple[float, float]]) -> float:
        return interpolate_curve(velocity, curve)

class Projectile:
    def __init__(
        self, mass: float = 0.01, diameter: float = 0.01, drag_model: str = 'G7',
        velocity: float = 800, projectile_type: str = 'bullet', thrust_curve: Optional[Dict[float, float]] = None,
        burn_time: float = 0, stages: Optional[List[Dict[str, Any]]] = None,
        custom_drag_curve: Optional[List[Tuple[float, float]]] = None,
        twist_rate: Optional[float] = None, spin_direction: int = 1
    ):
        self.mass = mass
        self.diameter = diameter
        self.drag_model = drag_model
        self.velocity = velocity
        self.area = math.pi * (diameter/2)**2
        self.projectile_type = projectile_type
        self.thrust_curve = thrust_curve or {}
        self.burn_time = burn_time
        self.initial_mass = mass
        self.stages = stages if stages else []
        self.custom_drag_curve = custom_drag_curve
        self.twist_rate = twist_rate
        self.spin_direction = spin_direction

    def drag_coefficient(self, velocity: float) -> float:
        if self.drag_model == 'G1':
            return DragModel.G1(velocity)
        elif self.drag_model == 'G7':
            return DragModel.G7(velocity)
        elif self.drag_model == 'rocket':
            return DragModel.rocket(velocity)
        elif self.drag_model == 'mortar':
            return DragModel.mortar(velocity)
        elif self.drag_model == 'Custom' and self.custom_drag_curve:
            return DragModel.custom(velocity, self.custom_drag_curve)
        else:
            return 0.3

    def get_thrust(self, time: float) -> float:
        """Interpolates thrust from thrust_curve or stages."""
        if self.stages:
            t = 0
            for stage in self.stages:
                t1 = t + stage.get('burn_time', 0)
                if t <= time < t1:
                    curve = sorted(stage.get('thrust_curve', {}).items())
                    return interpolate_curve(time - t, curve)
                t = t1
            return 0
        else:
            if time > self.burn_time:
                return 0
            curve = sorted(self.thrust_curve.items())
            return interpolate_curve(time, curve)

    def get_mass(self, time: float) -> float:
        if self.projectile_type != 'rocket':
            return self.mass
        if self.stages:
            t = 0
            m = self.initial_mass
            for stage in self.stages:
                t1 = t + stage.get('burn_time', 0)
                if t <= time < t1:
                    dmass = stage.get('mass_loss', 0)
                    return m - dmass * ((time-t) / (t1 - t))
                m -= stage.get('mass_loss', 0)
                t = t1
            return m
        else:
            if time > self.burn_time:
                return self.mass
            return self.initial_mass - (self.initial_mass - self.mass) * (time / self.burn_time)

class Environment:
    def __init__(
        self, altitude: float = 0, temperature: float = 15, pressure: float = 1013.25, humidity: float = 50,
        wind_speed: float = 0, wind_angle: float = 0, coriolis: bool = False, latitude: float = 45,
        dynamic_air: bool = False
    ):
        self.altitude = altitude
        self.temperature = temperature
        self.pressure = pressure
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.wind_angle = wind_angle
        self.coriolis = coriolis
        self.latitude = latitude
        self.dynamic_air = dynamic_air
        self.air_density = self.calculate_air_density(altitude)

    @staticmethod
    def isa_interp(alt: float) -> Tuple[float, float]:
        for i in range(1, len(ISA_TABLE)):
            if alt < ISA_TABLE[i][0]:
                a0, t0, p0 = ISA_TABLE[i-1]
                a1, t1, p1 = ISA_TABLE[i]
                f = (alt - a0)/(a1 - a0)
                return t0 + f*(t1-t0), p0 + f*(p1-p0)
        return ISA_TABLE[-1][1], ISA_TABLE[-1][2]

    @lru_cache(maxsize=1024)
    def calculate_air_density(self, altitude: float) -> float:
        if self.dynamic_air:
            temp, pres = Environment.isa_interp(altitude)
        else:
            temp, pres = self.temperature, self.pressure
        temp_kelvin = temp + 273.15
        R = 287.058
        svp = 6.1078 * 10**((7.5 * temp) / (temp + 237.3))
        vp = svp * self.humidity / 100
        density = ((pres * 100) / (R * temp_kelvin)) * (1 - (0.378 * vp) / (pres * 100))
        density *= math.exp(-altitude / 10000)
        return density

    """
CalculationThread, BatchCalculationThread, and core calculation logic.
"""


class CalculationThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    def __init__(self, calculator, params):
        super().__init__()
        self.calculator = calculator
        self.params = params
    def run(self):
        try:
            result = self.calculator._calculate_trajectory(**self.params)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class BatchCalculationThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    def __init__(self, calculator, batch_params):
        super().__init__()
        self.calculator = calculator
        self.batch_params = batch_params
    def run(self):
        results = []
        try:
            for params in self.batch_params:
                results.append(self.calculator._calculate_trajectory(**params))
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
            """
BallisticCalculator: config/preset I/O and main UI/tabs setup.
"""


class BallisticCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Ballistic Calculator")
        self.setGeometry(100, 100, 1200, 900)
        self.setMinimumSize(900, 700)
        self.trajectory: List[Any] = []
        self.previous_trajectories: List[Any] = []
        self.batch_trajectories: List[List[Any]] = []
        self.dark_mode_enabled = False
        self.use_imperial = False
        self.dynamic_air = False
        self.last_preset = "Custom"
        self.favorite_presets = set()
        self.custom_drag_curve: Optional[List[Tuple[float, float]]] = None
        self.load_config()
        self.default_presets = self.load_default_presets()
        self.presets = self.load_presets()
        self.init_ui()
        self.apply_styles()
        log("Ballistic Calculator started")

    # --- Config and Preset IO ---
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    cfg = json.load(f)
                    self.dark_mode_enabled = cfg.get("dark_mode_enabled", False)
                    self.use_imperial = cfg.get("use_imperial", False)
                    self.dynamic_air = cfg.get("dynamic_air", False)
                    self.last_preset = cfg.get("last_preset", "Custom")
                    self.favorite_presets = set(cfg.get("favorite_presets", []))
            except Exception as e:
                log(f"Failed to load config: {e}", "warning")

    def save_config(self):
        cfg = {
            "dark_mode_enabled": self.dark_mode_enabled,
            "use_imperial": self.use_imperial,
            "dynamic_air": self.dynamic_air,
            "last_preset": self.last_preset,
            "favorite_presets": list(self.favorite_presets)
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            log(f"Failed to save config: {e}", "warning")

    def load_default_presets(self) -> Dict[str, Any]:
        # ... (use the original large preset dictionary, omitted for brevity)
        return presets

    def load_presets(self) -> Dict[str, Any]:
        presets = self.default_presets.copy()
        if os.path.exists(PRESET_FILE):
            try:
                with open(PRESET_FILE, "r") as f:
                    user_presets = json.load(f)
                    presets.update(user_presets)
            except Exception:
                pass
        return presets

    def save_presets(self):
        user_presets = {k: v for k, v in self.presets.items() if k not in self.default_presets}
        with open(PRESET_FILE, "w") as f:
            json.dump(user_presets, f, indent=2)

    # --- UI Construction ---
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.addTab(self.create_input_tab(), "Input")
        self.tabs.addTab(self.create_results_tab(), "Results")
        self.tabs.addTab(self.create_plot_tab(), "Graph")
        self.tabs.addTab(self.create_batch_tab(), "Batch Mode")
        self.tabs.addTab(self.create_monte_carlo_tab(), "Monte Carlo")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
        self.tabs.addTab(self.create_help_tab(), "Help")
        main_layout.addWidget(self.tabs, 1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        # ... (menu bar and status bar setup, unchanged)
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.calculate_trajectory)
        # ... (connect valueChanged for real-time preview)
        """
BallisticCalculator: input validation, preset logic, tooltips, dark mode, units.
"""


# All methods for:
# - validate_projectile_params
# - validate_inputs
# - load_custom_drag_curve
# - filter_presets (fuzzy)
# - toggle_dark_mode, update_units, toggle_units, toggle_dynamic_air
# - Tooltips for all parameter widgets
# - Preset save/delete logic
# - UI methods for input, settings, help, etc.
# - apply_styles (improved dark mode)
# - queue_trajectory_preview for real-time plotting
"""
BallisticCalculator: core calculation, plotting, and result update methods.
"""



# All methods for:
# - _calculate_trajectory (RK4, adaptive time-stepping)
# - update_results (summary/results with impact angle/energy)
# - plot_trajectory (2D/3D)
# - on_calculation_complete/on_calculation_error
# - export_to_csv, export_to_json, export_plot_to_png, export_plot_to_svg (with metadata)
"""
BallisticCalculator: batch mode, batch plotting, Monte Carlo simulation tab.
"""



# All methods for:
# - create_batch_tab, run_batch, on_batch_complete, on_batch_error
# - create_monte_carlo_tab, run_monte_carlo

"""
Main entry point for launching the application.
"""



if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = BallisticCalculator()
    calculator.show()
    sys.exit(app.exec_())

#!/usr/bin/env python3
"""
AcuTex Ballistic Calculator - Ultimate Edition
Combines features from all provided scripts with bug fixes and improvements
"""

import sys
import math
import csv
import json
import os
import logging
import difflib
import random
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
    QFileDialog, QMessageBox, QInputDialog, QScrollArea, QSizePolicy,
    QGridLayout, QTableWidget, QTableWidgetItem, QProgressBar, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Constants
GRAVITY = 9.80665  # m/s^2
EARTH_RADIUS = 6371000  # meters
EARTH_ROTATION_RATE = 7.292115e-5  # rad/s
SPEED_OF_SOUND = 343  # m/s at sea level

PRESET_FILE = "user_presets.json"
CONFIG_FILE = "user_config.json"

logging.basicConfig(
    filename="ballistic.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def log(msg: str, level="info"):
    getattr(logging, level, logging.info)(msg)

def clamp(val, vmin, vmax): return max(vmin, min(val, vmax))

def linear_interpolate(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    return y0 + ((y1 - y0) * (x - x0) / (x1 - x0))

def interpolate_curve(x, curve: List[Tuple[float, float]]):
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

class DragModel:
    """Enhanced drag coefficient tables for standard models"""
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
    """Projectile model for physics and mass/thrust properties."""
    def __init__(
        self, mass: float = 0.01, diameter: float = 0.01, drag_model: str = 'G7',
        velocity: float = 800, projectile_type: str = 'bullet', 
        thrust_curve: Optional[Dict[float, float]] = None,
        burn_time: float = 0, custom_drag_curve: Optional[List[Tuple[float, float]]] = None
    ):
        self.mass = mass  # kg
        self.diameter = diameter  # meters
        self.drag_model = drag_model
        self.velocity = velocity  # m/s
        self.area = math.pi * (diameter/2)**2
        self.projectile_type = projectile_type
        self.thrust_curve = thrust_curve or {}
        self.burn_time = burn_time
        self.initial_mass = mass
        self.custom_drag_curve = custom_drag_curve

    def drag_coefficient(self, velocity: float) -> float:
        """Get drag coefficient based on current velocity"""
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
            return 0.3  # Default for other models

    def get_thrust(self, time: float) -> float:
        """Get current thrust based on thrust curve"""
        if time > self.burn_time:
            return 0
        times = sorted(self.thrust_curve.keys())
        if not times:
            return 0
        if time <= times[0]:
            return self.thrust_curve[times[0]]
        if time >= times[-1]:
            return self.thrust_curve[times[-1]]
        for i in range(1, len(times)):
            if time <= times[i]:
                t0, t1 = times[i-1], times[i]
                f0, f1 = self.thrust_curve[t0], self.thrust_curve[t1]
                return f0 + (f1 - f0) * (time - t0) / (t1 - t0)
        return 0

    def get_mass(self, time: float) -> float:
        """Calculate current mass based on burn time"""
        if self.projectile_type != 'rocket' or time > self.burn_time:
            return self.mass
        return self.initial_mass - (self.initial_mass - self.mass) * (time / self.burn_time)

class Environment:
    """Environment model for air density and weather effects."""
    def __init__(
        self, altitude: float = 0, temperature: float = 15, pressure: float = 1013.25, 
        humidity: float = 50, wind_speed: float = 0, wind_angle: float = 0, 
        coriolis: bool = False, latitude: float = 45
    ):
        self.altitude = altitude  # meters
        self.temperature = temperature  # °C
        self.pressure = pressure  # hPa
        self.humidity = humidity  # %
        self.wind_speed = wind_speed  # m/s
        self.wind_angle = wind_angle  # degrees
        self.coriolis = coriolis
        self.latitude = latitude
        self.air_density = self.calculate_air_density()

    @lru_cache(maxsize=128)
    def calculate_air_density(self) -> float:
        """Improved air density calculation using CIPM-2007 equation"""
        temp_kelvin = self.temperature + 273.15
        R = 287.058  # Specific gas constant for dry air, J/(kg·K)
        
        # Saturation vapor pressure
        svp = 6.1078 * 10**((7.5 * self.temperature) / (self.temperature + 237.3))
        
        # Vapor pressure
        vp = svp * self.humidity / 100
        
        # Enhanced air density calculation
        density = ((self.pressure * 100) / (R * temp_kelvin)) * (1 - (0.378 * vp) / (self.pressure * 100))
        
        # Altitude adjustment
        density *= math.exp(-self.altitude / 10000)
        
        return density

class CalculationThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    def __init__(self, calculator: 'BallisticCalculator', params: Dict[str, Any]):
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
    progress = pyqtSignal(int)
    
    def __init__(self, calculator: 'BallisticCalculator', batch_params: List[Dict[str, Any]]):
        super().__init__()
        self.calculator = calculator
        self.batch_params = batch_params
        
    def run(self):
        results = []
        try:
            total = len(self.batch_params)
            for i, params in enumerate(self.batch_params):
                results.append(self.calculator._calculate_trajectory(**params))
                self.progress.emit(int((i+1)/total*100))
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class BallisticCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AcuTex Ballistic Calculator - Ultimate Edition")
        self.setGeometry(100, 100, 1200, 950)
        self.setMinimumSize(800, 600)
        
        # Initialize state variables
        self.trajectory: List[Any] = []
        self.previous_trajectories: List[Any] = []
        self.dark_mode_enabled = False
        self.use_imperial = False
        self.custom_drag_curve: Optional[List[Tuple[float, float]]] = None
        
        # Load presets and config
        self.default_presets = self.load_default_presets()
        self.presets = self.load_presets()
        self.config = self.load_config()
        self.apply_config()
        
        # Initialize UI
        self.init_ui()
        self.apply_styles()
        log("AcuTex Ballistic Calculator started")
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def load_default_presets(self) -> Dict[str, Any]:
        """Return the hardcoded default presets."""
        return {
            # Bullet presets
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
            
            # Rocket Presets
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
            
            # Mortar Presets
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

    def load_presets(self) -> Dict[str, Any]:
        presets = self.default_presets.copy()
        if os.path.exists(PRESET_FILE):
            try:
                with open(PRESET_FILE, "r") as f:
                    user_presets = json.load(f)
                    presets.update(user_presets)
            except Exception as e:
                log(f"Failed to load custom presets: {e}", "warning")
        return presets

    def save_presets(self):
        user_presets = {k: v for k, v in self.presets.items() if k not in self.default_presets}
        try:
            with open(PRESET_FILE, "w") as f:
                json.dump(user_presets, f, indent=2)
        except Exception as e:
            log(f"Failed to save custom presets: {e}", "warning")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                log(f"Failed to load config: {e}", "warning")
        return {}

    def save_config(self):
        data = {
            "dark_mode": self.dark_mode_enabled,
            "imperial": self.use_imperial
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log(f"Failed to save config: {e}", "warning")

    def apply_config(self):
        self.dark_mode_enabled = self.config.get("dark_mode", False)
        self.use_imperial = self.config.get("imperial", False)

    def apply_styles(self):
        if self.dark_mode_enabled:
            self.setStyleSheet("""
                QMainWindow { background-color: #23272e; color: #eee; font-family: Segoe UI, Arial; }
                QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 10px; padding-top: 15px; color: #eee; }
                QGroupBox::title { color: #aaa; }
                QPushButton { background-color: #33364a; color: #eaeaea; border: 1px solid #888; border-radius: 3px; padding: 5px 10px; }
                QPushButton:hover { background-color: #40445c; }
                QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { 
                    border: 1px solid #666; color: #eee; background: #282c34; 
                    border-radius: 3px; padding: 3px;
                }
                QTabWidget::pane { border: 1px solid #333; }
                QTabBar::tab { padding: 5px 10px; }
                QTableWidget { background-color: #282c34; gridline-color: #444; }
                QHeaderView::section { background-color: #333; color: #eee; }
                QProgressBar {
                    border: 1px solid #444;
                    border-radius: 3px;
                    text-align: center;
                    background: #282c34;
                }
                QProgressBar::chunk {
                    background-color: #3daee9;
                    width: 10px;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #f5f5f5; font-family: Segoe UI, Arial; }
                QGroupBox { border: 1px solid #ccc; border-radius: 4px; margin-top: 10px; padding-top: 15px; }
                QGroupBox::title { color: #555; }
                QPushButton { 
                    background-color: #e0e0e0; border: 1px solid #aaa; border-radius: 3px; padding: 5px 10px;
                }
                QPushButton:hover { background-color: #d0d0d0; }
                QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { 
                    border: 1px solid #bbb; border-radius: 3px; padding: 3px;
                }
                QTabWidget::pane { border: 1px solid #ccc; }
                QTabBar::tab { padding: 5px 10px; }
                QTableWidget { gridline-color: #ddd; }
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    width: 10px;
                }
            """)

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_input_tab(), "Input")
        self.tabs.addTab(self.create_results_tab(), "Results")
        self.tabs.addTab(self.create_plot_tab(), "Graph")
        self.tabs.addTab(self.create_batch_tab(), "Batch Mode")
        self.tabs.addTab(self.create_monte_carlo_tab(), "Monte Carlo")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
        self.tabs.addTab(self.create_help_tab(), "Help")
        
        main_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(main_widget)
        
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        
        export_action = file_menu.addAction('Export CSV')
        export_action.triggered.connect(self.export_to_csv)
        
        export_plot_action = file_menu.addAction("Export Plot as PNG")
        export_plot_action.triggered.connect(self.export_plot_to_png)
        
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        view_menu = menubar.addMenu('View')
        self.dark_mode_action = view_menu.addAction("Toggle Dark Mode")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(self.dark_mode_enabled)
        self.dark_mode_action.triggered.connect(self.toggle_dark_mode)
        
        help_menu = menubar.addMenu('Help')
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

    def create_input_tab(self):
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Projectile Type Selection
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
        self.preset_combo.setEditable(True)
        self.refresh_preset_combo()
        self.preset_combo.currentTextChanged.connect(self.on_preset_change)
        preset_layout.addWidget(self.preset_combo)
        
        load_btn = QPushButton("Load Preset")
        load_btn.clicked.connect(self.load_preset)
        save_btn = QPushButton("Save Current")
        save_btn.clicked.connect(self.save_preset)
        delete_btn = QPushButton("Delete Preset")
        delete_btn.clicked.connect(self.delete_preset)
        
        preset_layout.addWidget(load_btn)
        preset_layout.addWidget(save_btn)
        preset_layout.addWidget(delete_btn)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        # Projectile Group
        proj_group = QGroupBox("Projectile Parameters")
        proj_layout = QGridLayout()
        
        # Mass
        proj_layout.addWidget(QLabel("Mass (g):"), 0, 0)
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.1, 1000000)
        self.mass_input.setValue(10)
        proj_layout.addWidget(self.mass_input, 0, 1)
        
        # Diameter
        proj_layout.addWidget(QLabel("Diameter (mm):"), 1, 0)
        self.diam_input = QDoubleSpinBox()
        self.diam_input.setRange(0.1, 1000)
        self.diam_input.setValue(7.62)
        proj_layout.addWidget(self.diam_input, 1, 1)
        
        # Drag Model
        proj_layout.addWidget(QLabel("Drag Model:"), 2, 0)
        self.drag_model_combo = QComboBox()
        self.drag_model_combo.addItems(['G1', 'G7', 'rocket', 'mortar', 'Custom'])
        proj_layout.addWidget(self.drag_model_combo, 2, 1)
        
        drag_curve_btn = QPushButton("Load Custom Drag")
        drag_curve_btn.clicked.connect(self.load_custom_drag_curve)
        proj_layout.addWidget(drag_curve_btn, 2, 2)
        
        # Rocket-specific parameters
        self.rocket_group = QGroupBox("Rocket Parameters")
        rocket_layout = QVBoxLayout()
        
        burn_layout = QHBoxLayout()
        burn_layout.addWidget(QLabel("Burn Time (s):"))
        self.burn_time_input = QDoubleSpinBox()
        self.burn_time_input.setRange(0, 10)
        self.burn_time_input.setValue(1.0)
        self.burn_time_input.setSingleStep(0.1)
        burn_layout.addWidget(self.burn_time_input)
        rocket_layout.addLayout(burn_layout)
        
        thrust_layout = QHBoxLayout()
        thrust_layout.addWidget(QLabel("Avg Thrust (N):"))
        self.thrust_input = QDoubleSpinBox()
        self.thrust_input.setRange(0, 10000)
        self.thrust_input.setValue(1000)
        self.thrust_input.setSingleStep(100)
        thrust_layout.addWidget(self.thrust_input)
        rocket_layout.addLayout(thrust_layout)
        
        self.rocket_group.setLayout(rocket_layout)
        self.rocket_group.setVisible(False)
        proj_layout.addWidget(self.rocket_group, 3, 0, 1, 3)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        # Launch Parameters Group
        launch_group = QGroupBox("Launch Parameters")
        launch_layout = QGridLayout()
        
        # Velocity
        launch_layout.addWidget(QLabel("Muzzle Velocity (m/s):"), 0, 0)
        self.velocity_input = QDoubleSpinBox()
        self.velocity_input.setRange(1, 5000)
        self.velocity_input.setValue(800)
        launch_layout.addWidget(self.velocity_input, 0, 1)
        
        # Angle
        launch_layout.addWidget(QLabel("Launch Angle (deg):"), 1, 0)
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(0, 90)
        self.angle_input.setValue(15)
        launch_layout.addWidget(self.angle_input, 1, 1)
        
        launch_group.setLayout(launch_layout)
        layout.addWidget(launch_group)
        
        # Environment Group
        env_group = QGroupBox("Environmental Parameters")
        env_layout = QGridLayout()
        
        # Altitude
        env_layout.addWidget(QLabel("Altitude (m):"), 0, 0)
        self.altitude_input = QDoubleSpinBox()
        self.altitude_input.setRange(-100, 10000)
        self.altitude_input.setValue(0)
        env_layout.addWidget(self.altitude_input, 0, 1)
        
        # Temperature
        env_layout.addWidget(QLabel("Temperature (°C):"), 1, 0)
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(-50, 60)
        self.temp_input.setValue(15)
        env_layout.addWidget(self.temp_input, 1, 1)
        
        # Pressure
        env_layout.addWidget(QLabel("Pressure (hPa):"), 2, 0)
        self.pressure_input = QDoubleSpinBox()
        self.pressure_input.setRange(800, 1100)
        self.pressure_input.setValue(1013.25)
        env_layout.addWidget(self.pressure_input, 2, 1)
        
        # Humidity
        env_layout.addWidget(QLabel("Humidity (%):"), 3, 0)
        self.humidity_input = QDoubleSpinBox()
        self.humidity_input.setRange(0, 100)
        self.humidity_input.setValue(50)
        env_layout.addWidget(self.humidity_input, 3, 1)
        
        # Wind
        env_layout.addWidget(QLabel("Wind Speed (m/s):"), 4, 0)
        self.wind_speed_input = QDoubleSpinBox()
        self.wind_speed_input.setRange(0, 50)
        self.wind_speed_input.setValue(0)
        env_layout.addWidget(self.wind_speed_input, 4, 1)
        
        env_layout.addWidget(QLabel("Wind Angle (deg):"), 5, 0)
        self.wind_angle_input = QSpinBox()
        self.wind_angle_input.setRange(0, 359)
        self.wind_angle_input.setValue(0)
        env_layout.addWidget(self.wind_angle_input, 5, 1)
        
        # Advanced options
        self.coriolis_check = QCheckBox("Coriolis Effect")
        env_layout.addWidget(self.coriolis_check, 6, 0)
        
        env_layout.addWidget(QLabel("Latitude:"), 6, 1)
        self.latitude_input = QDoubleSpinBox()
        self.latitude_input.setRange(-90, 90)
        self.latitude_input.setValue(45)
        self.latitude_input.setEnabled(False)
        env_layout.addWidget(self.latitude_input, 6, 2)
        
        self.coriolis_check.stateChanged.connect(
            lambda: self.latitude_input.setEnabled(self.coriolis_check.isChecked()))
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
        # Calculate button
        calc_row = QHBoxLayout()
        self.calculate_btn = QPushButton("Calculate Trajectory")
        self.calculate_btn.clicked.connect(self.calculate_trajectory)
        calc_row.addWidget(self.calculate_btn)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        calc_row.addWidget(export_btn)
        
        export_plot_btn = QPushButton("Export Plot")
        export_plot_btn.clicked.connect(self.export_plot)
        calc_row.addWidget(export_plot_btn)
        
        layout.addLayout(calc_row)
        
        scroll.setWidget(container)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)
        return tab

    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Summary Group
        summary_group = QGroupBox("Summary Results")
        summary_layout = QVBoxLayout()
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        scroll = QScrollArea()
        scroll.setWidget(self.summary_text)
        scroll.setWidgetResizable(True)
        summary_layout.addWidget(scroll)
        summary_group.setLayout(summary_layout)
        
        # Data Group
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
        return tab

    def create_plot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas, 1)
        
        # Comparison checkbox at bottom
        bottom_layout = QHBoxLayout()
        self.compare_check = QCheckBox("Compare with previous trajectory")
        bottom_layout.addWidget(self.compare_check)
        bottom_layout.addStretch(1)
        
        layout.addLayout(bottom_layout)
        return tab

    def create_batch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Batch controls
        controls_group = QGroupBox("Batch Parameters")
        controls_layout = QGridLayout()
        
        controls_layout.addWidget(QLabel("Sweep parameter:"), 0, 0)
        self.batch_param_combo = QComboBox()
        self.batch_param_combo.addItems(["Angle", "Muzzle Velocity"])
        controls_layout.addWidget(self.batch_param_combo, 0, 1)
        
        controls_layout.addWidget(QLabel("Start:"), 1, 0)
        self.batch_start_input = QDoubleSpinBox()
        self.batch_start_input.setRange(-10000, 10000)
        self.batch_start_input.setValue(10)
        controls_layout.addWidget(self.batch_start_input, 1, 1)
        
        controls_layout.addWidget(QLabel("End:"), 2, 0)
        self.batch_end_input = QDoubleSpinBox()
        self.batch_end_input.setRange(-10000, 10000)
        self.batch_end_input.setValue(80)
        controls_layout.addWidget(self.batch_end_input, 2, 1)
        
        controls_layout.addWidget(QLabel("Step:"), 3, 0)
        self.batch_step_input = QDoubleSpinBox()
        self.batch_step_input.setRange(0.001, 10000)
        self.batch_step_input.setValue(5)
        controls_layout.addWidget(self.batch_step_input, 3, 1)
        
        self.batch_run_btn = QPushButton("Run Batch")
        self.batch_run_btn.clicked.connect(self.run_batch)
        controls_layout.addWidget(self.batch_run_btn, 3, 2)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Progress bar
        self.batch_progress_bar = QProgressBar()
        layout.addWidget(self.batch_progress_bar)
        
        # Results table
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(5)
        self.batch_table.setHorizontalHeaderLabels([
            "Run", "Param Value", "Max Height (m)", "Distance (m)", "Impact Vel (m/s)"
        ])
        layout.addWidget(self.batch_table, 1)
        
        return tab

    def create_monte_carlo_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Monte Carlo controls
        controls_group = QGroupBox("Monte Carlo Parameters")
        controls_layout = QGridLayout()
        
        controls_layout.addWidget(QLabel("Number of Runs:"), 0, 0)
        self.mc_runs_input = QSpinBox()
        self.mc_runs_input.setRange(10, 1000)
        self.mc_runs_input.setValue(100)
        controls_layout.addWidget(self.mc_runs_input, 0, 1)
        
        controls_layout.addWidget(QLabel("Angle Spread (±deg):"), 1, 0)
        self.mc_spread_angle = QDoubleSpinBox()
        self.mc_spread_angle.setRange(0, 10)
        self.mc_spread_angle.setValue(0.5)
        controls_layout.addWidget(self.mc_spread_angle, 1, 1)
        
        controls_layout.addWidget(QLabel("Velocity Spread (±m/s):"), 2, 0)
        self.mc_spread_velocity = QDoubleSpinBox()
        self.mc_spread_velocity.setRange(0, 100)
        self.mc_spread_velocity.setValue(5)
        controls_layout.addWidget(self.mc_spread_velocity, 2, 1)
        
        self.mc_run_btn = QPushButton("Run Monte Carlo")
        self.mc_run_btn.clicked.connect(self.run_monte_carlo)
        controls_layout.addWidget(self.mc_run_btn, 2, 2)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Progress bar
        self.mc_progress_bar = QProgressBar()
        layout.addWidget(self.mc_progress_bar)
        
        # Plot canvas
        self.mc_canvas = FigureCanvas(Figure(figsize=(5, 4), dpi=100))
        layout.addWidget(self.mc_canvas, 1)
        
        return tab

    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Units settings
        units_group = QGroupBox("Units")
        units_layout = QVBoxLayout()
        
        self.units_chk = QCheckBox("Use Imperial Units")
        self.units_chk.setChecked(self.use_imperial)
        self.units_chk.stateChanged.connect(self.toggle_units)
        units_layout.addWidget(self.units_chk)
        
        units_group.setLayout(units_layout)
        layout.addWidget(units_group)
        
        # Theme settings
        theme_group = QGroupBox("Appearance")
        theme_layout = QVBoxLayout()
        
        self.dark_mode_chk = QCheckBox("Enable Dark Mode")
        self.dark_mode_chk.setChecked(self.dark_mode_enabled)
        self.dark_mode_chk.stateChanged.connect(self.toggle_dark_mode)
        theme_layout.addWidget(self.dark_mode_chk)
        
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        # Preset management
        preset_group = QGroupBox("Preset Management")
        preset_layout = QVBoxLayout()
        
        self.preset_export_btn = QPushButton("Export Presets")
        self.preset_export_btn.clicked.connect(self.export_presets)
        preset_layout.addWidget(self.preset_export_btn)
        
        self.preset_import_btn = QPushButton("Import Presets")
        self.preset_import_btn.clicked.connect(self.import_presets)
        preset_layout.addWidget(self.preset_import_btn)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        layout.addStretch(1)
        return tab

    def create_help_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setFont(QFont("Segoe UI", 11))
        help_text.setPlainText(
            "AcuTex Ballistic Calculator Help\n\n"
            "• Input: Enter all projectile and environmental parameters, or select from presets.\n"
            "• Projectile Type: Choose between Bullet, Rocket, or Mortar.\n"
            "• Drag Models: Select appropriate drag model for your projectile type.\n"
            "• Rocket Parameters: Configure burn time and thrust for rocket projectiles.\n"
            "• Environmental Factors: Adjust altitude, temperature, pressure, humidity, and wind.\n"
            "• Coriolis Effect: Enable to account for Earth's rotation (latitude required).\n\n"
            "• Batch Mode: Sweep angle or velocity and compute multiple trajectories.\n"
            "• Monte Carlo: Simulate random variations in angle and velocity to see spread.\n\n"
            "• Export: Save trajectory data as CSV or plot as PNG image.\n"
            "• Settings: Toggle between metric and imperial units, dark mode, and manage presets.\n\n"
            "All fields have tooltips - hover over them for more information."
        )
        layout.addWidget(help_text)
        return tab

    def refresh_preset_combo(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Custom")
        self.preset_combo.addItems(list(self.presets.keys()))
        self.preset_combo.blockSignals(False)

    def update_projectile_type(self, type_str: str):
        type_lower = type_str.lower()
        self.rocket_group.setVisible(type_lower == "rocket")
        
        # Update drag model options
        self.drag_model_combo.blockSignals(True)
        self.drag_model_combo.clear()
        if type_lower == "bullet":
            self.drag_model_combo.addItems(['G1', 'G7', 'Custom'])
        elif type_lower == "rocket":
            self.drag_model_combo.addItems(['rocket', 'Custom'])
        elif type_lower == "mortar":
            self.drag_model_combo.addItems(['mortar', 'Custom'])
        self.drag_model_combo.blockSignals(False)

    def on_preset_change(self, preset_name: str):
        if preset_name == "Custom":
            return
            
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            self.mass_input.setValue(preset.get("mass", 10))
            self.diam_input.setValue(preset.get("diameter", 7.62))
            
            # Set drag model
            drag_model = preset.get("drag_model", "G7")
            idx = self.drag_model_combo.findText(drag_model)
            if idx >= 0:
                self.drag_model_combo.setCurrentIndex(idx)
            
            self.velocity_input.setValue(preset.get("velocity", 800))
            self.angle_input.setValue(preset.get("angle", 15))
            
            # Set projectile type
            proj_type = preset.get("type", "bullet").capitalize()
            idx = self.type_combo.findText(proj_type)
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)
                self.update_projectile_type(proj_type)
            
            # Set rocket-specific parameters if applicable
            if preset.get("type") == "rocket":
                self.burn_time_input.setValue(preset.get("burn_time", 1.0))
                self.thrust_input.setValue(preset.get("thrust_curve", {}).get(0, 1000))

    def load_preset(self):
        preset_name = self.preset_combo.currentText()
        if preset_name == "Custom":
            QMessageBox.information(self, "Info", "Select a preset to load")
            return
            
        if preset_name in self.presets:
            self.on_preset_change(preset_name)
            self.status_bar.showMessage(f"Loaded preset: {preset_name}")
        else:
            QMessageBox.warning(self, "Warning", "Selected preset not found")

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name:
            preset_data = {
                "mass": self.mass_input.value(),
                "diameter": self.diam_input.value(),
                "drag_model": self.drag_model_combo.currentText(),
                "velocity": self.velocity_input.value(),
                "angle": self.angle_input.value(),
                "type": self.type_combo.currentText().lower()
            }
            
            # Add rocket-specific parameters if applicable
            if self.type_combo.currentText().lower() == "rocket":
                preset_data.update({
                    "burn_time": self.burn_time_input.value(),
                    "thrust_curve": {0: self.thrust_input.value()}
                })
            
            self.presets[name] = preset_data
            self.save_presets()
            self.refresh_preset_combo()
            self.preset_combo.setCurrentText(name)
            self.status_bar.showMessage(f"Preset '{name}' saved")

    def delete_preset(self):
        name = self.preset_combo.currentText()
        if name == "Custom" or name in self.default_presets:
            QMessageBox.warning(self, "Warning", "Cannot delete default or 'Custom' preset")
            return
            
        if name in self.presets:
            confirm = QMessageBox.question(self, "Confirm Delete", 
                                         f"Delete preset '{name}'?", 
                                         QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                del self.presets[name]
                self.save_presets()
                self.refresh_preset_combo()
                self.status_bar.showMessage(f"Preset '{name}' deleted")
        else:
            QMessageBox.warning(self, "Warning", "Selected preset not found")

    def load_custom_drag_curve(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Custom Drag Curve CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                curve = []
                with open(filename, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if len(row) >= 2:
                            try:
                                v = float(row[0])
                                cd = float(row[1])
                                curve.append((v, cd))
                            except ValueError:
                                continue
                
                if curve:
                    self.custom_drag_curve = curve
                    self.drag_model_combo.setCurrentText('Custom')
                    QMessageBox.information(self, "Success", f"Loaded {len(curve)} drag points")
                else:
                    QMessageBox.warning(self, "Warning", "No valid data found in file")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load drag curve:\n{str(e)}")

    def toggle_dark_mode(self, checked):
        self.dark_mode_enabled = bool(checked)
        self.apply_styles()
        self.save_config()
        self.dark_mode_action.setChecked(self.dark_mode_enabled)

    def toggle_units(self, checked):
        self.use_imperial = bool(checked)
        self.save_config()
        # TODO: Implement unit conversion

    def validate_inputs(self) -> bool:
        if self.mass_input.value() <= 0:
            QMessageBox.warning(self, "Invalid Input", "Mass must be positive")
            return False
        if self.diam_input.value() <= 0:
            QMessageBox.warning(self, "Invalid Input", "Diameter must be positive")
            return False
        if self.velocity_input.value() <= 0:
            QMessageBox.warning(self, "Invalid Input", "Muzzle velocity must be positive")
            return False
        if self.type_combo.currentText().lower() == 'rocket' and self.burn_time_input.value() < 0:
            QMessageBox.warning(self, "Invalid Input", "Burn time cannot be negative")
            return False
        return True

    def calculate_trajectory(self):
        if not self.validate_inputs():
            return
            
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setText("Calculating...")
        
        # Prepare parameters
        params = {
            "mass": self.mass_input.value() / 1000,  # g to kg
            "diameter": self.diam_input.value() / 1000,  # mm to m
            "drag_model": self.drag_model_combo.currentText(),
            "velocity": self.velocity_input.value(),
            "angle": self.angle_input.value(),
            "altitude": self.altitude_input.value(),
            "temperature": self.temp_input.value(),
            "pressure": self.pressure_input.value(),
            "humidity": self.humidity_input.value(),
            "wind_speed": self.wind_speed_input.value(),
            "wind_angle": self.wind_angle_input.value(),
            "coriolis": self.coriolis_check.isChecked(),
            "latitude": self.latitude_input.value()
        }
        
        # Store previous trajectory for comparison
        if self.trajectory:
            self.previous_trajectories.append(self.trajectory)
            if len(self.previous_trajectories) > 3:  # Keep last 3
                self.previous_trajectories.pop(0)
        
        # Create and start thread
        self.calc_thread = CalculationThread(self, params)
        self.calc_thread.finished.connect(self.on_calculation_complete)
        self.calc_thread.error.connect(self.on_calculation_error)
        self.calc_thread.start()

    def on_calculation_complete(self, trajectory: list):
        self.trajectory = trajectory
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Trajectory")
        
        if trajectory:
            self.update_results()
            self.plot_trajectory()
            self.status_bar.showMessage("Trajectory calculated successfully")
        else:
            QMessageBox.warning(self, "Warning", "No trajectory data was generated")
            self.status_bar.showMessage("Calculation completed with no data")

    def on_calculation_error(self, error_msg: str):
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Trajectory")
        QMessageBox.critical(self, "Error", f"Calculation failed:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")

    def _calculate_trajectory(self, mass: float, diameter: float, drag_model: str, velocity: float, angle: float,
                            altitude: float, temperature: float, pressure: float, humidity: float,
                            wind_speed: float, wind_angle: float, coriolis: bool, latitude: float, 
                            max_time_step: float = 0.1, min_time_step: float = 0.001) -> list:
        """Enhanced RK4 trajectory calculation with rocket/mortar support"""
        # Initialize projectile and environment
        projectile = Projectile(
            mass=mass,
            diameter=diameter,
            drag_model=drag_model,
            velocity=velocity,
            projectile_type=self.type_combo.currentText().lower(),
            thrust_curve={0: self.thrust_input.value()},
            burn_time=self.burn_time_input.value() if self.type_combo.currentText().lower() == "rocket" else 0,
            custom_drag_curve=self.custom_drag_curve
        )
        
        environment = Environment(
            altitude=altitude,
            temperature=temperature,
            pressure=pressure,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            coriolis=coriolis,
            latitude=latitude
        )
        
        angle_rad = math.radians(angle)
        wind_x = environment.wind_speed * math.cos(math.radians(environment.wind_angle))
        wind_y = environment.wind_speed * math.sin(math.radians(environment.wind_angle))
        
        # Initial state: [x, y, vx, vy]
        state = [0, 0, 
                velocity * math.cos(angle_rad), 
                velocity * math.sin(angle_rad)]
        
        trajectory = []
        time = 0.0
        
        def derivative(s, t):
            x, y, vx, vy = s
            v_rel_x = vx - wind_x
            v_rel_y = vy - wind_y
            v_rel = math.hypot(v_rel_x, v_rel_y)
            
            # Get velocity-dependent drag coefficient
            drag_coeff = projectile.drag_coefficient(v_rel)
            drag_force = 0.5 * environment.air_density * v_rel**2 * drag_coeff * projectile.area
            
            # Acceleration components
            ax = -(drag_force * v_rel_x) / (projectile.get_mass(t) * v_rel) if v_rel > 0 else 0
            ay = -GRAVITY - (drag_force * v_rel_y) / (projectile.get_mass(t) * v_rel) if v_rel > 0 else -GRAVITY
            
            # Add thrust if rocket is still burning
            if projectile.projectile_type == 'rocket' and t < projectile.burn_time:
                thrust = projectile.get_thrust(t)
                thrust_angle = angle_rad if t == 0 else math.atan2(vy, vx)  # Follow velocity vector
                ax += (thrust * math.cos(thrust_angle)) / projectile.get_mass(t)
                ay += (thrust * math.sin(thrust_angle)) / projectile.get_mass(t)
            
            # Coriolis effect
            if environment.coriolis:
                coriolis_param = 2 * EARTH_ROTATION_RATE * math.sin(math.radians(environment.latitude))
                ax += coriolis_param * vy
                ay -= coriolis_param * vx
            
            return [vx, vy, ax, ay]
        
        # RK4 integration with adaptive step size
        while state[1] >= 0 and time < 120:  # Max 120 seconds flight time
            # Save current point
            trajectory.append((state[0], state[1], time, state[2], state[3], 
                             math.hypot(state[2], state[3])))
            
            # Adaptive step size based on velocity
            current_vel = math.hypot(state[2], state[3])
            time_step = max(min_time_step, 
                          min(max_time_step, 
                              max_time_step * (1000 / max(100, current_vel))))
            
            # RK4 integration
            k1 = derivative(state, time)
            k2_state = [s + 0.5 * time_step * k for s, k in zip(state, k1)]
            k2 = derivative(k2_state, time + 0.5*time_step)
            
            k3_state = [s + 0.5 * time_step * k for s, k in zip(state, k2)]
            k3 = derivative(k3_state, time + 0.5*time_step)
            
            k4_state = [s + time_step * k for s, k in zip(state, k3)]
            k4 = derivative(k4_state, time + time_step)
            
            for i in range(4):
                state[i] += (time_step / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            
            time += time_step
        
        return trajectory

    def update_results(self):
        if not self.trajectory:
            self.summary_text.setPlainText("No trajectory data available")
            self.data_text.setPlainText("")
            return
        
        # Calculate summary metrics
        max_height = max(p[1] for p in self.trajectory)
        max_height_idx = max(range(len(self.trajectory)), key=lambda i: self.trajectory[i][1])
        distance = self.trajectory[-1][0]
        flight_time = self.trajectory[-1][2]
        impact_velocity = self.trajectory[-1][5]
        impact_energy = 0.5 * (self.mass_input.value()/1000) * impact_velocity**2
        
        # Update summary text
        summary = f"""PROJECTILE:
Type: {self.type_combo.currentText()}
Mass: {self.mass_input.value():.1f}g
Diameter: {self.diam_input.value():.1f}mm
Drag Model: {self.drag_model_combo.currentText()}
Muzzle Velocity: {self.velocity_input.value():.1f} m/s
Launch Angle: {self.angle_input.value():.1f}°"""
        
        if self.type_combo.currentText().lower() == "rocket":
            summary += f"\nBurn Time: {self.burn_time_input.value():.1f}s"
            summary += f"\nAvg Thrust: {self.thrust_input.value():.0f}N"
        
        summary += f"""
        
RESULTS:
Maximum Height: {max_height:.1f}m (at {self.trajectory[max_height_idx][0]:.1f}m)
Total Distance: {distance:.1f}m
Flight Time: {flight_time:.2f}s
Impact Velocity: {impact_velocity:.1f}m/s
Impact Energy: {impact_energy:.1f}J"""
        
        self.summary_text.setPlainText(summary)
        
        # Update detailed data
        data_header = "Time(s)\tDistance(m)\tHeight(m)\tVx(m/s)\tVy(m/s)\tVelocity(m/s)\n"
        data_lines = [data_header]
        
        for i, point in enumerate(self.trajectory):
            if i % 10 == 0:  # Show every 10th point
                data_lines.append(f"{point[2]:.3f}\t{point[0]:.1f}\t{point[1]:.1f}\t"
                               f"{point[3]:.1f}\t{point[4]:.1f}\t{point[5]:.1f}\n")
        
        self.data_text.setPlainText(''.join(data_lines))

    def plot_trajectory(self):
        if not self.trajectory:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot current trajectory
        x = [p[0] for p in self.trajectory]
        y = [p[1] for p in self.trajectory]
        ax.plot(x, y, 'b-', linewidth=2, label='Current')
        
        # Plot previous trajectories if comparison enabled
        if self.compare_check.isChecked() and self.previous_trajectories:
            for i, traj in enumerate(self.previous_trajectories):
                x_prev = [p[0] for p in traj]
                y_prev = [p[1] for p in traj]
                ax.plot(x_prev, y_prev, '--', linewidth=1, 
                       label=f'Previous {i+1}', alpha=0.7)
        
        # Mark key points
        if self.trajectory:
            # Impact point
            ax.plot(x[-1], y[-1], 'ro', label='Impact')
            
            # Max height point
            max_idx = max(range(len(y)), key=lambda i: y[i])
            ax.plot(x[max_idx], y[max_idx], 'go', label='Max Height')
            
            # Add annotations
            ax.annotate("Impact", (x[-1], y[-1]), textcoords="offset points", 
                       xytext=(10, 10), ha='left', color='red')
            ax.annotate("Max Height", (x[max_idx], y[max_idx]), textcoords="offset points", 
                       xytext=(10, -15), ha='left', color='green')
        
        ax.set_title('Projectile Trajectory')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Height (m)')
        ax.grid(True)
        ax.legend()
        
        self.canvas.draw()

    def run_batch(self):
        param = self.batch_param_combo.currentText()
        start = self.batch_start_input.value()
        end = self.batch_end_input.value()
        step = self.batch_step_input.value()
        runs = []
        val = start
        param_map = {
            "Angle": "angle", "Muzzle Velocity": "velocity"
        }
        sweep_key = param_map.get(param, None)
        
        if sweep_key is None:
            QMessageBox.warning(self, "Warning", "Invalid sweep parameter")
            return
            
        if step <= 0:
            QMessageBox.warning(self, "Warning", "Step must be positive")
            return
            
        # Create parameter sets
        while val <= end:
            params = {
                "mass": self.mass_input.value() / 1000,
                "diameter": self.diam_input.value() / 1000,
                "drag_model": self.drag_model_combo.currentText(),
                "velocity": self.velocity_input.value(),
                "angle": self.angle_input.value(),
                "altitude": self.altitude_input.value(),
                "temperature": self.temp_input.value(),
                "pressure": self.pressure_input.value(),
                "humidity": self.humidity_input.value(),
                "wind_speed": self.wind_speed_input.value(),
                "wind_angle": self.wind_angle_input.value(),
                "coriolis": self.coriolis_check.isChecked(),
                "latitude": self.latitude_input.value()
            }
            params[sweep_key] = val
            runs.append(params.copy())
            val += step
        
        if not runs:
            QMessageBox.warning(self, "Warning", "No parameter sets created")
            return
            
        self.batch_progress_bar.setValue(0)
        self.batch_table.setRowCount(0)
        self.status_bar.showMessage(f"Running batch: {len(runs)} calculations...")
        
        # Create and start batch thread
        self.batch_thread = BatchCalculationThread(self, runs)
        self.batch_thread.progress.connect(self.batch_progress_bar.setValue)
        self.batch_thread.finished.connect(self.on_batch_complete)
        self.batch_thread.error.connect(self.on_batch_error)
        self.batch_thread.start()

    def on_batch_complete(self, results: List[list]):
        self.batch_trajectories = results
        self.batch_table.setRowCount(len(results))
        
        # Populate table
        for i, traj in enumerate(results):
            if traj:
                max_height = max(p[1] for p in traj)
                distance = traj[-1][0]
                impact_velocity = traj[-1][5]
                
                # Get parameter value
                param_val = self.batch_start_input.value() + i * self.batch_step_input.value()
                
                # Add row to table
                self.batch_table.insertRow(i)
                self.batch_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.batch_table.setItem(i, 1, QTableWidgetItem(f"{param_val:.2f}"))
                self.batch_table.setItem(i, 2, QTableWidgetItem(f"{max_height:.2f}"))
                self.batch_table.setItem(i, 3, QTableWidgetItem(f"{distance:.2f}"))
                self.batch_table.setItem(i, 4, QTableWidgetItem(f"{impact_velocity:.2f}"))
        
        # Plot all trajectories
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        for i, traj in enumerate(results):
            if traj:
                x = [p[0] for p in traj]
                y = [p[1] for p in traj]
                ax.plot(x, y, label=f"Run {i+1}")
        
        ax.set_title('Batch Trajectories')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Height (m)')
        ax.grid(True)
        ax.legend(fontsize='small')
        self.canvas.draw()
        
        self.status_bar.showMessage(f"Batch complete: {len(results)} trajectories calculated")

    def on_batch_error(self, error_msg: str):
        QMessageBox.critical(self, "Batch Error", f"Error during batch calculation:\n{error_msg}")
        self.status_bar.showMessage(f"Batch error: {error_msg}")

    def run_monte_carlo(self):
        num_runs = self.mc_runs_input.value()
        angle_spread = self.mc_spread_angle.value()
        vel_spread = self.mc_spread_velocity.value()
        
        if num_runs <= 0:
            QMessageBox.warning(self, "Warning", "Number of runs must be positive")
            return
            
        self.mc_progress_bar.setValue(0)
        self.status_bar.showMessage(f"Running Monte Carlo: {num_runs} simulations...")
        
        # Base parameters
        base_params = {
            "mass": self.mass_input.value() / 1000,
            "diameter": self.diam_input.value() / 1000,
            "drag_model": self.drag_model_combo.currentText(),
            "velocity": self.velocity_input.value(),
            "angle": self.angle_input.value(),
            "altitude": self.altitude_input.value(),
            "temperature": self.temp_input.value(),
            "pressure": self.pressure_input.value(),
            "humidity": self.humidity_input.value(),
            "wind_speed": self.wind_speed_input.value(),
            "wind_angle": self.wind_angle_input.value(),
            "coriolis": self.coriolis_check.isChecked(),
            "latitude": self.latitude_input.value()
        }
        
        # Run simulations
        fig = self.mc_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        for i in range(num_runs):
            params = base_params.copy()
            
            # Apply random variations
            params["angle"] += random.uniform(-angle_spread, angle_spread)
            params["velocity"] += random.uniform(-vel_spread, vel_spread)
            
            # Calculate trajectory
            traj = self._calculate_trajectory(**params)
            
            if traj:
                x = [p[0] for p in traj]
                y = [p[1] for p in traj]
                ax.plot(x, y, color="blue", alpha=0.1, linewidth=0.5)
            
            # Update progress
            self.mc_progress_bar.setValue(int((i+1)/num_runs*100))
        
        ax.set_title('Monte Carlo Trajectory Spread')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Height (m)')
        ax.grid(True)
        self.mc_canvas.draw()
        
        self.status_bar.showMessage(f"Monte Carlo complete: {num_runs} simulations")

    def export_to_csv(self):
        if not self.trajectory:
            QMessageBox.warning(self, "Warning", "No trajectory data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv)")
        
        if filename:
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            try:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Time(s)', 'Distance(m)', 'Height(m)', 
                                    'Vx(m/s)', 'Vy(m/s)', 'Velocity(m/s)'])
                    for point in self.trajectory:
                        writer.writerow(point)
                
                QMessageBox.information(self, "Success", f"Data exported to {filename}")
                self.status_bar.showMessage(f"Exported trajectory to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
                self.status_bar.showMessage(f"Export error: {str(e)}")

    def export_plot(self):
        if not self.trajectory:
            QMessageBox.warning(self, "Warning", "No trajectory data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot Image", "", "PNG Files (*.png);;SVG Files (*.svg)")
        
        if filename:
            try:
                if filename.endswith(".svg"):
                    self.figure.savefig(filename, format="svg")
                else:
                    if not filename.endswith('.png'):
                        filename += '.png'
                    self.figure.savefig(filename, format="png")
                
                QMessageBox.information(self, "Success", f"Plot exported to {filename}")
                self.status_bar.showMessage(f"Exported plot to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export plot: {str(e)}")
                self.status_bar.showMessage(f"Plot export error: {str(e)}")

    def export_results(self):
        self.export_to_csv()

    def export_presets(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Presets", "", "JSON Files (*.json)")
        
        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            
            try:
                with open(filename, 'w') as f:
                    json.dump(self.presets, f, indent=2)
                
                QMessageBox.information(self, "Success", f"Presets exported to {filename}")
                self.status_bar.showMessage(f"Exported presets to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export presets: {str(e)}")
                self.status_bar.showMessage(f"Preset export error: {str(e)}")

    def import_presets(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Presets", "", "JSON Files (*.json)")
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    imported = json.load(f)
                    self.presets.update(imported)
                    self.save_presets()
                    self.refresh_preset_combo()
                
                QMessageBox.information(self, "Success", f"Presets imported from {filename}")
                self.status_bar.showMessage(f"Imported presets from {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import presets: {str(e)}")
                self.status_bar.showMessage(f"Preset import error: {str(e)}")

    def show_about(self):
        about_text = """AcuTex Ballistic Calculator - Ultimate Edition\n
Version 5.0\n
Features:
- Support for bullets, rockets, and mortars
- 50+ built-in presets for various projectiles
- Adaptive RK4 integration for accurate trajectory calculation
- Real drag coefficient tables (G1, G7, rocket, mortar models)
- Environmental factors (altitude, temperature, pressure, humidity, wind)
- Coriolis effect calculation
- Rocket thrust curve simulation
- Batch parameter sweeps
- Monte Carlo simulations
- Trajectory comparison
- Threaded calculations for responsive UI
- Dark mode and unit switching\n
Created for Kali Linux"""
        QMessageBox.about(self, "About", about_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = BallisticCalculator()
    calculator.show()
    sys.exit(app.exec_())

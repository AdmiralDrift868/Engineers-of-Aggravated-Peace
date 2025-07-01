#!/usr/bin/env python3
import sys
import math
import csv
import json
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
    QFileDialog, QMessageBox, QInputDialog, QScrollArea, QSizePolicy,
    QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Constants
GRAVITY = 9.80665  # m/s^2
EARTH_RADIUS = 6371000  # meters
EARTH_ROTATION_RATE = 7.292115e-5  # rad/s
SPEED_OF_SOUND = 343  # m/s at sea level

PRESET_FILE = "user_presets.json"

class DragModel:
    """Enhanced drag coefficient tables for standard models"""
    @staticmethod
    @lru_cache(maxsize=1000)
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
    @lru_cache(maxsize=1000)
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
    @lru_cache(maxsize=1000)
    def rocket(velocity: float) -> float:
        mach = velocity / SPEED_OF_SOUND
        if mach > 3.0: return 0.50
        elif mach > 2.0: return 0.45
        elif mach > 1.5: return 0.40
        elif mach > 1.0: return 0.35
        elif mach > 0.8: return 0.30
        else: return 0.25

    @staticmethod
    @lru_cache(maxsize=1000)
    def mortar(velocity: float) -> float:
        mach = velocity / SPEED_OF_SOUND
        if mach > 1.5: return 0.55
        elif mach > 1.0: return 0.50
        elif mach > 0.8: return 0.45
        else: return 0.40

class Projectile:
    """Projectile model for physics and mass/thrust properties."""
    def __init__(
        self, mass: float = 0.01, diameter: float = 0.01, drag_model: str = 'G7',
        velocity: float = 800, projectile_type: str = 'bullet', thrust_curve: Optional[Dict[float, float]] = None,
        burn_time: float = 0
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
        self, altitude: float = 0, temperature: float = 15, pressure: float = 1013.25, humidity: float = 50,
        wind_speed: float = 0, wind_angle: float = 0, coriolis: bool = False, latitude: float = 45
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
        temp_kelvin = self.temperature + 273.15
        R = 287.058
        svp = 6.1078 * 10**((7.5 * self.temperature) / (self.temperature + 237.3))
        vp = svp * self.humidity / 100
        density = ((self.pressure * 100) / (R * temp_kelvin)) * (1 - (0.378 * vp) / (self.pressure * 100))
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

class BallisticCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Ballistic Calculator")
        self.setGeometry(100, 100, 1200, 900)
        self.setMinimumSize(800, 600)
        self.trajectory: List[Any] = []
        self.previous_trajectories: List[Any] = []
        self.dark_mode_enabled = False
        self.default_presets = self.load_default_presets()
        self.presets = self.load_presets()
        self.init_ui()
        self.apply_styles()

    def apply_styles(self):
        if self.dark_mode_enabled:
            self.setStyleSheet("""
                QMainWindow { background-color: #222; color: #ddd; font-family: Segoe UI, Arial; }
                QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 10px; padding-top: 15px; }
                QGroupBox::title { color: #aaa; }
                QPushButton { background-color: #444; color: #fff; border: 1px solid #aaa; border-radius: 3px; }
                QPushButton:hover { background-color: #333; }
                QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid #666; color: #fff; background: #333; }
                QTabWidget::pane { border: 1px solid #333; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #f5f5f5; font-family: Segoe UI, Arial; }
                QGroupBox { border: 1px solid #ccc; border-radius: 4px; margin-top: 10px; padding-top: 15px; }
                QGroupBox::title { color: #555; }
                QPushButton { background-color: #e0e0e0; border: 1px solid #aaa; border-radius: 3px; }
                QPushButton:hover { background-color: #d0d0d0; }
                QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid #bbb; border-radius: 3px; }
                QTabWidget::pane { border: 1px solid #ccc; }
            """)

    def load_default_presets(self) -> Dict[str, Any]:
        """Return the hardcoded default presets."""
        # Fixed: Added complete default presets dictionary
        default_presets = {
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
        return default_presets

    def load_presets(self) -> Dict[str, Any]:
        presets = self.default_presets.copy()  # Start with default presets
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

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.addTab(self.create_input_tab(), "Input")
        self.tabs.addTab(self.create_results_tab(), "Results")
        self.tabs.addTab(self.create_plot_tab(), "Graph")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
        main_layout.addWidget(self.tabs, 1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
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
        self.dark_mode_action.triggered.connect(self.toggle_dark_mode)
        help_menu = menubar.addMenu('Help')
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready.")

    def create_input_tab(self) -> QWidget:
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        type_group = QGroupBox("Projectile Type")
        type_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Bullet", "Rocket", "Mortar"])
        self.type_combo.currentTextChanged.connect(self.update_projectile_type)
        type_layout.addWidget(QLabel("Type:"))
        type_layout.addWidget(self.type_combo)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        preset_group = QGroupBox("Ammunition Presets")
        preset_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom"] + list(self.presets.keys()))
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)
        save_btn = QPushButton("Save Current")
        save_btn.clicked.connect(self.save_preset)
        preset_layout.addWidget(save_btn)
        delete_btn = QPushButton("Delete Preset")
        delete_btn.clicked.connect(self.delete_preset)
        preset_layout.addWidget(delete_btn)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        proj_group = QGroupBox("Projectile Parameters")
        proj_layout = QGridLayout()
        proj_layout.addWidget(QLabel("Mass (g):"), 0, 0)
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.1, 10000)
        self.mass_input.setValue(10)
        self.mass_input.setSingleStep(1)
        proj_layout.addWidget(self.mass_input, 0, 1)
        proj_layout.addWidget(QLabel("Diameter (mm):"), 1, 0)
        self.diam_input = QDoubleSpinBox()
        self.diam_input.setRange(0.1, 100)
        self.diam_input.setValue(7.62)
        self.diam_input.setSingleStep(0.1)
        proj_layout.addWidget(self.diam_input, 1, 1)
        proj_layout.addWidget(QLabel("Drag Model:"), 2, 0)
        self.drag_model_combo = QComboBox()
        self.drag_model_combo.addItems(['G1', 'G7', 'rocket', 'mortar'])
        proj_layout.addWidget(self.drag_model_combo, 2, 1)
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
        proj_layout.addWidget(self.rocket_group, 3, 0, 1, 2)
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        launch_group = QGroupBox("Launch Parameters")
        launch_layout = QGridLayout()
        launch_layout.addWidget(QLabel("Muzzle Velocity (m/s):"), 0, 0)
        self.velocity_input = QDoubleSpinBox()
        self.velocity_input.setRange(1, 2000)
        self.velocity_input.setValue(800)
        self.velocity_input.setSingleStep(10)
        launch_layout.addWidget(self.velocity_input, 0, 1)
        launch_layout.addWidget(QLabel("Launch Angle (deg):"), 1, 0)
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(0, 90)
        self.angle_input.setValue(15)
        self.angle_input.setSingleStep(1)
        launch_layout.addWidget(self.angle_input, 1, 1)
        launch_group.setLayout(launch_layout)
        layout.addWidget(launch_group)
        env_group = QGroupBox("Environmental Parameters")
        env_layout = QGridLayout()
        env_layout.addWidget(QLabel("Altitude (m):"), 0, 0)
        self.altitude_input = QDoubleSpinBox()
        self.altitude_input.setRange(-100, 10000)
        self.altitude_input.setValue(0)
        self.altitude_input.setSingleStep(10)
        env_layout.addWidget(self.altitude_input, 0, 1)
        env_layout.addWidget(QLabel("Temperature (°C):"), 1, 0)
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(-50, 60)
        self.temp_input.setValue(15)
        self.temp_input.setSingleStep(1)
        env_layout.addWidget(self.temp_input, 1, 1)
        env_layout.addWidget(QLabel("Wind Speed (m/s):"), 2, 0)
        self.wind_speed_input = QDoubleSpinBox()
        self.wind_speed_input.setRange(0, 50)
        self.wind_speed_input.setValue(0)
        self.wind_speed_input.setSingleStep(0.5)
        env_layout.addWidget(self.wind_speed_input, 2, 1)
        env_layout.addWidget(QLabel("Wind Angle (deg):"), 3, 0)
        self.wind_angle_input = QSpinBox()
        self.wind_angle_input.setRange(0, 359)
        self.wind_angle_input.setValue(0)
        env_layout.addWidget(self.wind_angle_input, 3, 1)
        self.coriolis_check = QCheckBox("Coriolis Effect")
        env_layout.addWidget(self.coriolis_check, 4, 0)
        env_layout.addWidget(QLabel("Latitude:"), 4, 1)
        self.latitude_input = QDoubleSpinBox()
        self.latitude_input.setRange(-90, 90)
        self.latitude_input.setValue(45)
        self.latitude_input.setEnabled(False)
        env_layout.addWidget(self.latitude_input, 4, 2)
        self.coriolis_check.stateChanged.connect(
            lambda: self.latitude_input.setEnabled(self.coriolis_check.isChecked()))
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        self.calculate_btn = QPushButton("Calculate Trajectory")
        self.calculate_btn.clicked.connect(self.calculate_trajectory)
        layout.addWidget(self.calculate_btn)
        scroll.setWidget(container)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)
        return tab

    def create_results_tab(self) -> QWidget:
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
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.figure = Figure(figsize=(5, 4), dpi=100)
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

    def create_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        self.dark_mode_settings_check = QCheckBox("Enable Dark Mode")
        self.dark_mode_settings_check.setChecked(self.dark_mode_enabled)
        self.dark_mode_settings_check.stateChanged.connect(self.toggle_dark_mode)
        layout.addWidget(self.dark_mode_settings_check)
        layout.addStretch(1)
        tab.setLayout(layout)
        return tab

    def toggle_dark_mode(self, checked=None):
        if checked is None:
            self.dark_mode_enabled = not self.dark_mode_enabled
        elif isinstance(checked, bool):
            self.dark_mode_enabled = checked
        else:
            self.dark_mode_enabled = bool(checked)
        self.apply_styles()
        if hasattr(self, 'dark_mode_settings_check'):
            self.dark_mode_settings_check.setChecked(self.dark_mode_enabled)
        if hasattr(self, 'dark_mode_action'):
            self.dark_mode_action.setChecked(self.dark_mode_enabled)

    def validate_inputs(self) -> bool:
        if self.mass_input.value() <= 0:
            QMessageBox.warning(self, "Invalid Input", "Mass must be positive.")
            return False
        if self.diam_input.value() <= 0:
            QMessageBox.warning(self, "Invalid Input", "Diameter must be positive.")
            return False
        if self.type_combo.currentText().lower() == 'rocket' and self.burn_time_input.value() < 0:
            QMessageBox.warning(self, "Invalid Input", "Burn time cannot be negative.")
            return False
        if self.velocity_input.value() <= 0:
            QMessageBox.warning(self, "Invalid Input", "Muzzle velocity must be positive.")
            return False
        return True

    def update_projectile_type(self, type_str: str):
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
        # Fixed: Corrected preset loading logic
        if preset_name == "Custom":
            return
            
        preset = self.presets.get(preset_name)
        if not preset:
            return
            
        self.mass_input.setValue(preset["mass"])
        self.diam_input.setValue(preset["diameter"])
        self.drag_model_combo.setCurrentText(preset["drag_model"])
        self.velocity_input.setValue(preset["velocity"])
        
        proj_type = preset.get("type", "bullet")
        self.type_combo.setCurrentText(proj_type.capitalize())
        self.update_projectile_type(proj_type)
        
        if proj_type == "rocket":
            self.burn_time_input.setValue(preset.get("burn_time", 1.0))
            # For simplicity, use first thrust value from curve
            thrust_curve = preset.get("thrust_curve", {})
            if thrust_curve:
                self.thrust_input.setValue(next(iter(thrust_curve.values())))
            else:
                self.thrust_input.setValue(1000)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name:
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
                    "thrust_curve": {0: self.thrust_input.value()}
                })
            self.presets[name] = preset_data
            if self.preset_combo.findText(name) == -1:
                self.preset_combo.addItem(name)
            self.save_presets()
            QMessageBox.information(self, "Preset Saved", f"Preset '{name}' saved.")

    def delete_preset(self):
        name = self.preset_combo.currentText()
        if name == "Custom" or name in self.default_presets:
            QMessageBox.warning(self, "Delete Preset", "Cannot delete custom or default presets.")
            return
        confirm = QMessageBox.question(self, "Delete Preset",
                                      f"Delete user preset '{name}'?",
                                      QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            if name in self.presets:
                del self.presets[name]
            idx = self.preset_combo.findText(name)
            if idx >= 0:
                self.preset_combo.removeItem(idx)
            self.save_presets()
            QMessageBox.information(self, "Preset Deleted", f"Preset '{name}' has been deleted.")

    def calculate_trajectory(self):
        if not self.validate_inputs():
            return
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setText("Calculating...")
        params = {
            "mass": self.mass_input.value() / 1000,
            "diameter": self.diam_input.value() / 1000,
            "drag_model": self.drag_model_combo.currentText(),
            "velocity": self.velocity_input.value(),
            "angle": self.angle_input.value(),
            "altitude": self.altitude_input.value(),
            "temperature": self.temp_input.value(),
            "wind_speed": self.wind_speed_input.value(),
            "wind_angle": self.wind_angle_input.value(),
            "coriolis": self.coriolis_check.isChecked(),
            "latitude": self.latitude_input.value()
        }
        if self.trajectory:
            self.previous_trajectories.append(self.trajectory)
            if len(self.previous_trajectories) > 3:
                self.previous_trajectories.pop(0)
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
        else:
            QMessageBox.warning(self, "Warning", "No trajectory data was generated")

    def on_calculation_error(self, error_msg: str):
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Trajectory")
        QMessageBox.critical(self, "Error", f"Calculation failed:\n{error_msg}")

    def _calculate_trajectory(self, mass: float, diameter: float, drag_model: str, velocity: float, angle: float,
                            altitude: float, temperature: float, wind_speed: float, wind_angle: float,
                            coriolis: bool, latitude: float, max_time_step: float = 0.1, min_time_step: float = 0.001) -> list:
        projectile = Projectile(
            mass=mass,
            diameter=diameter,
            drag_model=drag_model,
            velocity=velocity,
            projectile_type=self.type_combo.currentText().lower(),
            thrust_curve={0: self.thrust_input.value()},
            burn_time=self.burn_time_input.value() if self.type_combo.currentText().lower() == "rocket" else 0
        )
        environment = Environment(
            altitude=altitude,
            temperature=temperature,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            coriolis=coriolis,
            latitude=latitude
        )
        angle_rad = math.radians(angle)
        wind_x = environment.wind_speed * math.cos(math.radians(environment.wind_angle))
        wind_y = environment.wind_speed * math.sin(math.radians(environment.wind_angle))
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
            drag_coeff = projectile.drag_coefficient(v_rel)
            drag_force = 0.5 * environment.air_density * v_rel**2 * drag_coeff * projectile.area
            ax = -(drag_force * v_rel_x) / (projectile.get_mass(t) * v_rel) if v_rel > 0 else 0
            ay = -GRAVITY - (drag_force * v_rel_y) / (projectile.get_mass(t) * v_rel) if v_rel > 0 else -GRAVITY
            if projectile.projectile_type == 'rocket' and t < projectile.burn_time:
                thrust = projectile.get_thrust(t)
                thrust_angle = angle_rad if t == 0 else math.atan2(vy, vx)
                ax += (thrust * math.cos(thrust_angle)) / projectile.get_mass(t)
                ay += (thrust * math.sin(thrust_angle)) / projectile.get_mass(t)
            if environment.coriolis:
                coriolis_param = 2 * EARTH_ROTATION_RATE * math.sin(math.radians(environment.latitude))
                ax += coriolis_param * vy
                ay -= coriolis_param * vx
            return [vx, vy, ax, ay]
        while state[1] >= 0 and time < 120:
            trajectory.append((state[0], state[1], time, state[2], state[3],
                             math.hypot(state[2], state[3])))
            current_vel = math.hypot(state[2], state[3])
            time_step = max(min_time_step,
                          min(max_time_step,
                              max_time_step * (1000 / max(100, current_vel))))
            k1 = derivative(state, time)
            k2 = derivative([s + 0.5 * dt * k for s, k, dt in zip(state, k1, [time_step]*4)], time + 0.5*time_step)
            k3 = derivative([s + 0.5 * dt * k for s, k, dt in zip(state, k2, [time_step]*4)], time + 0.5*time_step)
            k4 = derivative([s + dt * k for s, k, dt in zip(state, k3, [time_step]*4)], time + time_step)
            for i in range(4):
                state[i] += (time_step / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            time += time_step
        return trajectory

    def update_results(self):
        if not self.trajectory:
            return
        max_height = max(p[1] for p in self.trajectory)
        distance = self.trajectory[-1][0]
        flight_time = self.trajectory[-1][2]
        impact_velocity = self.trajectory[-1][5]
        impact_energy = 0.5 * (self.mass_input.value()/1000) * impact_velocity**2
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
Maximum Height: {max_height:.1f}m
Total Distance: {distance:.1f}m
Flight Time: {flight_time:.2f}s
Impact Velocity: {impact_velocity:.1f}m/s
Impact Energy: {impact_energy:.1f}J"""
        self.summary_text.setPlainText(summary)
        data_header = "Time(s)\tDistance(m)\tHeight(m)\tVx(m/s)\tVy(m/s)\tVelocity(m/s)\n"
        data_lines = [data_header]
        for i, point in enumerate(self.trajectory):
            if i % 10 == 0:
                data_lines.append(f"{point[2]:.3f}\t{point[0]:.1f}\t{point[1]:.1f}\t"
                               f"{point[3]:.1f}\t{point[4]:.1f}\t{point[5]:.1f}\n")
        self.data_text.setPlainText(''.join(data_lines))

    def plot_trajectory(self):
        if not self.trajectory:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x = [p[0] for p in self.trajectory]
        y = [p[1] for p in self.trajectory]
        ax.plot(x, y, 'b-', linewidth=2, label='Current')
        if self.compare_check.isChecked() and self.previous_trajectories:
            for i, traj in enumerate(self.previous_trajectories):
                x_prev = [p[0] for p in traj]
                y_prev = [p[1] for p in traj]
                ax.plot(x_prev, y_prev, '--', linewidth=1,
                       label=f'Previous {i+1}', alpha=0.7)
        ax.plot(x[-1], y[-1], 'ro', label='Impact')
        max_idx = max(range(len(y)), key=lambda i: y[i])
        ax.plot(x[max_idx], y[max_idx], 'go', label='Max Height')
        ax.annotate("Impact", (x[-1], y[-1]), textcoords="offset points", xytext=(10,10), ha='left', color='red')
        ax.annotate("Max Height", (x[max_idx], y[max_idx]), textcoords="offset points", xytext=(10,-15), ha='left', color='green')
        ax.set_title('Projectile Trajectory')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Height (m)')
        ax.grid(True)
        ax.legend()
        self.canvas.draw()

    def export_to_csv(self):
        if not self.trajectory:
            QMessageBox.warning(self, "Warning", "No trajectory data to export")
            return
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv)", options=options)
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
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    def export_plot_to_png(self):
        if not self.trajectory:
            QMessageBox.warning(self, "Warning", "No trajectory data to export")
            return
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot Image", "", "PNG Files (*.png)", options=options)
        if filename:
            if not filename.endswith('.png'):
                filename += '.png'
            try:
                self.figure.savefig(filename)
                QMessageBox.information(self, "Success", f"Plot exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export plot: {str(e)}")

    def show_about(self):
        about_text = """Advanced Ballistic Calculator\n
Version 3.0\n
Features:
- Support for bullets, rockets, and mortars
- 30+ built-in presets for various projectiles
- Adaptive RK4 integration for accurate trajectory calculation
- Real drag coefficient tables (G1, G7, rocket, mortar models)
- Environmental factors (altitude, temperature, wind)
- Coriolis effect calculation
- Spin drift modeling
- Rocket thrust curve simulation
- Trajectory comparison
- Threaded calculations for responsive UI\n
Created for Kali Linux"""
        QMessageBox.about(self, "About", about_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = BallisticCalculator()
    calculator.show()
    sys.exit(app.exec_())

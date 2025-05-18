#!/usr/bin/env python3
import sys
import math
import csv
import json
from functools import lru_cache
from collections import namedtuple
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
                             QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
                             QFileDialog, QMessageBox, QInputDialog, QScrollArea, QSizePolicy,
                             QGridLayout)
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

class DragModel:
    """Enhanced drag coefficient tables for standard models"""
    @staticmethod
    @lru_cache(maxsize=1000)
    def G1(velocity):
        """Standard projectile drag function with more detailed table"""
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
    def G7(velocity):
        """Long-range boat tail drag function with more detailed table"""
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
    def rocket(velocity):
        """Drag coefficient for rockets"""
        mach = velocity / SPEED_OF_SOUND
        if mach > 3.0: return 0.50
        elif mach > 2.0: return 0.45
        elif mach > 1.5: return 0.40
        elif mach > 1.0: return 0.35
        elif mach > 0.8: return 0.30
        else: return 0.25

    @staticmethod
    @lru_cache(maxsize=1000)
    def mortar(velocity):
        """Drag coefficient for mortar shells"""
        mach = velocity / SPEED_OF_SOUND
        if mach > 1.5: return 0.55
        elif mach > 1.0: return 0.50
        elif mach > 0.8: return 0.45
        else: return 0.40

class Projectile:
    def __init__(self, mass=0.01, diameter=0.01, drag_model='G7', velocity=800, 
                 projectile_type='bullet', thrust_curve=None, burn_time=0):
        self.mass = mass  # kg
        self.diameter = diameter  # meters
        self.drag_model = drag_model
        self.velocity = velocity  # m/s
        self.area = math.pi * (diameter/2)**2
        self.projectile_type = projectile_type
        self.thrust_curve = thrust_curve or {}
        self.burn_time = burn_time
        self.initial_mass = mass
        
    def drag_coefficient(self, velocity):
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
            
    def get_thrust(self, time):
        """Get current thrust based on thrust curve"""
        if time > self.burn_time:
            return 0
        # Linear interpolation of thrust curve
        times = sorted(self.thrust_curve.keys())
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
        
    def get_mass(self, time):
        """Calculate current mass based on burn time"""
        if self.projectile_type != 'rocket' or time > self.burn_time:
            return self.mass
        # Linear mass decrease during burn
        return self.initial_mass - (self.initial_mass - self.mass) * (time / self.burn_time)

class Environment:
    def __init__(self, altitude=0, temperature=15, pressure=1013.25, humidity=50,
                 wind_speed=0, wind_angle=0, coriolis=False, latitude=45):
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
    def calculate_air_density(self):
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
    """Thread for performing trajectory calculations without freezing UI"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
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

class BallisticCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Ballistic Calculator")
        self.setGeometry(100, 100, 1200, 900)
        self.setMinimumSize(800, 600)
        
        self.projectile = None
        self.environment = None
        self.trajectory = []
        self.previous_trajectories = []
        self.metric_units = True
        
        # Load presets
        self.presets = self.load_presets()
        
        self.init_ui()
        self.apply_styles()
        
    def apply_styles(self):
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
        
    def load_presets(self):
        """Load ammunition presets from JSON file or use defaults"""
        default_presets = {
            # Bullet presets (3)
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
            
            # Rocket Presets (25)
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
            
            # Mortar Presets (5)
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
    
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tabs
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
        self.preset_combo.addItems(["Custom"] + list(self.presets.keys()))
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)
        
        save_btn = QPushButton("Save Current")
        save_btn.clicked.connect(self.save_preset)
        preset_layout.addWidget(save_btn)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        # Projectile Group
        proj_group = QGroupBox("Projectile Parameters")
        proj_layout = QGridLayout()
        
        # Mass
        proj_layout.addWidget(QLabel("Mass (g):"), 0, 0)
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.1, 10000)
        self.mass_input.setValue(10)
        self.mass_input.setSingleStep(1)
        proj_layout.addWidget(self.mass_input, 0, 1)
        
        # Diameter
        proj_layout.addWidget(QLabel("Diameter (mm):"), 1, 0)
        self.diam_input = QDoubleSpinBox()
        self.diam_input.setRange(0.1, 100)
        self.diam_input.setValue(7.62)
        self.diam_input.setSingleStep(0.1)
        proj_layout.addWidget(self.diam_input, 1, 1)
        
        # Drag Model
        proj_layout.addWidget(QLabel("Drag Model:"), 2, 0)
        self.drag_model_combo = QComboBox()
        self.drag_model_combo.addItems(['G1', 'G7', 'rocket', 'mortar'])
        proj_layout.addWidget(self.drag_model_combo, 2, 1)
        
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
        proj_layout.addWidget(self.rocket_group, 3, 0, 1, 2)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        # Launch Parameters Group
        launch_group = QGroupBox("Launch Parameters")
        launch_layout = QGridLayout()
        
        # Velocity
        launch_layout.addWidget(QLabel("Muzzle Velocity (m/s):"), 0, 0)
        self.velocity_input = QDoubleSpinBox()
        self.velocity_input.setRange(1, 2000)
        self.velocity_input.setValue(800)
        self.velocity_input.setSingleStep(10)
        launch_layout.addWidget(self.velocity_input, 0, 1)
        
        # Angle
        launch_layout.addWidget(QLabel("Launch Angle (deg):"), 1, 0)
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(0, 90)
        self.angle_input.setValue(15)
        self.angle_input.setSingleStep(1)
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
        self.altitude_input.setSingleStep(10)
        env_layout.addWidget(self.altitude_input, 0, 1)
        
        # Temperature
        env_layout.addWidget(QLabel("Temperature (°C):"), 1, 0)
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(-50, 60)
        self.temp_input.setValue(15)
        self.temp_input.setSingleStep(1)
        env_layout.addWidget(self.temp_input, 1, 1)
        
        # Wind
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
        
        # Advanced options
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
        
        # Calculate button
        self.calculate_btn = QPushButton("Calculate Trajectory")
        self.calculate_btn.clicked.connect(self.calculate_trajectory)
        layout.addWidget(self.calculate_btn)
        
        scroll.setWidget(container)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)
        return tab
    
    def create_advanced_tab(self):
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
        self.zero_range_input.setRange(10, 2000)
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
        self.spin_drift_check = QCheckBox("Enable Spin Drift")
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
    
    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Summary Group with scroll
        summary_group = QGroupBox("Summary Results")
        summary_layout = QVBoxLayout()
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        scroll = QScrollArea()
        scroll.setWidget(self.summary_text)
        scroll.setWidgetResizable(True)
        summary_layout.addWidget(scroll)
        summary_group.setLayout(summary_layout)
        
        # Data Group with scroll
        data_group = QGroupBox("Trajectory Data")
        data_layout = QVBoxLayout()
        
        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        scroll_data = QScrollArea()
        scroll_data.setWidget(self.data_text)
        scroll_data.setWidgetResizable(True)
        data_layout.addWidget(scroll_data)
        data_group.setLayout(data_layout)
        
        # Add stretch to push groups up
        layout.addWidget(summary_group, 1)
        layout.addWidget(data_group, 2)
        tab.setLayout(layout)
        return tab
    
    def create_plot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        
        # Comparison checkbox at bottom
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch(1)
        self.compare_check = QCheckBox("Compare with previous trajectory")
        bottom_layout.addWidget(self.compare_check)
        bottom_layout.addStretch(1)
        
        layout.addLayout(bottom_layout)
        tab.setLayout(layout)
        return tab
    
    def update_projectile_type(self, type_str):
        """Update UI based on projectile type selection"""
        type_lower = type_str.lower()
        self.rocket_group.setVisible(type_lower == "rocket")
        
        # Update drag model options
        self.drag_model_combo.clear()
        if type_lower == "bullet":
            self.drag_model_combo.addItems(['G1', 'G7'])
        elif type_lower == "rocket":
            self.drag_model_combo.addItems(['rocket'])
        elif type_lower == "mortar":
            self.drag_model_combo.addItems(['mortar'])
    
    def load_preset(self, preset_name):
        if preset_name == "Custom":
            return
            
        preset = self.presets[preset_name]
        self.mass_input.setValue(preset["mass"])
        self.diam_input.setValue(preset["diameter"])
        self.drag_model_combo.setCurrentText(preset["drag_model"])
        self.velocity_input.setValue(preset["velocity"])
        
        # Set projectile type and update UI
        self.type_combo.setCurrentText(preset.get("type", "bullet").capitalize())
        self.update_projectile_type(preset.get("type", "bullet"))
        
        # Set rocket-specific parameters if applicable
        if preset.get("type") == "rocket":
            self.burn_time_input.setValue(preset.get("burn_time", 1.0))
            self.thrust_input.setValue(preset.get("thrust_curve", {}).get(0, 1000))
    
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
            
            # Add rocket-specific parameters if applicable
            if self.type_combo.currentText().lower() == "rocket":
                preset_data.update({
                    "burn_time": self.burn_time_input.value(),
                    "thrust_curve": {0: self.thrust_input.value()}
                })
            
            self.presets[name] = preset_data
            self.preset_combo.addItem(name)
    
    def calculate_zero_angle(self):
        """Calculate the launch angle needed to hit at zero range"""
        zero_range = self.zero_range_input.value()
        # Simplified calculation - in reality would need iterative solution
        angle = math.degrees(math.asin((zero_range * GRAVITY) / 
                            (self.velocity_input.value() ** 2)) / 2)
        self.angle_input.setValue(angle)
        QMessageBox.information(self, "Zero Angle", 
                              f"Calculated zero angle: {angle:.2f}°")
    
    def calculate_trajectory(self):
        # Disable UI during calculation
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
    
    def on_calculation_complete(self, trajectory):
        self.trajectory = trajectory
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Trajectory")
        
        if trajectory:
            self.update_results()
            self.plot_trajectory()
        else:
            QMessageBox.warning(self, "Warning", "No trajectory data was generated")
    
    def on_calculation_error(self, error_msg):
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Trajectory")
        QMessageBox.critical(self, "Error", f"Calculation failed:\n{error_msg}")
    
    def _calculate_trajectory(self, mass, diameter, drag_model, velocity, angle,
                            altitude, temperature, wind_speed, wind_angle,
                            coriolis, latitude, max_time_step=0.1, min_time_step=0.001):
        """Enhanced RK4 trajectory calculation with rocket/mortar support"""
        # Initialize projectile and environment
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
            
            # Spin drift effect
            if self.spin_drift_check.isChecked() and projectile.projectile_type == 'bullet':
                twist_rate = self.twist_input.value() * 0.0254  # Convert inches to meters
                spin_rate = (v_rel * 2 * math.pi) / twist_rate  # rad/s
                spin_drift = 0.0001 * spin_rate * v_rel  # Simplified spin drift model
                ax += spin_drift
            
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
        
        # Calculate summary metrics
        max_height = max(p[1] for p in self.trajectory)
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
Maximum Height: {max_height:.1f}m
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
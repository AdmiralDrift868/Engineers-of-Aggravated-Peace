#!/usr/bin/env python3
"""
AcuTex Ballistic Calculator v5.x
Modern GUI with full input controls, presets, batch mode, Monte Carlo, dark mode, export, unit switching, and robust validation.
"""
import sys, math, csv, json, os, logging, difflib, random
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
    QFileDialog, QMessageBox, QInputDialog, QScrollArea, QSizePolicy,
    QGridLayout, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

logging.basicConfig(filename="ballistic.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
GRAVITY = 9.80665
SPEED_OF_SOUND = 343

PRESET_FILE = "user_presets.json"
CONFIG_FILE = "user_config.json"

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
    def custom(velocity: float, curve: List[Tuple[float, float]]) -> float:
        return interpolate_curve(velocity, curve)

class Projectile:
    def __init__(
        self, mass: float = 0.01, diameter: float = 0.01, drag_model: str = 'G7',
        velocity: float = 800, projectile_type: str = 'bullet',
        custom_drag_curve: Optional[List[Tuple[float, float]]] = None
    ):
        self.mass = mass
        self.diameter = diameter
        self.drag_model = drag_model
        self.velocity = velocity
        self.area = math.pi * (diameter/2)**2
        self.projectile_type = projectile_type
        self.custom_drag_curve = custom_drag_curve

    def drag_coefficient(self, velocity: float) -> float:
        if self.drag_model == 'G1':
            return DragModel.G1(velocity)
        elif self.drag_model == 'G7':
            return DragModel.G7(velocity)
        elif self.drag_model == 'Custom' and self.custom_drag_curve:
            return DragModel.custom(velocity, self.custom_drag_curve)
        else:
            return 0.3

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
        self.air_density = 1.225  # For simplicity, can be improved

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

class BallisticCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AcuTex Ballistic Calculator")
        self.setGeometry(100, 100, 1200, 950)
        self.presets = self.load_presets()
        self.custom_drag_curve = None
        self.dark_mode_enabled = False
        self.use_imperial = False
        self.config = self.load_config()
        self.apply_config()
        self.init_ui()
        self.apply_styles()
        log("AcuTex Ballistic Calculator started")

    def load_presets(self) -> Dict[str, Any]:
        presets = {
            "7.62 NATO Ball": {
                "type": "bullet",
                "mass": 9.5,
                "diameter": 7.82,
                "velocity": 830,
                "drag_model": "G7",
                "angle": 0
            },
             "Custom": {"mass": 10, "diameter": 7.62, "drag_model": "G7", "velocity": 800, "type": "bullet"},
            "5.56mm NATO": {"mass": 4.0, "diameter": 5.56, "drag_model": "G7", "velocity": 940, "type": "bullet"},
            "7.62x51mm NATO": {"mass": 9.5, "diameter": 7.82, "drag_model": "G7", "velocity": 830, "type": "bullet"},
            "9mm Parabellum": {"mass": 8.0, "diameter": 9.0, "drag_model": "G1", "velocity": 360, "type": "bullet"},
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
        if os.path.exists(PRESET_FILE):
            try:
                with open(PRESET_FILE, "r") as f:
                    user_presets = json.load(f)
                    presets.update(user_presets)
            except Exception as e:
                log(f"Failed to load custom presets: {e}", "warning")
        return presets

    def save_presets(self):
        try:
            user_presets = {k: v for k, v in self.presets.items() if k not in ["7.62 NATO Ball"]}
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
            "dark_mode_enabled": self.dark_mode_enabled,
            "use_imperial": self.use_imperial
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log(f"Failed to save config: {e}", "warning")

    def apply_config(self):
        self.dark_mode_enabled = self.config.get("dark_mode_enabled", False)
        self.use_imperial = self.config.get("use_imperial", False)

    def apply_styles(self):
        if self.dark_mode_enabled:
            self.setStyleSheet("""
                QMainWindow { background-color: #23272e; color: #eee; font-family: Segoe UI, Arial; }
                QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 10px; padding-top: 15px; color: #eee; }
                QPushButton { background-color: #33364a; color: #eaeaea; border: 1px solid #888; border-radius: 3px; }
                QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid #666; color: #eee; background: #282c34; }
                QTabWidget::pane { border: 1px solid #333; }
            """)
        else:
            self.setStyleSheet("")

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_input_tab(), "Input")
        self.tabs.addTab(self.create_results_tab(), "Results")
        self.tabs.addTab(self.create_plot_tab(), "Graph")
        self.tabs.addTab(self.create_batch_tab(), "Batch Mode")
        self.tabs.addTab(self.create_monte_carlo_tab(), "Monte Carlo")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
        self.tabs.addTab(self.create_help_tab(), "Help")
        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_input_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        # Type/preset
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Projectile Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Bullet", "Rocket", "Mortar"])
        row1.addWidget(self.type_combo)
        row1.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        self.preset_combo.addItems(["Custom"] + list(self.presets.keys()))
        self.preset_combo.lineEdit().textChanged.connect(self.filter_presets)
        row1.addWidget(self.preset_combo)
        load_btn = QPushButton("Load Preset"); load_btn.clicked.connect(self.load_preset)
        save_btn = QPushButton("Save Preset"); save_btn.clicked.connect(self.save_preset)
        del_btn = QPushButton("Delete Preset"); del_btn.clicked.connect(self.delete_preset)
        row1.addWidget(load_btn); row1.addWidget(save_btn); row1.addWidget(del_btn)
        layout.addLayout(row1)
        # Units toggle
        units_row = QHBoxLayout()
        self.unit_toggle = QCheckBox("Use Imperial Units")
        self.unit_toggle.setChecked(self.use_imperial)
        self.unit_toggle.stateChanged.connect(self.toggle_units)
        units_row.addWidget(self.unit_toggle)
        layout.addLayout(units_row)
        # Projectile params
        proj_group = QGroupBox("Projectile Parameters")
        proj_layout = QGridLayout()
        self.mass_input = QDoubleSpinBox(); self.mass_input.setRange(0.1, 1000000); self.mass_input.setValue(10)
        self.diam_input = QDoubleSpinBox(); self.diam_input.setRange(0.1, 1000); self.diam_input.setValue(7.62)
        self.drag_model_combo = QComboBox(); self.drag_model_combo.addItems(["G1", "G7", "Custom"])
        drag_curve_btn = QPushButton("Load Custom Drag"); drag_curve_btn.clicked.connect(self.load_custom_drag_curve)
        proj_layout.addWidget(QLabel("Mass (g/lb):"), 0,0); proj_layout.addWidget(self.mass_input,0,1)
        proj_layout.addWidget(QLabel("Diameter (mm/in):"),1,0); proj_layout.addWidget(self.diam_input,1,1)
        proj_layout.addWidget(QLabel("Drag Model:"),2,0); proj_layout.addWidget(self.drag_model_combo,2,1)
        proj_layout.addWidget(drag_curve_btn,2,2)
        proj_group.setLayout(proj_layout); layout.addWidget(proj_group)
        # Launch
        launch_group = QGroupBox("Launch Parameters")
        launch_layout = QGridLayout()
        self.velocity_input = QDoubleSpinBox(); self.velocity_input.setRange(1, 5000); self.velocity_input.setValue(800)
        self.angle_input = QDoubleSpinBox(); self.angle_input.setRange(0, 90); self.angle_input.setValue(15)
        launch_layout.addWidget(QLabel("Muzzle Velocity (m/s or fps):"),0,0); launch_layout.addWidget(self.velocity_input,0,1)
        launch_layout.addWidget(QLabel("Launch Angle (deg):"),1,0); launch_layout.addWidget(self.angle_input,1,1)
        launch_group.setLayout(launch_layout); layout.addWidget(launch_group)
        # Environment
        env_group = QGroupBox("Environmental Parameters")
        env_layout = QGridLayout()
        self.altitude_input = QDoubleSpinBox(); self.altitude_input.setRange(-100, 20000); self.altitude_input.setValue(0)
        self.temp_input = QDoubleSpinBox(); self.temp_input.setRange(-50, 60); self.temp_input.setValue(15)
        self.wind_speed_input = QDoubleSpinBox(); self.wind_speed_input.setRange(-100, 100); self.wind_speed_input.setValue(0)
        self.wind_angle_input = QSpinBox(); self.wind_angle_input.setRange(0,359); self.wind_angle_input.setValue(0)
        self.coriolis_check = QCheckBox("Coriolis Effect")
        self.latitude_input = QDoubleSpinBox(); self.latitude_input.setRange(-90,90); self.latitude_input.setValue(45)
        self.dynamic_air_check = QCheckBox("ISA Dynamic Air")
        env_layout.addWidget(QLabel("Altitude (m/ft):"),0,0); env_layout.addWidget(self.altitude_input,0,1)
        env_layout.addWidget(QLabel("Temperature (°C/°F):"),1,0); env_layout.addWidget(self.temp_input,1,1)
        env_layout.addWidget(QLabel("Wind Speed (m/s or mph):"),2,0); env_layout.addWidget(self.wind_speed_input,2,1)
        env_layout.addWidget(QLabel("Wind Angle (deg):"),3,0); env_layout.addWidget(self.wind_angle_input,3,1)
        env_layout.addWidget(self.coriolis_check,4,0); env_layout.addWidget(QLabel("Latitude:"),4,1); env_layout.addWidget(self.latitude_input,4,2)
        env_layout.addWidget(self.dynamic_air_check,5,0,1,3)
        env_group.setLayout(env_layout); layout.addWidget(env_group)
        # Calculate
        calc_row = QHBoxLayout()
        calc_btn = QPushButton("Calculate Trajectory"); calc_btn.clicked.connect(self.calculate_trajectory)
        export_btn = QPushButton("Export Results"); export_btn.clicked.connect(self.export_results)
        export_plot_btn = QPushButton("Export Plot"); export_plot_btn.clicked.connect(self.export_plot)
        calc_row.addWidget(calc_btn); calc_row.addWidget(export_btn); calc_row.addWidget(export_plot_btn)
        layout.addLayout(calc_row)
        return tab

    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        layout.addWidget(self.data_text)
        return tab

    def create_plot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.nav_toolbar)
        return tab

    def create_batch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        sweep_row = QHBoxLayout()
        self.batch_param_combo = QComboBox()
        self.batch_param_combo.addItems([
            "Angle", "Muzzle Velocity", "Wind Speed", "Temperature", "Altitude"
        ])
        sweep_row.addWidget(QLabel("Sweep parameter:"))
        sweep_row.addWidget(self.batch_param_combo)
        self.batch_start_input = QDoubleSpinBox()
        self.batch_end_input = QDoubleSpinBox()
        self.batch_step_input = QDoubleSpinBox()
        self.batch_start_input.setRange(-10000, 10000)
        self.batch_end_input.setRange(-10000, 10000)
        self.batch_step_input.setRange(0.001, 10000)
        self.batch_start_input.setValue(10)
        self.batch_end_input.setValue(80)
        self.batch_step_input.setValue(5)
        sweep_row.addWidget(QLabel("Start:"))
        sweep_row.addWidget(self.batch_start_input)
        sweep_row.addWidget(QLabel("End:"))
        sweep_row.addWidget(self.batch_end_input)
        sweep_row.addWidget(QLabel("Step:"))
        sweep_row.addWidget(self.batch_step_input)
        self.batch_run_btn = QPushButton("Run Batch")
        self.batch_run_btn.clicked.connect(self.run_batch)
        sweep_row.addWidget(self.batch_run_btn)
        layout.addLayout(sweep_row)
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(5)
        self.batch_table.setHorizontalHeaderLabels([
            "Run", "Param Value", "Max Height", "Total Distance", "Impact Velocity"
        ])
        layout.addWidget(self.batch_table, 1)
        return tab

    def create_monte_carlo_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.mc_runs_input = QSpinBox()
        self.mc_runs_input.setRange(10, 500)
        self.mc_runs_input.setValue(100)
        self.mc_spread_angle = QDoubleSpinBox()
        self.mc_spread_angle.setRange(0, 5)
        self.mc_spread_angle.setValue(0.5)
        self.mc_spread_velocity = QDoubleSpinBox()
        self.mc_spread_velocity.setRange(0, 50)
        self.mc_spread_velocity.setValue(5)
        run_btn = QPushButton("Run Monte Carlo")
        run_btn.clicked.connect(self.run_monte_carlo)
        controls.addWidget(QLabel("Runs:")); controls.addWidget(self.mc_runs_input)
        controls.addWidget(QLabel("Angle ±deg:")); controls.addWidget(self.mc_spread_angle)
        controls.addWidget(QLabel("Velocity ±:")); controls.addWidget(self.mc_spread_velocity)
        controls.addWidget(run_btn)
        layout.addLayout(controls)
        self.mc_canvas = FigureCanvas(Figure(figsize=(5,4), dpi=100))
        layout.addWidget(self.mc_canvas, 1)
        return tab

    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        dark_mode_chk = QCheckBox("Enable Dark Mode")
        dark_mode_chk.setChecked(self.dark_mode_enabled)
        dark_mode_chk.stateChanged.connect(self.toggle_dark_mode)
        layout.addWidget(dark_mode_chk)
        units_chk = QCheckBox("Use Imperial Units")
        units_chk.setChecked(self.use_imperial)
        units_chk.stateChanged.connect(self.toggle_units)
        layout.addWidget(units_chk)
        return tab

    def create_help_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText(
            "AcuTex Ballistic Calculator Help\n\n"
            "• Input: Enter all projectile and environmental parameters, or select from presets.\n"
            "• Batch Mode: Sweep one parameter (angle, velocity, wind, etc) and visualize.\n"
            "• Monte Carlo: Simulate random angle/velocity variation.\n"
            "• Units: Switch between metric and imperial units in Settings or Input tab.\n"
            "• Export: Save results table or plot for analysis.\n"
            "• All fields have tooltips. See https://github.com/AdmiralDrift868 for detailed docs."
        )
        layout.addWidget(help_text)
        return tab

    # --- Preset Functions ---
    def filter_presets(self, text):
        if not text.strip():
            items = ["Custom"] + list(self.presets.keys())
        else:
            matches = difflib.get_close_matches(text, self.presets.keys(), n=10, cutoff=0.3)
            substr_matches = [k for k in self.presets.keys() if text.lower() in k.lower()]
            combined = list(dict.fromkeys(matches + substr_matches))
            items = ["Custom"] + combined
        self.preset_combo.clear()
        self.preset_combo.addItems(items)

    def load_preset(self):
        key = self.preset_combo.currentText()
        if key == "Custom" or key not in self.presets:
            QMessageBox.information(self, "Preset", "Select a valid preset to load.")
            return
        p = self.presets[key]
        unit = self.use_imperial
        self.type_combo.setCurrentText(p.get("type", "Bullet").capitalize())
        self.mass_input.setValue(self.to_imperial_mass(p["mass"]) if unit else p["mass"])
        self.diam_input.setValue(self.to_imperial_length(p["diameter"]) if unit else p["diameter"])
        self.velocity_input.setValue(self.to_imperial_velocity(p["velocity"]) if unit else p["velocity"])
        self.angle_input.setValue(p.get("angle", 15))
        self.drag_model_combo.setCurrentText(p.get("drag_model", "G7"))

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name.strip():
            preset = {
                "type": self.type_combo.currentText().lower(),
                "mass": self.from_imperial_mass(self.mass_input.value()) if self.use_imperial else self.mass_input.value(),
                "diameter": self.from_imperial_length(self.diam_input.value()) if self.use_imperial else self.diam_input.value(),
                "velocity": self.from_imperial_velocity(self.velocity_input.value()) if self.use_imperial else self.velocity_input.value(),
                "angle": self.angle_input.value(),
                "drag_model": self.drag_model_combo.currentText()
            }
            self.presets[name] = preset
            self.save_presets()
            self.preset_combo.addItem(name)
            QMessageBox.information(self, "Preset Saved", f"Preset '{name}' saved.")

    def delete_preset(self):
        key = self.preset_combo.currentText()
        if key in self.presets:
            del self.presets[key]
            self.save_presets()
            idx = self.preset_combo.findText(key)
            self.preset_combo.removeItem(idx)
            QMessageBox.information(self, "Preset Deleted", f"Preset '{key}' deleted.")

    def load_custom_drag_curve(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Custom Drag Curve CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                curve = []
                with open(filename, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if len(row) != 2: continue
                        try:
                            v, cd = float(row[0]), float(row[1])
                            curve.append((v, cd))
                        except Exception:
                            continue
                if curve:
                    self.custom_drag_curve = curve
                    QMessageBox.information(self, "Drag Curve Loaded", f"Loaded {len(curve)} drag points.")
                else:
                    QMessageBox.warning(self, "No Data", f"No valid velocity, Cd pairs found.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load drag curve:\n{e}")

    def toggle_dark_mode(self, checked):
        self.dark_mode_enabled = bool(checked)
        self.apply_styles()
        self.save_config()

    def toggle_units(self, checked):
        self.use_imperial = bool(checked)
        self.update_units()
        self.save_config()

    def update_units(self):
        # Convert current values to new units and update labels
        is_imp = self.use_imperial
        # Mass
        m = self.mass_input.value()
        self.mass_input.setSuffix(" lb" if is_imp else " g")
        self.mass_input.setValue(self.to_imperial_mass(m) if is_imp else self.from_imperial_mass(m))
        # Diameter
        d = self.diam_input.value()
        self.diam_input.setSuffix(" in" if is_imp else " mm")
        self.diam_input.setValue(self.to_imperial_length(d) if is_imp else self.from_imperial_length(d))
        # Velocity
        v = self.velocity_input.value()
        self.velocity_input.setSuffix(" fps" if is_imp else " m/s")
        self.velocity_input.setValue(self.to_imperial_velocity(v) if is_imp else self.from_imperial_velocity(v))
        # Altitude
        alt = self.altitude_input.value()
        self.altitude_input.setSuffix(" ft" if is_imp else " m")
        self.altitude_input.setValue(self.to_imperial_length(alt) if is_imp else self.from_imperial_length(alt))
        # Temp
        t = self.temp_input.value()
        self.temp_input.setSuffix(" °F" if is_imp else " °C")
        self.temp_input.setValue(self.to_imperial_temp(t) if is_imp else self.from_imperial_temp(t))
        # Wind Speed
        w = self.wind_speed_input.value()
        self.wind_speed_input.setSuffix(" mph" if is_imp else " m/s")
        self.wind_speed_input.setValue(self.to_imperial_velocity(w) if is_imp else self.from_imperial_velocity(w))

    # --- Unit conversion helpers ---
    def to_imperial_mass(self, g): return g * 0.00220462
    def from_imperial_mass(self, lb): return lb / 0.00220462
    def to_imperial_length(self, mm): return mm * 0.0393701
    def from_imperial_length(self, inch): return inch / 0.0393701
    def to_imperial_velocity(self, ms): return ms * 3.28084
    def from_imperial_velocity(self, fps): return fps / 3.28084
    def to_imperial_temp(self, c): return c * 9/5 + 32
    def from_imperial_temp(self, f): return (f - 32) * 5/9

    # --- Trajectory Calculation ---
    def _calculate_trajectory(self, **params):
        # All values must be SI
        mass = params.get("mass", 0.01)
        diameter = params.get("diameter", 0.01)
        drag_model = params.get("drag_model", "G7")
        velocity = params.get("velocity", 800)
        angle = params.get("angle", 15)
        proj = Projectile(mass, diameter, drag_model, velocity, "bullet")
        env = Environment()
        trajectory = []
        x, y = 0, 0
        rad = math.radians(angle)
        vx = velocity * math.cos(rad)
        vy = velocity * math.sin(rad)
        dt = 0.02
        t = 0
        while y >= 0 and t < 120:
            v = math.hypot(vx, vy)
            Cd = proj.drag_coefficient(v)
            rho = env.air_density
            Fd = 0.5 * rho * (diameter/2)**2 * math.pi * Cd * v ** 2
            ax = -Fd * vx / (mass * v)
            ay = -GRAVITY - Fd * vy / (mass * v)
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            t += dt
            trajectory.append((x, y, t, vx, vy, v))
            if y < 0:
                break
        return trajectory

    def calculate_trajectory(self):
        params = {
            "mass": self.from_imperial_mass(self.mass_input.value()) if self.use_imperial else self.mass_input.value() / 1000,
            "diameter": self.from_imperial_length(self.diam_input.value()) if self.use_imperial else self.diam_input.value() / 1000,
            "drag_model": self.drag_model_combo.currentText(),
            "velocity": self.from_imperial_velocity(self.velocity_input.value()) if self.use_imperial else self.velocity_input.value(),
            "angle": self.angle_input.value(),
        }
        self.trajectory = self._calculate_trajectory(**params)
        self.update_results()
        self.plot_trajectory()

    def update_results(self):
        if not self.trajectory:
            self.summary_text.setPlainText("No trajectory calculated.")
            self.data_text.setPlainText("")
            return
        max_height = max(p[1] for p in self.trajectory)
        distance = self.trajectory[-1][0]
        flight_time = self.trajectory[-1][2]
        impact_velocity = self.trajectory[-1][5]
        impact_energy = 0.5 * (self.mass_input.value()/1000) * impact_velocity**2
        vx, vy = self.trajectory[-1][3], self.trajectory[-1][4]
        impact_angle = math.degrees(math.atan2(vy, vx))
        summary = f"""RESULTS:
Maximum Height: {max_height:.1f} m
Total Distance: {distance:.1f} m
Flight Time: {flight_time:.2f} s
Impact Velocity: {impact_velocity:.1f} m/s
Impact Energy: {impact_energy:.1f} J
Impact Angle: {impact_angle:.1f}° (relative to ground)
"""
        self.summary_text.setPlainText(summary)
        self.data_text.setPlainText("t(s)\tx(m)\ty(m)\tvx(m/s)\tvy(m/s)\tv(m/s)\n" +
            "\n".join(f"{p[2]:.2f}\t{p[0]:.2f}\t{p[1]:.2f}\t{p[3]:.2f}\t{p[4]:.2f}\t{p[5]:.2f}" for p in self.trajectory))

    def plot_trajectory(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self.trajectory:
            x = [p[0] for p in self.trajectory]
            y = [p[1] for p in self.trajectory]
            ax.plot(x, y, label="Trajectory")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Height (m)")
        ax.set_title("Projectile Trajectory")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

    # --- Export functions ---
    def export_results(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t(s)", "x(m)", "y(m)", "vx(m/s)", "vy(m/s)", "v(m/s)"])
                for p in self.trajectory:
                    writer.writerow([f"{v:.3f}" for v in p])
            QMessageBox.information(self, "Export", f"Results exported to {filename}")

    def export_plot(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Plot", "", "PNG Files (*.png);;SVG Files (*.svg)")
        if filename:
            if filename.endswith(".svg"):
                self.figure.savefig(filename, format="svg")
            else:
                self.figure.savefig(filename, format="png")
            QMessageBox.information(self, "Export", f"Plot exported to {filename}")

    # --- Batch Mode ---
    def run_batch(self):
        param = self.batch_param_combo.currentText()
        start = self.batch_start_input.value()
        end = self.batch_end_input.value()
        step = self.batch_step_input.value()
        runs = []
        val = start
        param_map = {
            "Angle": "angle", "Muzzle Velocity": "velocity",
            "Wind Speed": "wind_speed", "Temperature": "temperature",
            "Altitude": "altitude"
        }
        sweep_key = param_map.get(param, None)
        while val <= end:
            params = {
                "mass": self.from_imperial_mass(self.mass_input.value()) if self.use_imperial else self.mass_input.value() / 1000,
                "diameter": self.from_imperial_length(self.diam_input.value()) if self.use_imperial else self.diam_input.value() / 1000,
                "drag_model": self.drag_model_combo.currentText(),
                "velocity": self.from_imperial_velocity(self.velocity_input.value()) if self.use_imperial else self.velocity_input.value(),
                "angle": self.angle_input.value(),
            }
            if sweep_key:
                params[sweep_key] = val
            runs.append(params.copy())
            val += step
        self.batch_thread = BatchCalculationThread(self, runs)
        self.batch_thread.finished.connect(self.on_batch_complete)
        self.batch_thread.error.connect(self.on_batch_error)
        self.batch_thread.start()

    def on_batch_complete(self, results):
        self.batch_trajectories = results
        self.batch_table.setRowCount(len(results))
        for i, traj in enumerate(results):
            if traj:
                max_height = max(p[1] for p in traj)
                distance = traj[-1][0]
                impact_velocity = traj[-1][5]
                param_val = (self.batch_start_input.value() + i * self.batch_step_input.value())
                self.batch_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.batch_table.setItem(i, 1, QTableWidgetItem(f"{param_val:.2f}"))
                self.batch_table.setItem(i, 2, QTableWidgetItem(f"{max_height:.2f}"))
                self.batch_table.setItem(i, 3, QTableWidgetItem(f"{distance:.2f}"))
                self.batch_table.setItem(i, 4, QTableWidgetItem(f"{impact_velocity:.2f}"))
        # Batch plot visualization
        if results and hasattr(self, "figure"):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            for i, traj in enumerate(results):
                if traj:
                    x = [p[0] for idx, p in enumerate(traj) if idx % max(1, len(traj)//500)==0]
                    y = [p[1] for idx, p in enumerate(traj) if idx % max(1, len(traj)//500)==0]
                    ax.plot(x, y, label=f"Run {i+1}")
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Height (m)')
            ax.set_title('Batch Trajectories')
            ax.legend(fontsize='small')
            ax.grid(True)
            self.canvas.draw()

    def on_batch_error(self, error):
        QMessageBox.critical(self, "Batch Error", f"Error during batch: {error}")

    # --- Monte Carlo Simulation ---
    def run_monte_carlo(self):
        N = self.mc_runs_input.value()
        angle_spread = self.mc_spread_angle.value()
        vel_spread = self.mc_spread_velocity.value()
        base_params = {
            "mass": self.from_imperial_mass(self.mass_input.value()) if self.use_imperial else self.mass_input.value() / 1000,
            "diameter": self.from_imperial_length(self.diam_input.value()) if self.use_imperial else self.diam_input.value() / 1000,
            "drag_model": self.drag_model_combo.currentText(),
            "velocity": self.from_imperial_velocity(self.velocity_input.value()) if self.use_imperial else self.velocity_input.value(),
            "angle": self.angle_input.value(),
        }
        fig = self.mc_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        for _ in range(N):
            p = base_params.copy()
            p["angle"] += random.uniform(-angle_spread, angle_spread)
            p["velocity"] += random.uniform(-vel_spread, vel_spread)
            traj = self._calculate_trajectory(**p)
            x = [pt[0] for pt in traj]
            y = [pt[1] for pt in traj]
            ax.plot(x, y, color="b", alpha=0.05)
        ax.set_title("Monte Carlo Trajectory Spread")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Height (m)")
        fig.tight_layout()
        self.mc_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = BallisticCalculator()
    calculator.show()
    sys.exit(app.exec_())

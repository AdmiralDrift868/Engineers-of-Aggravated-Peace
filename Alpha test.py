#!/usr/bin/env python3
import sys
import math
import csv
import json
import os
from functools import lru_cache
from collections import namedtuple
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
                             QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
                             QFileDialog, QMessageBox, QInputDialog, QScrollArea, QSizePolicy,
                             QGridLayout, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Constants
GRAVITY = 9.80665        # m/s^2
EARTH_RADIUS = 6371000   # meters
EARTH_ROTATION_RATE = 7.292115e-5  # rad/s

# PRESETS_FILE for persistent user presets
PRESETS_FILE = os.path.join(os.path.expanduser("~"), ".ballistic_presets.json")


def speed_of_sound(temperature_celsius):
    """Calculate speed of sound as a function of temperature (m/s)."""
    return 331.3 * math.sqrt(1 + temperature_celsius / 273.15)


class DragModel:
    """Enhanced drag coefficient tables for standard models."""

    @staticmethod
    @lru_cache(maxsize=1000)
    def G1(mach):
        """Standard projectile (flat-base) drag function – G1 table (Mach-indexed)."""
        if mach > 4.0:  return 0.45
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
        else:            return 0.25

    @staticmethod
    @lru_cache(maxsize=1000)
    def G7(mach):
        """Long-range boat-tail drag function – G7 table (Mach-indexed)."""
        if mach > 4.0:   return 0.38
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
        else:            return 0.21

    @staticmethod
    @lru_cache(maxsize=1000)
    def rocket(mach):
        """Drag coefficient for fin-stabilised rockets (Mach-indexed)."""
        if mach > 3.0:   return 0.50
        elif mach > 2.0: return 0.45
        elif mach > 1.5: return 0.40
        elif mach > 1.0: return 0.35
        elif mach > 0.8: return 0.30
        else:            return 0.25

    @staticmethod
    @lru_cache(maxsize=1000)
    def mortar(mach):
        """Drag coefficient for mortar shells (Mach-indexed)."""
        if mach > 1.5:   return 0.55
        elif mach > 1.0: return 0.50
        elif mach > 0.8: return 0.45
        else:            return 0.40


class Projectile:
    def __init__(self, mass=0.01, diameter=0.01, drag_model='G7', velocity=800,
                 projectile_type='bullet', thrust_curve=None, burn_time=0,
                 propellant_mass=0.0):
        self.mass = mass                          # kg – total initial (wet) mass
        self.diameter = diameter                  # metres
        self.drag_model = drag_model
        self.velocity = velocity                  # m/s
        self.area = math.pi * (diameter / 2) ** 2
        self.projectile_type = projectile_type
        self.thrust_curve = thrust_curve or {}    # {time_s: thrust_N, ...}
        self.burn_time = burn_time                # s
        self.initial_mass = mass                  # kg – kept for reference
        # Propellant mass burned during flight; defaults to 10 % of initial mass
        # if not supplied and is a rocket.
        if propellant_mass > 0:
            self.propellant_mass = propellant_mass
        elif projectile_type == 'rocket' and burn_time > 0:
            self.propellant_mass = 0.10 * mass    # 10 % heuristic
        else:
            self.propellant_mass = 0.0

    # ------------------------------------------------------------------
    def drag_coefficient(self, velocity, sos):
        """Return drag coefficient based on current velocity and local speed of sound."""
        mach = velocity / max(sos, 1.0)
        if self.drag_model == 'G1':
            return DragModel.G1(mach)
        elif self.drag_model == 'G7':
            return DragModel.G7(mach)
        elif self.drag_model == 'rocket':
            return DragModel.rocket(mach)
        elif self.drag_model == 'mortar':
            return DragModel.mortar(mach)
        else:
            return 0.3  # Default

    # ------------------------------------------------------------------
    def get_thrust(self, time):
        """Get instantaneous thrust via linear interpolation of the thrust curve."""
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
                return f0 + (f1 - f0) * (time - t0) / (t1 - t0)
        return 0.0

    # ------------------------------------------------------------------
    def get_mass(self, time):
        """
        Calculate current (instantaneous) mass.

        For rockets the propellant mass decreases linearly from initial_mass
        down to (initial_mass - propellant_mass) over the burn time.
        After burnout or for non-rockets the dry mass is constant.
        """
        if self.projectile_type != 'rocket' or self.burn_time <= 0:
            return self.mass
        if time >= self.burn_time:
            # Dry (post-burn) mass
            return self.initial_mass - self.propellant_mass
        # Linear burn
        fraction_burned = time / self.burn_time
        return self.initial_mass - self.propellant_mass * fraction_burned


class Environment:
    def __init__(self, altitude=0, temperature=15, pressure=1013.25, humidity=50,
                 wind_speed=0, wind_angle=0, coriolis=False, latitude=45):
        self.altitude = altitude        # metres
        self.temperature = temperature  # °C
        self.pressure = pressure        # hPa
        self.humidity = humidity        # %
        self.wind_speed = wind_speed    # m/s
        self.wind_angle = wind_angle    # degrees (meteorological: 0 = from North)
        self.coriolis = coriolis
        self.latitude = latitude        # degrees
        self.air_density = self._compute_air_density()
        self.sos = speed_of_sound(temperature)   # m/s – local speed of sound

    # ------------------------------------------------------------------
    def _compute_air_density(self):
        """
        Air density using the CIPM-2007 moist-air formula, then a standard-
        atmosphere altitude correction (lapse-rate model up to 11 km,
        isothermal stratosphere above).
        """
        T = self.temperature + 273.15   # Kelvin
        R_dry = 287.058                 # J/(kg·K)
        R_vap = 461.495                 # J/(kg·K)

        # Saturation vapour pressure (Buck equation, hPa)
        if self.temperature >= 0:
            svp = 6.1121 * math.exp((18.678 - self.temperature / 234.5)
                                    * (self.temperature / (257.14 + self.temperature)))
        else:
            svp = 6.1115 * math.exp((23.036 - self.temperature / 333.7)
                                    * (self.temperature / (279.82 + self.temperature)))

        # Partial pressures (Pa)
        p_total = self.pressure * 100.0
        p_vap   = (self.humidity / 100.0) * svp * 100.0
        p_dry   = p_total - p_vap

        # Density at station altitude (kg/m³)
        rho_0 = p_dry / (R_dry * T) + p_vap / (R_vap * T)

        # Standard atmosphere altitude correction
        h = self.altitude
        if h <= 11000:
            # Troposphere: T decreases at 6.5 K/km
            T_sl = 288.15
            L    = 0.0065    # K/m
            rho = rho_0 * ((T_sl - L * h) / T_sl) ** (GRAVITY / (R_dry * L) - 1)
        else:
            # Isothermal stratosphere base (11–20 km approximation)
            rho_11 = rho_0 * (216.65 / 288.15) ** (GRAVITY / (R_dry * 0.0065) - 1)
            rho    = rho_11 * math.exp(-GRAVITY * (h - 11000) / (R_dry * 216.65))

        return max(rho, 1e-6)   # guard against zero/negative


class CalculationThread(QThread):
    """Worker thread for trajectory integration (keeps UI responsive)."""
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)   # 0-100 %

    def __init__(self, calculator, params):
        super().__init__()
        self.calculator = calculator
        self.params     = params

    def run(self):
        try:
            result = self.calculator._calculate_trajectory(
                progress_cb=self.progress.emit, **self.params)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BallisticCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Ballistic Calculator")
        self.setGeometry(100, 100, 1200, 900)
        self.setMinimumSize(800, 600)

        self.projectile             = None
        self.environment            = None
        self.trajectory             = []
        self.previous_trajectories  = []
        self.metric_units           = True

        self.presets = self.load_presets()
        self.init_ui()
        self.apply_styles()

    # ==================================================================
    # Styles
    # ==================================================================
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

    # ==================================================================
    # Preset management
    # ==================================================================
    def load_presets(self):
        """Load presets: try user file first, fall back to built-in defaults."""
        default_presets = self._default_presets()
        if os.path.isfile(PRESETS_FILE):
            try:
                with open(PRESETS_FILE, 'r') as fh:
                    user_presets = json.load(fh)
                # Merge: user presets override defaults with same name
                merged = dict(default_presets)
                merged.update(user_presets)
                return merged
            except Exception:
                pass  # Corrupted file – fall through to defaults
        return default_presets

    def _persist_presets(self):
        """Save the full presets dict to disk (only user-added/modified ones)."""
        default_keys = set(self._default_presets().keys())
        user_presets = {k: v for k, v in self.presets.items()
                        if k not in default_keys}
        try:
            with open(PRESETS_FILE, 'w') as fh:
                json.dump(user_presets, fh, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Warning",
                                f"Could not persist presets to disk:\n{e}")

    def _default_presets(self):
        return {
            # ---- Bullets ------------------------------------------------
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

            # ---- Rockets (mass in grams, diameter in mm) ----------------
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

            # ---- Mortars ------------------------------------------------
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

    # ==================================================================
    # UI construction
    # ==================================================================
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tabs.addTab(self.create_input_tab(),    "Input")
        tabs.addTab(self.create_results_tab(),  "Results")
        tabs.addTab(self.create_plot_tab(),     "Graph")
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

        # Progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    # ------------------------------------------------------------------
    def create_input_tab(self):
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
        self.preset_combo.addItems(["Custom"] + list(self.presets.keys()))
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
        self.mass_input.setRange(0.1, 2_000_000)   # support up to ~2 000 kg
        self.mass_input.setDecimals(1)
        self.mass_input.setValue(10)
        self.mass_input.setSingleStep(1)
        proj_layout.addWidget(self.mass_input, 0, 1)

        proj_layout.addWidget(QLabel("Diameter (mm):"), 1, 0)
        self.diam_input = QDoubleSpinBox()
        self.diam_input.setRange(0.1, 1000)          # support up to 1 m diameter
        self.diam_input.setDecimals(2)
        self.diam_input.setValue(7.62)
        self.diam_input.setSingleStep(0.1)
        proj_layout.addWidget(self.diam_input, 1, 1)

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
        self.burn_time_input.setRange(0, 60)
        self.burn_time_input.setValue(1.0)
        self.burn_time_input.setSingleStep(0.1)
        burn_layout.addWidget(self.burn_time_input)
        rocket_layout.addLayout(burn_layout)

        thrust_layout = QHBoxLayout()
        thrust_layout.addWidget(QLabel("Peak Thrust (N):"))
        self.thrust_input = QDoubleSpinBox()
        self.thrust_input.setRange(0, 500_000)       # support large MRL rockets
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
            "Leave 0 to use 10 % of launch mass as a default.")
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

    # ------------------------------------------------------------------
    def create_results_tab(self):
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

    # ------------------------------------------------------------------
    def create_plot_tab(self):
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

    # ------------------------------------------------------------------
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

    # ==================================================================
    # Preset load/save
    # ==================================================================
    def update_projectile_type(self, type_str):
        """Update UI based on selected projectile type."""
        type_lower = type_str.lower()
        self.rocket_group.setVisible(type_lower == "rocket")
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
        if preset_name not in self.presets:
            return
        preset = self.presets[preset_name]

        # Block signals while updating to avoid partial-load callbacks
        self.preset_combo.blockSignals(True)

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
            peak  = max(curve.values()) if curve else 1000
            self.thrust_input.setValue(peak)
            # Propellant mass not stored in defaults; leave as 0 (auto 10 %)
            self.propellant_input.setValue(preset.get("propellant_mass", 0))

        self.preset_combo.blockSignals(False)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not (ok and name):
            return
        preset_data = {
            "mass":       self.mass_input.value(),
            "diameter":   self.diam_input.value(),
            "drag_model": self.drag_model_combo.currentText(),
            "velocity":   self.velocity_input.value(),
            "type":       self.type_combo.currentText().lower()
        }
        if self.type_combo.currentText().lower() == "rocket":
            preset_data.update({
                "burn_time":       self.burn_time_input.value(),
                "propellant_mass": self.propellant_input.value(),
                "thrust_curve":    {0: self.thrust_input.value(),
                                    self.burn_time_input.value(): 0}
            })
        self.presets[name] = preset_data
        if self.preset_combo.findText(name) == -1:
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentText(name)
        self._persist_presets()

    # ==================================================================
    # Zero-angle calculation (iterative bisection with drag)
    # ==================================================================
    def calculate_zero_angle(self):
        """
        Solve for the launch angle that causes the projectile to pass through
        zero drop (relative to sight line) at the specified zero range.
        Uses bisection over a full trajectory integration, so drag is included.
        """
        zero_range = self.zero_range_input.value()
        sight_h    = self.sight_height_input.value() / 1000.0  # mm → m
        velocity   = self.velocity_input.value()
        mass_kg    = self.mass_input.value() / 1000.0

        # Helper: compute height at zero_range for a given angle
        def height_at_range(angle_deg):
            traj = self._calculate_trajectory(
                mass=mass_kg,
                diameter=self.diam_input.value() / 1000.0,
                drag_model=self.drag_model_combo.currentText(),
                velocity=velocity,
                angle=angle_deg,
                altitude=self.altitude_input.value(),
                temperature=self.temp_input.value(),
                humidity=self.humidity_input.value(),
                wind_speed=0,    # No wind for zeroing
                wind_angle=0,
                coriolis=False,
                latitude=self.latitude_input.value(),
                progress_cb=None
            )
            if not traj:
                return None
            # Interpolate height at exactly zero_range
            for i in range(1, len(traj)):
                x0, x1 = traj[i-1][0], traj[i][0]
                if x0 <= zero_range <= x1:
                    frac = (zero_range - x0) / (x1 - x0) if x1 != x0 else 0
                    h = traj[i-1][1] + frac * (traj[i][1] - traj[i-1][1])
                    return h - sight_h   # Drop relative to sight line
            # Range not reached – projectile fell short
            return traj[-1][1] - sight_h

        # Bisection: find angle where height_at_range == 0
        lo, hi = 0.0, 45.0
        f_lo = height_at_range(lo)
        f_hi = height_at_range(hi)

        if f_lo is None or f_hi is None:
            QMessageBox.warning(self, "Zero Angle",
                                "Could not compute trajectory for zeroing.")
            return

        if f_lo * f_hi > 0:
            # Both same sign; try to extend range or report failure
            QMessageBox.warning(self, "Zero Angle",
                                f"No solution found in 0–45°.\n"
                                f"Check that zero range ({zero_range} m) is reachable.")
            return

        for _ in range(50):  # 50 iterations ≈ 0.000001° precision
            mid = (lo + hi) / 2.0
            f_mid = height_at_range(mid)
            if f_mid is None:
                break
            if f_lo * f_mid <= 0:
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

    # ==================================================================
    # Trajectory calculation
    # ==================================================================
    def calculate_trajectory(self):
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setText("Calculating…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        params = {
            "mass":        self.mass_input.value() / 1000.0,   # g → kg
            "diameter":    self.diam_input.value() / 1000.0,   # mm → m
            "drag_model":  self.drag_model_combo.currentText(),
            "velocity":    self.velocity_input.value(),
            "angle":       self.angle_input.value(),
            "altitude":    self.altitude_input.value(),
            "temperature": self.temp_input.value(),
            "humidity":    self.humidity_input.value(),
            "wind_speed":  self.wind_speed_input.value(),
            "wind_angle":  self.wind_angle_input.value(),
            "coriolis":    self.coriolis_check.isChecked(),
            "latitude":    self.latitude_input.value()
        }

        if self.trajectory:
            self.previous_trajectories.append(self.trajectory)
            if len(self.previous_trajectories) > 3:
                self.previous_trajectories.pop(0)

        self.calc_thread = CalculationThread(self, params)
        self.calc_thread.finished.connect(self.on_calculation_complete)
        self.calc_thread.error.connect(self.on_calculation_error)
        self.calc_thread.progress.connect(self.progress_bar.setValue)
        self.calc_thread.start()

    def on_calculation_complete(self, trajectory):
        self.trajectory = trajectory
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Trajectory")
        self.progress_bar.setVisible(False)
        if trajectory:
            self.update_results()
            self.plot_trajectory()
        else:
            QMessageBox.warning(self, "Warning", "No trajectory data was generated.")

    def on_calculation_error(self, error_msg):
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Calculate Trajectory")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Calculation failed:\n{error_msg}")

    # ------------------------------------------------------------------
    def _calculate_trajectory(self, mass, diameter, drag_model, velocity, angle,
                               altitude, temperature, humidity, wind_speed, wind_angle,
                               coriolis, latitude,
                               max_time_step=0.05, min_time_step=0.001,
                               progress_cb=None):
        """
        4th-order Runge-Kutta trajectory integrator with adaptive step size.

        State vector: [x, y, vx, vy]
          x  – horizontal range (m)
          y  – height above launch point (m)
          vx – horizontal velocity (m/s)
          vy – vertical velocity (m/s)

        Wind is decomposed into its headwind/tailwind component (along the
        launch azimuth) and a cross-wind component.  The 2-D simulation uses
        only the headwind component so that the drag force acts on the correct
        relative velocity.  Cross-wind drift is reported separately.
        """
        ptype = self.type_combo.currentText().lower()
        thrust_curve_raw = {0: self.thrust_input.value(),
                            self.burn_time_input.value(): 0} \
            if ptype == "rocket" else {}
        # If a preset was loaded, use its full thrust curve
        preset_name = self.preset_combo.currentText()
        if preset_name != "Custom" and preset_name in self.presets:
            p = self.presets[preset_name]
            if p.get("type") == "rocket" and "thrust_curve" in p:
                # Convert string keys (from JSON) to float
                thrust_curve_raw = {float(k): v
                                    for k, v in p["thrust_curve"].items()}

        projectile = Projectile(
            mass=mass,
            diameter=diameter,
            drag_model=drag_model,
            velocity=velocity,
            projectile_type=ptype,
            thrust_curve=thrust_curve_raw,
            burn_time=self.burn_time_input.value() if ptype == "rocket" else 0,
            propellant_mass=(self.propellant_input.value() / 1000.0)
                            if ptype == "rocket" else 0.0
        )

        environment = Environment(
            altitude=altitude,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            coriolis=coriolis,
            latitude=latitude
        )

        angle_rad = math.radians(angle)
        sos       = environment.sos            # local speed of sound

        # Decompose wind into headwind (along trajectory plane) and crosswind.
        # Wind angle is measured clockwise from the launch direction (0° = tailwind).
        wind_rad         = math.radians(wind_angle)
        wind_head        = wind_speed * math.cos(wind_rad)  # + = tailwind
        # wind_cross     = wind_speed * math.sin(wind_rad)  # lateral (not used in 2-D)

        # Effective wind components in 2-D plane (x = downrange, y = altitude)
        wind_x = wind_head   # horizontal wind adds to/subtracts from vx
        wind_y = 0.0         # no vertical wind component assumed

        # Initial state
        state = [0.0, 0.0,
                 velocity * math.cos(angle_rad),
                 velocity * math.sin(angle_rad)]

        def derivative(s, t):
            x, y, vx, vy = s
            v_rel_x = vx - wind_x
            v_rel_y = vy - wind_y
            v_rel   = math.hypot(v_rel_x, v_rel_y)

            if v_rel > 0:
                cd          = projectile.drag_coefficient(v_rel, sos)
                drag_force  = 0.5 * environment.air_density * v_rel**2 * cd * projectile.area
                cur_mass    = projectile.get_mass(t)
                ax = -(drag_force * v_rel_x) / (cur_mass * v_rel)
                ay = -GRAVITY - (drag_force * v_rel_y) / (cur_mass * v_rel)
            else:
                cur_mass = projectile.get_mass(t)
                ax = 0.0
                ay = -GRAVITY

            # Rocket thrust
            if projectile.projectile_type == 'rocket' and t < projectile.burn_time:
                thrust       = projectile.get_thrust(t)
                if v_rel > 0:
                    thrust_angle = math.atan2(vy, vx)
                else:
                    thrust_angle = angle_rad
                cur_mass = projectile.get_mass(t)
                if cur_mass > 0:
                    ax += (thrust * math.cos(thrust_angle)) / cur_mass
                    ay += (thrust * math.sin(thrust_angle)) / cur_mass

            # Coriolis effect (horizontal plane only in 2-D)
            if environment.coriolis:
                omega_z  = EARTH_ROTATION_RATE * math.sin(math.radians(environment.latitude))
                ax      +=  2 * omega_z * vy
                ay      -=  2 * omega_z * vx

            # Spin drift (bullets only)
            if (hasattr(self, 'spin_drift_check') and
                    self.spin_drift_check.isChecked() and
                    projectile.projectile_type == 'bullet'):
                twist_m   = self.twist_input.value() * 0.0254       # in → m
                spin_rate = (v_rel * 2 * math.pi) / max(twist_m, 1e-6)  # rad/s
                # Litz spin-drift approximation: Δy ≈ 1.25 × SG × t^1.83
                # Here we use a simplified per-step acceleration in x
                spin_drift_acc = 1e-5 * spin_rate * v_rel
                ax += spin_drift_acc

            return [vx, vy, ax, ay]

        trajectory  = []
        time        = 0.0
        max_flight  = 600.0   # hard ceiling: 10 minutes
        step_count  = 0
        # Estimate total steps for progress (rough)
        est_steps   = int(max_flight / min_time_step)

        while state[1] >= 0.0 and time < max_flight:
            trajectory.append((
                state[0],              # x (range)
                state[1],              # y (height)
                time,                  # t
                state[2],              # vx
                state[3],              # vy
                math.hypot(state[2], state[3])  # total speed
            ))

            # Adaptive step size: smaller when fast, larger when slow
            current_vel = math.hypot(state[2], state[3])
            time_step   = max(min_time_step,
                              min(max_time_step,
                                  max_time_step * (500.0 / max(50.0, current_vel))))

            # RK4
            k1 = derivative(state, time)
            s2 = [state[i] + 0.5 * time_step * k1[i] for i in range(4)]
            k2 = derivative(s2, time + 0.5 * time_step)
            s3 = [state[i] + 0.5 * time_step * k2[i] for i in range(4)]
            k3 = derivative(s3, time + 0.5 * time_step)
            s4 = [state[i] + time_step * k3[i] for i in range(4)]
            k4 = derivative(s4, time + time_step)

            for i in range(4):
                state[i] += (time_step / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])

            time      += time_step
            step_count += 1

            # Emit progress approximately every 500 steps
            if progress_cb is not None and step_count % 500 == 0:
                pct = min(99, int(100 * time / max_flight))
                progress_cb(pct)

        # Append final point (impact / landing)
        trajectory.append((
            state[0], max(0.0, state[1]), time,
            state[2], state[3],
            math.hypot(state[2], state[3])
        ))
        if progress_cb is not None:
            progress_cb(100)

        return trajectory

    # ==================================================================
    # Results display
    # ==================================================================
    def update_results(self):
        if not self.trajectory:
            return

        max_height    = max(p[1] for p in self.trajectory)
        distance      = self.trajectory[-1][0]
        flight_time   = self.trajectory[-1][2]
        impact_vel    = self.trajectory[-1][5]

        # Use final (dry) mass for impact energy
        ptype  = self.type_combo.currentText().lower()
        m_kg   = self.mass_input.value() / 1000.0
        if ptype == "rocket":
            prop_g = self.propellant_input.value()
            if prop_g > 0:
                m_final = m_kg - prop_g / 1000.0
            else:
                m_final = m_kg * 0.90   # 10 % propellant heuristic
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

        # Detailed data table
        data_header = "Time(s)\tRange(m)\tHeight(m)\tVx(m/s)\tVy(m/s)\tSpeed(m/s)\n"
        data_lines  = [data_header]
        stride      = max(1, len(self.trajectory) // 200)   # show ~200 rows
        for i, point in enumerate(self.trajectory):
            if i % stride == 0:
                data_lines.append(
                    f"{point[2]:.3f}\t{point[0]:.1f}\t{point[1]:.1f}\t"
                    f"{point[3]:.1f}\t{point[4]:.1f}\t{point[5]:.1f}\n"
                )
        self.data_text.setPlainText(''.join(data_lines))

    # ==================================================================
    # Plot
    # ==================================================================
    def plot_trajectory(self):
        if not self.trajectory:
            return

        self.figure.clear()

        # Two subplots: trajectory and velocity profile
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        x  = [p[0] for p in self.trajectory]
        y  = [p[1] for p in self.trajectory]
        t  = [p[2] for p in self.trajectory]
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

    # ==================================================================
    # Export
    # ==================================================================
    def export_to_csv(self):
        if not self.trajectory:
            QMessageBox.warning(self, "Warning", "No trajectory data to export.")
            return

        options  = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv)", options=options)
        if not filename:
            return
        if not filename.endswith('.csv'):
            filename += '.csv'

        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Metadata header
                writer.writerow(["# Advanced Ballistic Calculator Export",
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
                writer.writerow([])   # blank separator
                writer.writerow(['Time(s)', 'Range(m)', 'Height(m)',
                                  'Vx(m/s)', 'Vy(m/s)', 'Speed(m/s)'])
                for point in self.trajectory:
                    writer.writerow([f"{v:.4f}" for v in point])

            QMessageBox.information(self, "Success",
                                    f"Data exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    # ==================================================================
    # About
    # ==================================================================
    def show_about(self):
        about_text = (
            "Advanced Ballistic Calculator\n\n"
            "Version 3.1\n\n"
            "Features:\n"
            "  • Bullets, rockets, and mortars\n"
            "  • 30+ built-in presets\n"
            "  • Adaptive RK4 integration\n"
            "  • G1 / G7 / rocket / mortar drag tables (Mach-indexed)\n"
            "  • Temperature-dependent speed of sound\n"
            "  • CIPM-2007 moist-air density with standard-atmosphere altitude\n"
            "  • Wind decomposition (headwind/tailwind vs crosswind)\n"
            "  • Coriolis effect\n"
            "  • Spin drift modelling\n"
            "  • Rocket thrust curve & propellant mass burn\n"
            "  • Iterative zero-angle solver (drag-corrected)\n"
            "  • Trajectory & velocity-profile plots\n"
            "  • Threaded calculation with progress indicator\n"
            "  • CSV export with metadata\n"
            "  • Persistent user presets\n\n"
            "Created for Kali Linux"
        )
        QMessageBox.about(self, "About", about_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = BallisticCalculator()
    calculator.show()
    sys.exit(app.exec_())

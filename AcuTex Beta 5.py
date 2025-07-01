#!/usr/bin/env python3
"""
Advanced Ballistic Calculator v4.0

Original author: AdmiralDrift868
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
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
    QFileDialog, QMessageBox, QScrollArea, QSizePolicy,
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
        self.air_density = 1.225  # For simplicity

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
        self.setWindowTitle("Advanced Ballistic Calculator")
        self.setGeometry(100, 100, 1100, 900)
        self.trajectory: List[Any] = []
        self.batch_trajectories: List[List[Any]] = []
        self.dark_mode_enabled = False
        self.use_imperial = False
        self.dynamic_air = False
        self.last_preset = "Custom"
        self.favorite_presets = set()
        self.custom_drag_curve: Optional[List[Tuple[float, float]]] = None
        self.presets = self.load_presets()
        self.init_ui()
        self.apply_styles()
        log("Ballistic Calculator started")

    def load_presets(self) -> Dict[str, Any]:
        return {
            "7.62 NATO Ball": {
                "type": "bullet",
                "mass": 9.5,
                "diameter": 7.82,
                "velocity": 830,
                "drag_model": "G7",
                "angle": 0
            }
        }

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
        main_layout.addWidget(self.tabs, 1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def apply_styles(self):
        self.setStyleSheet("")

    def create_input_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Type and preset
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Projectile Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Bullet", "Rocket", "Mortar"])
        row1.addWidget(self.type_combo)
        row1.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        self.preset_combo.addItems(["Custom"] + list(self.presets.keys()))
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.lineEdit().textChanged.connect(self.filter_presets)
        row1.addWidget(self.preset_combo)
        load_btn = QPushButton("Load Preset")
        load_btn.clicked.connect(self.load_preset)
        row1.addWidget(load_btn)
        save_btn = QPushButton("Save Preset")
        save_btn.clicked.connect(self.save_preset)
        row1.addWidget(save_btn)
        del_btn = QPushButton("Delete Preset")
        del_btn.clicked.connect(self.delete_preset)
        row1.addWidget(del_btn)
        layout.addLayout(row1)

        # Projectile parameters
        proj_group = QGroupBox("Projectile Parameters")
        proj_layout = QGridLayout()
        proj_layout.addWidget(QLabel("Mass (g):"), 0, 0)
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.1, 1000000)
        self.mass_input.setValue(10)
        proj_layout.addWidget(self.mass_input, 0, 1)
        proj_layout.addWidget(QLabel("Diameter (mm):"), 1, 0)
        self.diam_input = QDoubleSpinBox()
        self.diam_input.setRange(0.1, 1000)
        self.diam_input.setValue(7.62)
        proj_layout.addWidget(self.diam_input, 1, 1)
        proj_layout.addWidget(QLabel("Drag Model:"), 2, 0)
        self.drag_model_combo = QComboBox()
        self.drag_model_combo.addItems(["G1", "G7", "Custom"])
        proj_layout.addWidget(self.drag_model_combo, 2, 1)
        drag_curve_btn = QPushButton("Load Custom Drag")
        drag_curve_btn.clicked.connect(self.load_custom_drag_curve)
        proj_layout.addWidget(drag_curve_btn, 2, 2)
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)

        # Launch Parameters
        launch_group = QGroupBox("Launch Parameters")
        launch_layout = QGridLayout()
        launch_layout.addWidget(QLabel("Muzzle Velocity (m/s):"), 0, 0)
        self.velocity_input = QDoubleSpinBox()
        self.velocity_input.setRange(1, 5000)
        self.velocity_input.setValue(800)
        launch_layout.addWidget(self.velocity_input, 0, 1)
        launch_layout.addWidget(QLabel("Launch Angle (deg):"), 1, 0)
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(0, 90)
        self.angle_input.setValue(15)
        launch_layout.addWidget(self.angle_input, 1, 1)
        launch_group.setLayout(launch_layout)
        layout.addWidget(launch_group)

        # Environmental Parameters
        env_group = QGroupBox("Environmental Parameters")
        env_layout = QGridLayout()
        env_layout.addWidget(QLabel("Altitude (m):"), 0, 0)
        self.altitude_input = QDoubleSpinBox()
        self.altitude_input.setRange(-100, 20000)
        self.altitude_input.setValue(0)
        env_layout.addWidget(self.altitude_input, 0, 1)
        env_layout.addWidget(QLabel("Temperature (°C):"), 1, 0)
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(-50, 60)
        self.temp_input.setValue(15)
        env_layout.addWidget(self.temp_input, 1, 1)
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)

        # Wind and advanced
        wind_group = QGroupBox("Wind and Advanced")
        wind_layout = QGridLayout()
        wind_layout.addWidget(QLabel("Wind Speed (m/s):"), 0, 0)
        self.wind_speed_input = QDoubleSpinBox()
        self.wind_speed_input.setRange(-100, 100)
        self.wind_speed_input.setValue(0)
        wind_layout.addWidget(self.wind_speed_input, 0, 1)
        wind_layout.addWidget(QLabel("Wind Angle (deg):"), 1, 0)
        self.wind_angle_input = QSpinBox()
        self.wind_angle_input.setRange(0, 359)
        self.wind_angle_input.setValue(0)
        wind_layout.addWidget(self.wind_angle_input, 1, 1)
        self.coriolis_check = QCheckBox("Coriolis Effect")
        wind_layout.addWidget(self.coriolis_check, 2, 0)
        wind_layout.addWidget(QLabel("Latitude:"), 2, 1)
        self.latitude_input = QDoubleSpinBox()
        self.latitude_input.setRange(-90, 90)
        self.latitude_input.setValue(45)
        wind_layout.addWidget(self.latitude_input, 2, 2)
        self.dynamic_air_check = QCheckBox("Use ISA Dynamic Air Density")
        wind_layout.addWidget(self.dynamic_air_check, 3, 0, 1, 3)
        wind_group.setLayout(wind_layout)
        layout.addWidget(wind_group)

        # Calculate Button
        calc_btn = QPushButton("Calculate Trajectory")
        calc_btn.clicked.connect(self.calculate_trajectory)
        layout.addWidget(calc_btn)

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
        # Sweep controls
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
            "Run", "Param Value", "Max Height (m)", "Total Distance (m)", "Impact Velocity (m/s)"
        ])
        layout.addWidget(self.batch_table, 1)
        return tab

    def create_monte_carlo_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        # Controls
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
        controls.addWidget(QLabel("Velocity ±m/s:")); controls.addWidget(self.mc_spread_velocity)
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
        return tab

    def create_help_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText(
            "Advanced Ballistic Calculator Help\n\n"
            "1. Select projectile type and preset or enter custom parameters.\n"
            "2. Use tooltips for guidance.\n"
            "3. Batch Mode allows sweeping over angle, velocity, wind, etc.\n"
            "4. Monte Carlo tab simulates random dispersion.\n"
            "5. Export, save, and analyze your results."
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
        self.type_combo.setCurrentText(p.get("type", "Bullet").capitalize())
        self.mass_input.setValue(p.get("mass", 10))
        self.diam_input.setValue(p.get("diameter", 7.62))
        self.velocity_input.setValue(p.get("velocity", 800))
        self.angle_input.setValue(p.get("angle", 15))
        self.drag_model_combo.setCurrentText(p.get("drag_model", "G7"))
        # Add more as needed (alt, wind, etc.)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name.strip():
            self.presets[name] = {
                "type": self.type_combo.currentText().lower(),
                "mass": self.mass_input.value(),
                "diameter": self.diam_input.value(),
                "velocity": self.velocity_input.value(),
                "angle": self.angle_input.value(),
                "drag_model": self.drag_model_combo.currentText()
            }
            self.preset_combo.addItem(name)
            QMessageBox.information(self, "Preset Saved", f"Preset '{name}' saved.")

    def delete_preset(self):
        key = self.preset_combo.currentText()
        if key in self.presets:
            del self.presets[key]
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

    # --- Trajectory Calculation ---
    def _calculate_trajectory(self, **params):
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
            "mass": self.mass_input.value() / 1000,
            "diameter": self.diam_input.value() / 1000,
            "drag_model": self.drag_model_combo.currentText(),
            "velocity": self.velocity_input.value(),
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
Maximum Height: {max_height:.1f}m
Total Distance: {distance:.1f}m
Flight Time: {flight_time:.2f}s
Impact Velocity: {impact_velocity:.1f}m/s
Impact Energy: {impact_energy:.1f}J
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
                "mass": self.mass_input.value() / 1000,
                "diameter": self.diam_input.value() / 1000,
                "drag_model": self.drag_model_combo.currentText(),
                "velocity": self.velocity_input.value(),
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
            "mass": self.mass_input.value() / 1000,
            "diameter": self.diam_input.value() / 1000,
            "drag_model": self.drag_model_combo.currentText(),
            "velocity": self.velocity_input.value(),
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

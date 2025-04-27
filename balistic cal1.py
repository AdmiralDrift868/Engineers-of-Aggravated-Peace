#!/usr/bin/env python3
import sys
import math
import csv
from collections import namedtuple
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
                             QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Constants
GRAVITY = 9.80665  # m/s^2
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m^3

# Ballistic Coefficients (G1-G7 models)
DRAG_MODELS = {
    'G1': {'coeff': 0.519, 'desc': 'Standard bullet'},
    'G2': {'coeff': 0.515, 'desc': '30-caliber'},
    'G5': {'coeff': 0.485, 'desc': 'Short 7° cone'},
    'G7': {'coeff': 0.371, 'desc': 'Long-range boat tail'},
    'Custom': {'coeff': 0.47, 'desc': 'User-defined coefficient'}
}

class Projectile:
    def __init__(self, mass=0.01, diameter=0.01, drag_model='G1', drag_coeff=0.47, velocity=800):
        self.mass = mass  # kg
        self.diameter = diameter  # meters
        self.drag_model = drag_model
        self.drag_coeff = drag_coeff if drag_model == 'Custom' else DRAG_MODELS[drag_model]['coeff']
        self.velocity = velocity  # m/s
        self.area = math.pi * (diameter/2)**2

class Environment:
    def __init__(self, altitude=0, temperature=15, pressure=1013.25, humidity=50,
                 wind_speed=0, wind_angle=0, coriolis=False, latitude=45):
        self.altitude = altitude  # meters
        self.temperature = temperature  # °C
        self.pressure = pressure  # hPa
        self.humidity = humidity  # %
        self.wind_speed = wind_speed  # m/s
        self.wind_angle = wind_angle  # degrees (0=headwind, 180=tailwind)
        self.coriolis = coriolis
        self.latitude = latitude
        self.air_density = self.calculate_air_density()
    
    def calculate_air_density(self):
        # Improved air density calculation using CIPM-2007 equation
        temp_kelvin = self.temperature + 273.15
        R = 287.058  # Specific gas constant for dry air, J/(kg·K)
        
        # Saturation vapor pressure
        svp = 6.1078 * 10**((7.5 * self.temperature) / (self.temperature + 237.3))
        
        # Vapor pressure
        vp = svp * self.humidity / 100
        
        # Enhanced air density calculation
        density = ((self.pressure * 100) / (R * temp_kelvin)) * (1 - (0.378 * vp) / (self.pressure * 100))
        
        # Simple altitude adjustment
        density *= math.exp(-self.altitude / 10000)
        
        return density

class BallisticCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Ballistic Calculator")
        self.setGeometry(100, 100, 1000, 800)
        
        self.projectile = Projectile()
        self.environment = Environment()
        self.trajectory = []
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_input_tab(), "Input")
        tabs.addTab(self.create_results_tab(), "Results")
        tabs.addTab(self.create_plot_tab(), "Graph")
        
        main_layout.addWidget(tabs)
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
    
    def create_input_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Projectile Group
        proj_group = QGroupBox("Projectile Parameters")
        proj_layout = QVBoxLayout()
        
        # Mass
        mass_layout = QHBoxLayout()
        mass_layout.addWidget(QLabel("Mass (g):"))
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.1, 10000)
        self.mass_input.setValue(10)
        self.mass_input.setSingleStep(1)
        mass_layout.addWidget(self.mass_input)
        proj_layout.addLayout(mass_layout)
        
        # Diameter
        diam_layout = QHBoxLayout()
        diam_layout.addWidget(QLabel("Diameter (mm):"))
        self.diam_input = QDoubleSpinBox()
        self.diam_input.setRange(0.1, 100)
        self.diam_input.setValue(7.62)
        self.diam_input.setSingleStep(0.1)
        diam_layout.addWidget(self.diam_input)
        proj_layout.addLayout(diam_layout)
        
        # Drag Model
        drag_layout = QHBoxLayout()
        drag_layout.addWidget(QLabel("Drag Model:"))
        self.drag_model_combo = QComboBox()
        self.drag_model_combo.addItems(DRAG_MODELS.keys())
        self.drag_model_combo.currentTextChanged.connect(self.update_drag_model)
        drag_layout.addWidget(self.drag_model_combo)
        
        # Custom Drag Coefficient
        self.drag_coeff_input = QDoubleSpinBox()
        self.drag_coeff_input.setRange(0.1, 1.0)
        self.drag_coeff_input.setValue(0.47)
        self.drag_coeff_input.setSingleStep(0.01)
        self.drag_coeff_input.setEnabled(False)
        drag_layout.addWidget(self.drag_coeff_input)
        proj_layout.addLayout(drag_layout)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        # Launch Parameters Group
        launch_group = QGroupBox("Launch Parameters")
        launch_layout = QVBoxLayout()
        
        # Velocity
        vel_layout = QHBoxLayout()
        vel_layout.addWidget(QLabel("Muzzle Velocity (m/s):"))
        self.velocity_input = QDoubleSpinBox()
        self.velocity_input.setRange(1, 2000)
        self.velocity_input.setValue(800)
        self.velocity_input.setSingleStep(10)
        vel_layout.addWidget(self.velocity_input)
        launch_layout.addLayout(vel_layout)
        
        # Angle
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Launch Angle (deg):"))
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(0, 90)
        self.angle_input.setValue(15)
        self.angle_input.setSingleStep(1)
        angle_layout.addWidget(self.angle_input)
        launch_layout.addLayout(angle_layout)
        
        launch_group.setLayout(launch_layout)
        layout.addWidget(launch_group)
        
        # Environment Group
        env_group = QGroupBox("Environmental Parameters")
        env_layout = QVBoxLayout()
        
        # Altitude
        alt_layout = QHBoxLayout()
        alt_layout.addWidget(QLabel("Altitude (m):"))
        self.altitude_input = QDoubleSpinBox()
        self.altitude_input.setRange(-100, 10000)
        self.altitude_input.setValue(0)
        self.altitude_input.setSingleStep(10)
        alt_layout.addWidget(self.altitude_input)
        env_layout.addLayout(alt_layout)
        
        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature (°C):"))
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(-50, 60)
        self.temp_input.setValue(15)
        self.temp_input.setSingleStep(1)
        temp_layout.addWidget(self.temp_input)
        env_layout.addLayout(temp_layout)
        
        # Pressure
        press_layout = QHBoxLayout()
        press_layout.addWidget(QLabel("Pressure (hPa):"))
        self.press_input = QDoubleSpinBox()
        self.press_input.setRange(800, 1100)
        self.press_input.setValue(1013.25)
        self.press_input.setSingleStep(1)
        press_layout.addWidget(self.press_input)
        env_layout.addLayout(press_layout)
        
        # Humidity
        hum_layout = QHBoxLayout()
        hum_layout.addWidget(QLabel("Humidity (%):"))
        self.humidity_input = QSpinBox()
        self.humidity_input.setRange(0, 100)
        self.humidity_input.setValue(50)
        hum_layout.addWidget(self.humidity_input)
        env_layout.addLayout(hum_layout)
        
        # Wind
        wind_layout = QHBoxLayout()
        wind_layout.addWidget(QLabel("Wind Speed (m/s):"))
        self.wind_speed_input = QDoubleSpinBox()
        self.wind_speed_input.setRange(0, 50)
        self.wind_speed_input.setValue(0)
        self.wind_speed_input.setSingleStep(0.5)
        wind_layout.addWidget(self.wind_speed_input)
        
        wind_layout.addWidget(QLabel("Angle (deg):"))
        self.wind_angle_input = QSpinBox()
        self.wind_angle_input.setRange(0, 359)
        self.wind_angle_input.setValue(0)
        wind_layout.addWidget(self.wind_angle_input)
        env_layout.addLayout(wind_layout)
        
        # Advanced options
        adv_layout = QHBoxLayout()
        self.coriolis_check = QCheckBox("Coriolis Effect")
        adv_layout.addWidget(self.coriolis_check)
        
        adv_layout.addWidget(QLabel("Latitude:"))
        self.latitude_input = QDoubleSpinBox()
        self.latitude_input.setRange(-90, 90)
        self.latitude_input.setValue(45)
        self.latitude_input.setEnabled(False)
        adv_layout.addWidget(self.latitude_input)
        
        self.coriolis_check.stateChanged.connect(
            lambda: self.latitude_input.setEnabled(self.coriolis_check.isChecked()))
        env_layout.addLayout(adv_layout)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
        # Calculate button
        self.calculate_btn = QPushButton("Calculate Trajectory")
        self.calculate_btn.clicked.connect(self.calculate_trajectory)
        layout.addWidget(self.calculate_btn)
        
        tab.setLayout(layout)
        return tab
    
    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Summary Group
        summary_group = QGroupBox("Summary Results")
        summary_layout = QVBoxLayout()
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Detailed Data Group
        data_group = QGroupBox("Trajectory Data")
        data_layout = QVBoxLayout()
        
        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        data_layout.addWidget(self.data_text)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        tab.setLayout(layout)
        return tab
    
    def create_plot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        tab.setLayout(layout)
        return tab
    
    def update_drag_model(self, model):
        self.drag_coeff_input.setEnabled(model == 'Custom')
        if model != 'Custom':
            self.drag_coeff_input.setValue(DRAG_MODELS[model]['coeff'])
    
    def calculate_trajectory(self):
        try:
            # Get projectile parameters
            mass = self.mass_input.value() / 1000  # Convert g to kg
            diameter = self.diam_input.value() / 1000  # Convert mm to m
            drag_model = self.drag_model_combo.currentText()
            drag_coeff = self.drag_coeff_input.value()
            velocity = self.velocity_input.value()
            angle = self.angle_input.value()
            
            # Get environmental parameters
            altitude = self.altitude_input.value()
            temperature = self.temp_input.value()
            pressure = self.press_input.value()
            humidity = self.humidity_input.value()
            wind_speed = self.wind_speed_input.value()
            wind_angle = self.wind_angle_input.value()
            coriolis = self.coriolis_check.isChecked()
            latitude = self.latitude_input.value()
            
            # Create projectile and environment objects
            self.projectile = Projectile(
                mass=mass,
                diameter=diameter,
                drag_model=drag_model,
                drag_coeff=drag_coeff,
                velocity=velocity
            )
            
            self.environment = Environment(
                altitude=altitude,
                temperature=temperature,
                pressure=pressure,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_angle=wind_angle,
                coriolis=coriolis,
                latitude=latitude
            )
            
            # Calculate trajectory
            self.trajectory = self._calculate_trajectory(
                initial_velocity=velocity,
                angle_degrees=angle,
                projectile=self.projectile,
                environment=self.environment
            )
            
            # Update results
            self.update_results()
            self.plot_trajectory()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def _calculate_trajectory(self, initial_velocity, angle_degrees, projectile, environment, time_step=0.1):
        """Core trajectory calculation with all factors."""
        angle_rad = math.radians(angle_degrees)
        wind_x = environment.wind_speed * math.cos(math.radians(environment.wind_angle))
        wind_y = environment.wind_speed * math.sin(math.radians(environment.wind_angle))
        
        x, y = 0, 0
        vx = initial_velocity * math.cos(angle_rad)
        vy = initial_velocity * math.sin(angle_rad)
        
        trajectory = []
        time = 0.0
        
        while y >= 0:
            # Relative velocity components (accounting for wind)
            v_rel_x = vx - wind_x
            v_rel_y = vy - wind_y
            v_rel = math.sqrt(v_rel_x**2 + v_rel_y**2)
            
            # Drag force (using ballistic coefficient)
            drag_force = 0.5 * environment.air_density * v_rel**2 * projectile.drag_coeff * projectile.area
            
            # Acceleration components
            ax = -(drag_force * v_rel_x) / (projectile.mass * v_rel) if v_rel > 0 else 0
            ay = -GRAVITY - (drag_force * v_rel_y) / (projectile.mass * v_rel) if v_rel > 0 else -GRAVITY
            
            # Coriolis effect (simplified)
            if environment.coriolis:
                coriolis_param = 2 * 7.292115e-5 * math.sin(math.radians(environment.latitude))
                ax += coriolis_param * vy
                ay -= coriolis_param * vx
            
            # Update velocity and position
            vx += ax * time_step
            vy += ay * time_step
            x += vx * time_step
            y += vy * time_step
            time += time_step
            
            if y >= 0:
                trajectory.append((x, y, time, vx, vy, math.sqrt(vx**2 + vy**2)))
        
        return trajectory
    
    def update_results(self):
        if not self.trajectory:
            return
        
        # Calculate summary metrics
        max_height = max(p[1] for p in self.trajectory)
        distance = self.trajectory[-1][0]
        flight_time = self.trajectory[-1][2]
        impact_velocity = self.trajectory[-1][5]
        impact_energy = 0.5 * self.projectile.mass * impact_velocity**2
        
        # Update summary text
        summary = f"""PROJECTILE:
Mass: {self.mass_input.value():.1f}g
Diameter: {self.diam_input.value():.1f}mm
Drag Model: {self.drag_model_combo.currentText()} (C={self.projectile.drag_coeff:.3f})
Muzzle Velocity: {self.velocity_input.value():.1f} m/s
Launch Angle: {self.angle_input.value():.1f}°

ENVIRONMENT:
Altitude: {self.environment.altitude:.0f}m
Temperature: {self.environment.temperature:.1f}°C
Pressure: {self.environment.pressure:.1f} hPa
Humidity: {self.environment.humidity}%
Wind: {self.environment.wind_speed:.1f}m/s @ {self.environment.wind_angle}°
Air Density: {self.environment.air_density:.5f} kg/m³
Coriolis: {'Yes' if self.environment.coriolis else 'No'}

RESULTS:
Maximum Height: {max_height:.1f}m
Total Distance: {distance:.1f}m
Flight Time: {flight_time:.2f}s
Impact Velocity: {impact_velocity:.1f}m/s
Impact Energy: {impact_energy:.1f}J ({impact_energy/9.81:.1f} kg·m)"""
        
        self.summary_text.setPlainText(summary)
        
        # Update detailed data
        data_header = "Time(s)\tDistance(m)\tHeight(m)\tVx(m/s)\tVy(m/s)\tVelocity(m/s)\n"
        data_lines = [data_header]
        
        for i, point in enumerate(self.trajectory):
            if i % 10 == 0:  # Show every 10th point
                data_lines.append(f"{point[2]:.2f}\t{point[0]:.1f}\t{point[1]:.1f}\t"
                               f"{point[3]:.1f}\t{point[4]:.1f}\t{point[5]:.1f}\n")
        
        self.data_text.setPlainText(''.join(data_lines))
    
    def plot_trajectory(self):
        if not self.trajectory:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        x = [p[0] for p in self.trajectory]
        y = [p[1] for p in self.trajectory]
        
        ax.plot(x, y, label='Trajectory')
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
Version 1.0\n
Features:
- Projectile trajectory calculation with air resistance
- Multiple drag models (G1-G7)
- Environmental factors (altitude, temperature, wind)
- Coriolis effect for long-range shots
- Graphical trajectory visualization
- Data export capability\n
Created for Kali Linux"""
        QMessageBox.about(self, "About", about_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = BallisticCalculator()
    calculator.show()
    sys.exit(app.exec_())
#!/usr/bin/env python3
import sys
import math
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
                             QGroupBox, QDoubleSpinBox, QTextEdit, QMessageBox, QTableWidget,
                             QHeaderView, QTableWidgetItem, QFileDialog)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Constants
GRAVITY = 9.80665  # m/s^2
EARTH_RADIUS = 6371000  # meters
DEG_TO_RAD = math.pi / 180
RAD_TO_DEG = 180 / math.pi
SOUND_SPEED = 343  # m/s at sea level, 20°C

class MapWidget(QWebEngineView):
    position_selected = pyqtSignal(float, float)  # lat, lon

    def __init__(self):
        super().__init__()
        self.load_map()

    def load_map(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ballistic Target Selection</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <style>
                #map { height: 100%; }
                body { margin: 0; padding: 0; }
                html, body, #map { width: 100%; height: 100%; }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <script>
                var map = L.map('map').setView([45.0, 9.0], 12);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                }).addTo(map);
                
                var fireMarker, targetMarker;
                var firePos = null, targetPos = null;
                
                function initMap() {
                    map.on('click', function(e) {
                        if (!firePos) {
                            firePos = e.latlng;
                            fireMarker = L.marker(firePos, {draggable: true})
                                .bindPopup("Firing Position")
                                .addTo(map);
                            fireMarker.on('dragend', updateFirePos);
                            external.invoke('fire:' + firePos.lat + ',' + firePos.lng);
                        } else if (!targetPos) {
                            targetPos = e.latlng;
                            targetMarker = L.marker(targetPos, {draggable: true})
                                .bindPopup("Target Position")
                                .addTo(map);
                            targetMarker.on('dragend', updateTargetPos);
                            external.invoke('target:' + targetPos.lat + ',' + targetPos.lng);
                        }
                    });
                }
                
                function updateFirePos() {
                    firePos = this.getLatLng();
                    external.invoke('fire:' + firePos.lat + ',' + firePos.lng);
                }
                
                function updateTargetPos() {
                    targetPos = this.getLatLng();
                    external.invoke('target:' + targetPos.lat + ',' + targetPos.lng);
                }
                
                initMap();
            </script>
        </body>
        </html>
        """
        self.setHtml(html, QUrl("about:blank"))

class BallisticCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tactical Ballistic Calculator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize systems
        self.projectile = None
        self.environment = None
        self.trajectory = []
        self.target_distance = 0
        self.target_bearing = 0
        self.elevation_diff = 0
        self.fire_pos = None
        self.target_pos = None
        
        # Setup UI
        self.init_ui()
        
        # Default values
        self.type_combo.setCurrentText("Mortar")
        self.update_projectile_type("Mortar")

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel (controls)
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        # Create tabs for controls
        tabs = QTabWidget()
        tabs.addTab(self.create_input_tab(), "Input")
        tabs.addTab(self.create_results_tab(), "Results")
        tabs.addTab(self.create_payload_tab(), "Payload")
        
        control_layout.addWidget(tabs)
        control_panel.setLayout(control_layout)
        
        # Right panel (map and plots)
        display_panel = QWidget()
        display_layout = QVBoxLayout()
        
        # Map widget
        self.map_widget = MapWidget()
        self.map_widget.position_selected.connect(self.handle_map_position)
        display_layout.addWidget(self.map_widget, stretch=2)
        
        # Altitude controls
        alt_layout = QHBoxLayout()
        alt_layout.addWidget(QLabel("Firing Alt (m):"))
        self.fire_alt_input = QDoubleSpinBox()
        self.fire_alt_input.setRange(-100, 5000)
        self.fire_alt_input.setValue(100)
        alt_layout.addWidget(self.fire_alt_input)
        
        alt_layout.addWidget(QLabel("Target Alt (m):"))
        self.target_alt_input = QDoubleSpinBox()
        self.target_alt_input.setRange(-100, 5000)
        self.target_alt_input.setValue(100)
        alt_layout.addWidget(self.target_alt_input)
        
        display_layout.addLayout(alt_layout)
        
        # Plot widget
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        display_layout.addWidget(self.canvas, stretch=1)
        
        display_panel.setLayout(display_layout)
        
        # Combine panels
        main_layout.addWidget(control_panel, stretch=1)
        main_layout.addWidget(display_panel, stretch=2)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        export_action = file_menu.addAction('Export Data')
        export_action.triggered.connect(self.export_data)
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)

    def create_input_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Projectile Type Selection
        type_group = QGroupBox("Projectile Type")
        type_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Mortar", "Bullet"])
        self.type_combo.currentTextChanged.connect(self.update_projectile_type)
        type_layout.addWidget(QLabel("Type:"))
        type_layout.addWidget(self.type_combo)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Projectile Parameters
        proj_group = QGroupBox("Projectile Parameters")
        proj_layout = QVBoxLayout()
        
        # Mass
        mass_layout = QHBoxLayout()
        mass_layout.addWidget(QLabel("Mass (kg):"))
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.1, 100)
        self.mass_input.setValue(3.2)
        mass_layout.addWidget(self.mass_input)
        proj_layout.addLayout(mass_layout)
        
        # Diameter
        diam_layout = QHBoxLayout()
        diam_layout.addWidget(QLabel("Diameter (mm):"))
        self.diam_input = QDoubleSpinBox()
        self.diam_input.setRange(10, 300)
        self.diam_input.setValue(82)
        diam_layout.addWidget(self.diam_input)
        proj_layout.addLayout(diam_layout)
        
        # Length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Length (mm):"))
        self.length_input = QDoubleSpinBox()
        self.length_input.setRange(10, 2000)
        self.length_input.setValue(400)
        length_layout.addWidget(self.length_input)
        proj_layout.addLayout(length_layout)
        
        # Drag Model
        drag_layout = QHBoxLayout()
        drag_layout.addWidget(QLabel("Drag Model:"))
        self.drag_model_combo = QComboBox()
        self.drag_model_combo.addItems(["M1", "M2", "M3", "G1", "G7"])
        drag_layout.addWidget(self.drag_model_combo)
        proj_layout.addLayout(drag_layout)
        
        # Charge Selection
        self.charge_layout = QHBoxLayout()
        self.charge_layout.addWidget(QLabel("Charge:"))
        self.charge_combo = QComboBox()
        self.charge_combo.addItems(["Charge 0", "Charge 1", "Charge 2", "Charge 3", "Charge 4"])
        self.charge_combo.currentTextChanged.connect(self.update_velocity_from_charge)
        self.charge_layout.addWidget(self.charge_combo)
        proj_layout.addLayout(self.charge_layout)
        
        # Velocity
        vel_layout = QHBoxLayout()
        vel_layout.addWidget(QLabel("Velocity (m/s):"))
        self.velocity_input = QDoubleSpinBox()
        self.velocity_input.setRange(1, 2000)
        self.velocity_input.setValue(200)
        vel_layout.addWidget(self.velocity_input)
        proj_layout.addLayout(vel_layout)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        # Launch Parameters
        launch_group = QGroupBox("Launch Parameters")
        launch_layout = QVBoxLayout()
        
        # Angle
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Angle (deg):"))
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(0, 90)
        self.angle_input.setValue(45)
        angle_layout.addWidget(self.angle_input)
        launch_layout.addLayout(angle_layout)
        
        # Charge Weight
        charge_weight_layout = QHBoxLayout()
        charge_weight_layout.addWidget(QLabel("Charge Weight (g):"))
        self.charge_weight_label = QLabel("0")
        charge_weight_layout.addWidget(self.charge_weight_label)
        launch_layout.addLayout(charge_weight_layout)
        
        launch_group.setLayout(launch_layout)
        layout.addWidget(launch_group)
        
        # Environment Group
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        # Wind
        wind_layout = QHBoxLayout()
        wind_layout.addWidget(QLabel("Wind Speed (m/s):"))
        self.wind_speed_input = QDoubleSpinBox()
        self.wind_speed_input.setRange(0, 50)
        self.wind_speed_input.setValue(5)
        wind_layout.addWidget(self.wind_speed_input)
        
        wind_layout.addWidget(QLabel("Direction (deg):"))
        self.wind_dir_input = QDoubleSpinBox()
        self.wind_dir_input.setRange(0, 359)
        self.wind_dir_input.setValue(90)
        wind_layout.addWidget(self.wind_dir_input)
        env_layout.addLayout(wind_layout)
        
        # Auto-calculate button
        self.calc_btn = QPushButton("Calculate Firing Solution")
        self.calc_btn.clicked.connect(self.calculate_solution)
        layout.addWidget(self.calc_btn)
        
        tab.setLayout(layout)
        return tab

    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Summary Group
        summary_group = QGroupBox("Firing Solution")
        summary_layout = QVBoxLayout()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Trajectory Data
        data_group = QGroupBox("Trajectory Data")
        data_layout = QVBoxLayout()
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(["Time (s)", "Distance (m)", "Height (m)", "Velocity (m/s)", "Mach", "Wind Drift (m)"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        data_layout.addWidget(self.data_table)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        tab.setLayout(layout)
        return tab

    def create_payload_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Payload Parameters
        payload_group = QGroupBox("Payload Configuration")
        payload_layout = QVBoxLayout()
        
        # Payload Type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.payload_type_combo = QComboBox()
        self.payload_type_combo.addItems(["HE", "Smoke", "Illumination", "Training"])
        type_layout.addWidget(self.payload_type_combo)
        payload_layout.addLayout(type_layout)
        
        # Payload Mass
        mass_layout = QHBoxLayout()
        mass_layout.addWidget(QLabel("Mass (kg):"))
        self.payload_mass_input = QDoubleSpinBox()
        self.payload_mass_input.setRange(0, 50)
        self.payload_mass_input.setValue(1.5)
        mass_layout.addWidget(self.payload_mass_input)
        payload_layout.addLayout(mass_layout)
        
        # Effects Table
        self.effects_table = QTableWidget()
        self.effects_table.setColumnCount(4)
        self.effects_table.setHorizontalHeaderLabels(["Range (m)", "Effect", "Value", "Units"])
        self.effects_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        payload_layout.addWidget(self.effects_table)
        
        payload_group.setLayout(payload_layout)
        layout.addWidget(payload_group)
        
        tab.setLayout(layout)
        return tab

    def update_projectile_type(self, proj_type):
        is_mortar = (proj_type == "Mortar")
        self.drag_model_combo.clear()
        
        if is_mortar:
            self.drag_model_combo.addItems(["M1", "M2", "M3"])
            self.charge_combo.setEnabled(True)
            self.velocity_input.setEnabled(False)
            self.mass_input.setRange(1, 50)
            self.diam_input.setRange(60, 120)
            self.charge_combo.setCurrentIndex(2)  # Default to Charge 2
        else:
            self.drag_model_combo.addItems(["G1", "G7"])
            self.charge_combo.setEnabled(False)
            self.velocity_input.setEnabled(True)
            self.mass_input.setRange(0.01, 1)
            self.diam_input.setRange(5, 20)
            self.velocity_input.setValue(800)  # Typical bullet velocity

    def update_velocity_from_charge(self, charge):
        try:
            charge_data = {
                "Charge 0": {"weight": 0, "velocity": 70},
                "Charge 1": {"weight": 100, "velocity": 110},
                "Charge 2": {"weight": 200, "velocity": 150},
                "Charge 3": {"weight": 300, "velocity": 190},
                "Charge 4": {"weight": 400, "velocity": 230}
            }.get(charge, {"weight": 0, "velocity": 200})
            
            self.velocity_input.setValue(charge_data["velocity"])
            self.charge_weight_label.setText(f"{charge_data['weight']}g")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update velocity: {str(e)}")

    def handle_map_position(self, lat, lon):
        try:
            if self.fire_pos is None:
                self.fire_pos = (lat, lon)
                QMessageBox.information(self, "Position Set", "Firing position recorded")
            else:
                self.target_pos = (lat, lon)
                self.calculate_geospatial()
                self.fire_pos = None  # Reset for next target selection
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to handle position: {str(e)}")

    def calculate_geospatial(self):
        try:
            if self.fire_pos is None or self.target_pos is None:
                raise ValueError("Both firing and target positions must be set")
            
            lat1, lon1 = self.fire_pos
            lat2, lon2 = self.target_pos
            alt1 = self.fire_alt_input.value()
            alt2 = self.target_alt_input.value()
            
            # Haversine distance calculation
            lat1_rad = lat1 * DEG_TO_RAD
            lon1_rad = lon1 * DEG_TO_RAD
            lat2_rad = lat2 * DEG_TO_RAD
            lon2_rad = lon2 * DEG_TO_RAD
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = EARTH_RADIUS * c
            
            # Bearing calculation
            y = math.sin(dlon) * math.cos(lat2_rad)
            x = math.cos(lat1_rad)*math.sin(lat2_rad) - math.sin(lat1_rad)*math.cos(lat2_rad)*math.cos(dlon)
            bearing = (math.atan2(y, x) * RAD_TO_DEG) % 360
            
            # Store results
            self.target_distance = distance
            self.target_bearing = bearing
            self.elevation_diff = alt2 - alt1
            
            # Update UI
            self.summary_text.setPlainText(
                f"Target Solution:\n"
                f"Distance: {distance:.1f}m\n"
                f"Bearing: {bearing:.1f}°\n"
                f"Elevation Δ: {self.elevation_diff:.1f}m\n"
                f"Wind: {self.wind_speed_input.value()} m/s from {self.wind_dir_input.value()}°"
            )
            
            # Auto-calculate if mortar
            if self.type_combo.currentText() == "Mortar":
                self.calculate_solution()
                
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Geospatial calculation failed: {str(e)}")

    def calculate_solution(self):
        try:
            # Get projectile parameters
            mass = self.mass_input.value()
            diameter = self.diam_input.value() / 1000  # Convert mm to m
            length = self.length_input.value() / 1000  # Convert mm to m
            drag_model = self.drag_model_combo.currentText()
            velocity = self.velocity_input.value()
            wind_speed = self.wind_speed_input.value()
            wind_dir = self.wind_dir_input.value()
            
            # Calculate optimal angle if target is set
            if hasattr(self, 'target_distance') and self.target_distance > 0:
                angle = self.calculate_optimal_angle(velocity, self.target_distance, self.elevation_diff)
                self.angle_input.setValue(angle)
            else:
                angle = self.angle_input.value()
            
            # Calculate trajectory with wind
            self.trajectory = self.calculate_trajectory(
                velocity,
                angle,
                mass, diameter, drag_model,
                wind_speed, wind_dir,
                self.elevation_diff
            )
            
            if not self.trajectory:
                raise ValueError("Trajectory calculation failed - no valid solution")
            
            # Update displays
            self.update_results_display()
            self.update_payload_effects()
            self.plot_trajectory()
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Failed to calculate solution: {str(e)}")

    def calculate_optimal_angle(self, velocity, distance, elevation_diff):
        try:
            best_angle = 45
            best_error = float('inf')
            
            # Test angles between 35-85° in 0.5° steps
            for test_angle in range(350, 851, 5):
                angle = test_angle / 10
                traj = self.calculate_trajectory(
                    velocity, angle,
                    self.mass_input.value(),
                    self.diam_input.value() / 1000,
                    self.drag_model_combo.currentText(),
                    self.wind_speed_input.value(),
                    self.wind_dir_input.value(),
                    elevation_diff
                )
                
                if not traj:
                    continue
                    
                impact_dist = traj[-1][1]
                error = abs(impact_dist - distance)
                
                if error < best_error:
                    best_error = error
                    best_angle = angle
                    
                if error < 1.0:  # 1m tolerance
                    break
            
            # Adjust for elevation
            if elevation_diff > 0:
                best_angle += elevation_diff / 100
            
            return min(85, max(35, best_angle))
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Angle optimization failed: {str(e)}")
            return 45  # Return default angle on failure

    def calculate_trajectory(self, velocity, angle, mass, diameter, drag_model, wind_speed, wind_dir, elevation_diff=0):
        try:
            angle_rad = math.radians(angle)
            g = GRAVITY
            
            # Initial conditions
            x, y = 0, 0
            vx = velocity * math.cos(angle_rad)
            vy = velocity * math.sin(angle_rad)
            
            # Wind components
            wind_x = wind_speed * math.cos(math.radians(wind_dir))
            wind_y = wind_speed * math.sin(math.radians(wind_dir))
            
            # Projectile parameters
            area = math.pi * (diameter/2)**2
            time_step = 0.05 if drag_model.startswith('M') else 0.01
            
            trajectory = []
            time = 0
            
            max_steps = 10000  # Prevent infinite loops
            step_count = 0
            
            while y >= -elevation_diff and step_count < max_steps:
                step_count += 1
                
                # Relative velocity (accounting for wind)
                v_rel_x = vx - wind_x
                v_rel_y = vy - wind_y
                v_rel = math.hypot(v_rel_x, v_rel_y)
                mach = v_rel / SOUND_SPEED
                
                # Get drag coefficient
                cd = self.get_drag_coefficient(drag_model, mach)
                
                # Drag force
                rho = 1.225  # Simplified air density
                drag_force = 0.5 * rho * v_rel**2 * cd * area
                
                # Acceleration components
                if v_rel > 0:
                    ax = -(drag_force * v_rel_x) / (mass * v_rel)
                    ay = -g - (drag_force * v_rel_y) / (mass * v_rel)
                else:
                    ax = 0
                    ay = -g
                
                # Update state
                vx += ax * time_step
                vy += ay * time_step
                x += vx * time_step
                y += vy * time_step
                time += time_step
                
                # Calculate wind drift
                drift = wind_speed * time * math.sin(math.radians(wind_dir - self.target_bearing))
                
                trajectory.append((
                    time, x, y, 
                    math.hypot(vx, vy), 
                    mach, drift
                ))
            
            return trajectory
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Trajectory calculation failed: {str(e)}")
            return []

    def get_drag_coefficient(self, model, mach):
        try:
            # Drag coefficient tables
            models = {
                'M1': [(0.0, 0.65), (1.0, 0.59)],
                'M2': [(0.0, 0.55), (1.0, 0.49)],
                'M3': [(0.0, 0.45), (1.0, 0.39)],
                'G1': [(0.0, 0.26), (1.0, 0.23), (3.0, 0.24)],
                'G7': [(0.0, 0.12), (1.0, 0.12), (3.0, 0.18)]
            }
            
            table = models.get(model, models['M1'])
            machs = [p[0] for p in table]
            cds = [p[1] for p in table]
            
            # Find interval for interpolation
            for i in range(len(machs)-1):
                if machs[i] <= mach <= machs[i+1]:
                    return cds[i] + (cds[i+1]-cds[i]) * (mach-machs[i])/(machs[i+1]-machs[i])
            
            # Extrapolate if beyond table bounds
            return cds[-1] if mach > machs[-1] else cds[0]
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Drag coefficient lookup failed: {str(e)}")
            return 0.3  # Default reasonable value

    def update_results_display(self):
        try:
            if not self.trajectory:
                raise ValueError("No trajectory data available")
            
            # Summary data
            flight_time = self.trajectory[-1][0]
            impact_dist = self.trajectory[-1][1]
            impact_vel = self.trajectory[-1][3]
            max_height = max(p[2] for p in self.trajectory)
            wind_drift = self.trajectory[-1][5]
            
            summary = (
                f"Projectile: {self.type_combo.currentText()}\n"
                f"Mass: {self.mass_input.value():.2f} kg\n"
                f"Diameter: {self.diam_input.value():.1f} mm\n"
                f"Velocity: {self.velocity_input.value():.1f} m/s\n"
                f"Angle: {self.angle_input.value():.1f}°\n\n"
                f"Flight Time: {flight_time:.1f} s\n"
                f"Impact Distance: {impact_dist:.1f} m\n"
                f"Impact Velocity: {impact_vel:.1f} m/s\n"
                f"Max Height: {max_height:.1f} m\n"
                f"Wind Drift: {wind_drift:.1f} m\n"
            )
            
            if hasattr(self, 'target_distance'):
                summary += f"\nTarget Distance: {self.target_distance:.1f} m\n"
                summary += f"Error: {abs(impact_dist - self.target_distance):.1f} m\n"
            
            self.summary_text.setPlainText(summary)
            
            # Trajectory table
            self.data_table.setRowCount(len(self.trajectory))
            for i, point in enumerate(self.trajectory):
                self.data_table.setItem(i, 0, QTableWidgetItem(f"{point[0]:.2f}"))
                self.data_table.setItem(i, 1, QTableWidgetItem(f"{point[1]:.1f}"))
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{point[2]:.1f}"))
                self.data_table.setItem(i, 3, QTableWidgetItem(f"{point[3]:.1f}"))
                self.data_table.setItem(i, 4, QTableWidgetItem(f"{point[4]:.2f}"))
                self.data_table.setItem(i, 5, QTableWidgetItem(f"{point[5]:.1f}"))
                
        except Exception as e:
            QMessageBox.warning(self, "Display Error", f"Failed to update results: {str(e)}")

    def update_payload_effects(self):
        try:
            payload_type = self.payload_type_combo.currentText()
            mass = self.payload_mass_input.value()
            
            if payload_type == "HE":
                frag_count = min(1000, mass * 1500)
                lethal_radius = math.sqrt(mass) * 15
                
                ranges = [100, 200, 500, 1000, 1500, 2000]
                self.effects_table.setRowCount(len(ranges))
                
                for i, r in enumerate(ranges):
                    eff_radius = max(0, lethal_radius - r*0.1)
                    self.effects_table.setItem(i, 0, QTableWidgetItem(f"{r}"))
                    self.effects_table.setItem(i, 1, QTableWidgetItem("Lethal Radius"))
                    self.effects_table.setItem(i, 2, QTableWidgetItem(f"{eff_radius:.1f}"))
                    self.effects_table.setItem(i, 3, QTableWidgetItem("m"))
                    
                    if i == 0:
                        self.effects_table.setItem(i, 1, QTableWidgetItem("Fragments"))
                        self.effects_table.setItem(i, 2, QTableWidgetItem(f"{frag_count}"))
                        self.effects_table.setItem(i, 3, QTableWidgetItem("count"))
            
            elif payload_type == "Smoke":
                self.effects_table.setRowCount(2)
                self.effects_table.setItem(0, 0, QTableWidgetItem("Coverage"))
                self.effects_table.setItem(0, 1, QTableWidgetItem("Area"))
                self.effects_table.setItem(0, 2, QTableWidgetItem(f"{mass * 500:.0f}"))
                self.effects_table.setItem(0, 3, QTableWidgetItem("m²"))
                
                self.effects_table.setItem(1, 0, QTableWidgetItem("Duration"))
                self.effects_table.setItem(1, 1, QTableWidgetItem("Time"))
                self.effects_table.setItem(1, 2, QTableWidgetItem(f"{mass * 60:.0f}"))
                self.effects_table.setItem(1, 3, QTableWidgetItem("s"))
            
            elif payload_type == "Illumination":
                self.effects_table.setRowCount(2)
                self.effects_table.setItem(0, 0, QTableWidgetItem("Brightness"))
                self.effects_table.setItem(0, 1, QTableWidgetItem("Intensity"))
                self.effects_table.setItem(0, 2, QTableWidgetItem(f"{mass * 200000:.0f}"))
                self.effects_table.setItem(0, 3, QTableWidgetItem("cd"))
                
                self.effects_table.setItem(1, 0, QTableWidgetItem("Duration"))
                self.effects_table.setItem(1, 1, QTableWidgetItem("Time"))
                self.effects_table.setItem(1, 2, QTableWidgetItem(f"{mass * 30:.0f}"))
                self.effects_table.setItem(1, 3, QTableWidgetItem("s"))
                
        except Exception as e:
            QMessageBox.warning(self, "Payload Error", f"Failed to calculate payload effects: {str(e)}")

    def plot_trajectory(self):
        try:
            if not self.trajectory:
                raise ValueError("No trajectory data to plot")
            
            self.figure.clear()
            
            # Create subplots
            ax1 = self.figure.add_subplot(131)
            ax2 = self.figure.add_subplot(132)
            ax3 = self.figure.add_subplot(133)
            
            # Extract data
            times = [p[0] for p in self.trajectory]
            distances = [p[1] for p in self.trajectory]
            heights = [p[2] for p in self.trajectory]
            velocities = [p[3] for p in self.trajectory]
            drifts = [p[5] for p in self.trajectory]
            
            # Plot 1: Trajectory
            ax1.plot(distances, heights)
            ax1.set_title('Trajectory Profile')
            ax1.set_xlabel('Distance (m)')
            ax1.set_ylabel('Height (m)')
            ax1.grid(True)
            
            # Plot 2: Velocity
            ax2.plot(times, velocities)
            ax2.set_title('Velocity Decay')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.grid(True)
            
            # Plot 3: Wind Drift
            ax3.plot(times, drifts)
            ax3.set_title('Wind Drift')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Drift (m)')
            ax3.grid(True)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Failed to plot trajectory: {str(e)}")

    def export_data(self):
        try:
            if not self.trajectory:
                raise ValueError("No trajectory data to export")
            
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Trajectory Data", "", 
                "JSON Files (*.json);;CSV Files (*.csv)", 
                options=options)
            
            if not filename:
                return
            
            if filename.endswith('.json'):
                data = {
                    'trajectory': self.trajectory,
                    'parameters': {
                        'type': self.type_combo.currentText(),
                        'mass': self.mass_input.value(),
                        'diameter': self.diam_input.value(),
                        'velocity': self.velocity_input.value(),
                        'angle': self.angle_input.value(),
                        'target_distance': getattr(self, 'target_distance', None),
                        'wind_speed': self.wind_speed_input.value(),
                        'wind_dir': self.wind_dir_input.value()
                    }
                }
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                with open(filename, 'w') as f:
                    f.write("Time(s),Distance(m),Height(m),Velocity(m/s),Mach,Drift(m)\n")
                    for point in self.trajectory:
                        f.write(f"{point[0]:.2f},{point[1]:.1f},{point[2]:.1f},{point[3]:.1f},{point[4]:.3f},{point[5]:.1f}\n")
            
            QMessageBox.information(self, "Export Complete", f"Data saved to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = BallisticCalculator()
    calculator.show()
    sys.exit(app.exec_())
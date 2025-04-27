#!/usr/bin/env python3
import sys
import math
import json
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
                             QGroupBox, QDoubleSpinBox, QTextEdit, QMessageBox, QTableWidget,
                             QHeaderView, QTableWidgetItem, QFileDialog, QFrame, QCheckBox,
                             QStackedWidget, QToolButton, QSizePolicy, QSpacerItem)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Constants
GRAVITY = 9.80665  # m/s^2
EARTH_RADIUS = 6371000  # meters
DEG_TO_RAD = math.pi / 180
RAD_TO_DEG = 180 / math.pi
STANDARD_TEMP = 288.15  # K (15°C)
STANDARD_PRESSURE = 101325  # Pa
GAS_CONSTANT = 287.05  # J/(kg·K)
AIR_MOLAR_MASS = 0.0289644  # kg/mol
EARTH_ROTATION_RATE = 7.292115e-5  # rad/s

class TacticalStyle:
    @staticmethod
    def apply(app):
        app.setStyle("Fusion")
        palette = app.palette()
        palette.setColor(palette.Window, QColor(53, 53, 53))
        palette.setColor(palette.WindowText, Qt.white)
        palette.setColor(palette.Base, QColor(35, 35, 35))
        palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(palette.ToolTipBase, Qt.white)
        palette.setColor(palette.ToolTipText, Qt.white)
        palette.setColor(palette.Text, Qt.white)
        palette.setColor(palette.Button, QColor(53, 53, 53))
        palette.setColor(palette.ButtonText, Qt.white)
        palette.setColor(palette.BrightText, Qt.red)
        palette.setColor(palette.Highlight, QColor(142, 45, 45))
        palette.setColor(palette.HighlightedText, Qt.white)
        app.setPalette(palette)

    @staticmethod
    def create_header_label(text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0a830;")
        return label

    @staticmethod
    def create_section_frame():
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 4px; }")
        return frame

class ProjectileDatabase:
    PROJECTILES = {
        "Mortar": {
            "60mm": {"mass": 1.7, "diameter": 60, "length": 300, "drag_model": "M2"},
            "81mm": {"mass": 3.2, "diameter": 81, "length": 400, "drag_model": "M2"},
            "120mm": {"mass": 13.0, "diameter": 120, "length": 800, "drag_model": "M3"}
        },
        "Bullet": {
            "5.56mm": {"mass": 0.004, "diameter": 5.56, "length": 23, "drag_model": "G7"},
            "7.62mm": {"mass": 0.009, "diameter": 7.62, "length": 28, "drag_model": "G7"},
            ".50 BMG": {"mass": 0.042, "diameter": 12.7, "length": 55, "drag_model": "G1"}
        }
    }

    CHARGES = {
        "60mm": ["Charge 0", "Charge 1", "Charge 2"],
        "81mm": ["Charge 0", "Charge 1", "Charge 2", "Charge 3"],
        "120mm": ["Charge 1", "Charge 2", "Charge 3", "Charge 4"]
    }

    CHARGE_DATA = {
        "Charge 0": {"weight": 0, "velocity": 70},
        "Charge 1": {"weight": 100, "velocity": 110},
        "Charge 2": {"weight": 200, "velocity": 150},
        "Charge 3": {"weight": 300, "velocity": 190},
        "Charge 4": {"weight": 400, "velocity": 230}
    }

class MapWidget(QWebEngineView):
    position_selected = pyqtSignal(float, float)  # lat, lon
    trajectory_updated = pyqtSignal(list, list)  # lats, lons

    def __init__(self):
        super().__init__()
        self.page().profile().setHttpUserAgent("Mozilla/5.0")
        self.load_map()
        self.trajectory_lats = []
        self.trajectory_lons = []

    def load_map(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tactical Ballistic Calculator</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <style>
                #map { height: 100%; }
                body { margin: 0; padding: 0; }
                html, body, #map { width: 100%; height: 100%; }
                .legend { padding: 6px 8px; background: rgba(255,255,255,0.8); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; line-height: 18px; }
                .legend i { width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.7; }
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
                
                var fireMarker, targetMarker, trajectoryLine;
                var firePos = null, targetPos = null;
                
                function initMap() {
                    // Add scale control
                    L.control.scale({imperial: false}).addTo(map);
                    
                    // Add legend
                    var legend = L.control({position: 'bottomright'});
                    legend.onAdd = function(map) {
                        var div = L.DomUtil.create('div', 'legend');
                        div.innerHTML = '<i style="background: #ff0000"></i>Firing Position<br>' +
                                       '<i style="background: #0000ff"></i>Target Position<br>' +
                                       '<i style="background: #00ff00"></i>Trajectory';
                        return div;
                    };
                    legend.addTo(map);
                    
                    map.on('click', function(e) {
                        if (!firePos) {
                            firePos = e.latlng;
                            fireMarker = L.marker(firePos, {draggable: true, icon: L.divIcon({className: 'fire-icon', html: '⚡', iconSize: [24, 24]})})
                                .bindPopup("Firing Position")
                                .addTo(map);
                            fireMarker.on('dragend', updateFirePos);
                            external.invoke('fire:' + firePos.lat + ',' + firePos.lng);
                        } else if (!targetPos) {
                            targetPos = e.latlng;
                            targetMarker = L.marker(targetPos, {draggable: true, icon: L.divIcon({className: 'target-icon', html: '🎯', iconSize: [24, 24]})})
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
                    updateTrajectory();
                }
                
                function updateTargetPos() {
                    targetPos = this.getLatLng();
                    external.invoke('target:' + targetPos.lat + ',' + targetPos.lng);
                    updateTrajectory();
                }
                
                function updateTrajectory(lats, lons) {
                    if (trajectoryLine) {
                        map.removeLayer(trajectoryLine);
                    }
                    
                    if (lats && lons && lats.length > 0 && lons.length > 0) {
                        var latlngs = [];
                        for (var i = 0; i < lats.length; i++) {
                            latlngs.push(L.latLng(lats[i], lons[i]));
                        }
                        trajectoryLine = L.polyline(latlngs, {color: 'green', weight: 2}).addTo(map);
                    }
                }
                
                function clearPositions() {
                    if (fireMarker) map.removeLayer(fireMarker);
                    if (targetMarker) map.removeLayer(targetMarker);
                    if (trajectoryLine) map.removeLayer(trajectoryLine);
                    firePos = null;
                    targetPos = null;
                    external.invoke('clear_positions');
                }
                
                initMap();
            </script>
        </body>
        </html>
        """
        self.setHtml(html, QUrl("about:blank"))
        self.page().javaScriptConsoleMessage = lambda level, message, line, source: print(f"JS: {message}")

    def update_trajectory(self, lats, lons):
        if lats and lons:
            self.trajectory_lats = lats
            self.trajectory_lons = lons
            js = f"updateTrajectory({json.dumps(lats)}, {json.dumps(lons)})"
            self.page().runJavaScript(js)

    def clear_positions(self):
        self.page().runJavaScript("clearPositions()")

class BallisticCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tactical Ballistic Calculator")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize systems
        self.projectile = None
        self.environment = None
        self.trajectory = []
        self.target_distance = 0
        self.target_bearing = 0
        self.elevation_diff = 0
        self.fire_pos = None
        self.target_pos = None
        self.calculating = False
        
        # Setup UI
        self.init_ui()
        
        # Default values
        self.type_combo.setCurrentText("Mortar")
        self.caliber_combo.setCurrentText("81mm")
        self.update_projectile_type("Mortar")

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Left panel (controls) - 35% width
        control_panel = QWidget()
        control_panel.setMaximumWidth(500)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        # Title and mode selector
        title_layout = QHBoxLayout()
        title_label = QLabel("TACTICAL BALLISTIC CALCULATOR")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f0a830;")
        title_layout.addWidget(title_label)
        
        # Create tabs for controls
        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { padding: 8px; }")
        
        # Input Tab
        input_tab = self.create_input_tab()
        tabs.addTab(input_tab, "Firing Solution")
        
        # Results Tab
        results_tab = self.create_results_tab()
        tabs.addTab(results_tab, "Trajectory Data")
        
        # Payload Tab
        payload_tab = self.create_payload_tab()
        tabs.addTab(payload_tab, "Payload Effects")
        
        # Environment Tab
        env_tab = self.create_environment_tab()
        tabs.addTab(env_tab, "Environment")
        
        control_layout.addLayout(title_layout)
        control_layout.addWidget(tabs)
        control_panel.setLayout(control_layout)
        
        # Right panel (map and plots) - 65% width
        display_panel = QWidget()
        display_layout = QVBoxLayout()
        display_layout.setContentsMargins(5, 5, 5, 5)
        
        # Map widget
        self.map_widget = MapWidget()
        self.map_widget.position_selected.connect(self.handle_map_position)
        self.map_widget.trajectory_updated.connect(self.map_widget.update_trajectory)
        display_layout.addWidget(self.map_widget, stretch=2)
        
        # Map controls
        map_control_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear Positions")
        clear_btn.setStyleSheet("background-color: #444; padding: 5px;")
        clear_btn.clicked.connect(self.clear_map_positions)
        map_control_layout.addWidget(clear_btn)
        
        map_control_layout.addStretch()
        
        # Altitude controls
        alt_group = QGroupBox("Altitude Settings")
        alt_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        alt_layout = QHBoxLayout()
        
        alt_layout.addWidget(QLabel("Firing Alt (m):"))
        self.fire_alt_input = QDoubleSpinBox()
        self.fire_alt_input.setRange(-100, 5000)
        self.fire_alt_input.setValue(100)
        self.fire_alt_input.setSingleStep(10)
        alt_layout.addWidget(self.fire_alt_input)
        
        alt_layout.addWidget(QLabel("Target Alt (m):"))
        self.target_alt_input = QDoubleSpinBox()
        self.target_alt_input.setRange(-100, 5000)
        self.target_alt_input.setValue(100)
        self.target_alt_input.setSingleStep(10)
        alt_layout.addWidget(self.target_alt_input)
        
        alt_group.setLayout(alt_layout)
        map_control_layout.addWidget(alt_group)
        
        display_layout.addLayout(map_control_layout)
        
        # Plot widget
        self.figure = Figure(facecolor='#353535')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #353535;")
        display_layout.addWidget(self.canvas, stretch=1)
        
        display_panel.setLayout(display_layout)
        
        # Combine panels
        main_layout.addWidget(control_panel, stretch=1)
        main_layout.addWidget(display_panel, stretch=3)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready", 5000)
        
        # Menu bar
        self.create_menu_bar()

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_action = file_menu.addAction('New Calculation')
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.reset_calculation)
        
        export_action = file_menu.addAction('Export Data')
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_data)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Exit')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        toggle_dark = view_menu.addAction('Toggle Dark Mode')
        toggle_dark.triggered.connect(self.toggle_dark_mode)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

    def create_input_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Projectile Selection
        proj_group = QGroupBox("Projectile Configuration")
        proj_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        proj_layout = QVBoxLayout()
        
        # Type and Caliber
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Mortar", "Bullet"])
        self.type_combo.currentTextChanged.connect(self.update_projectile_type)
        type_layout.addWidget(self.type_combo)
        
        type_layout.addWidget(QLabel("Caliber:"))
        self.caliber_combo = QComboBox()
        self.caliber_combo.currentTextChanged.connect(self.update_caliber)
        type_layout.addWidget(self.caliber_combo)
        
        proj_layout.addLayout(type_layout)
        
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
        self.charge_combo.currentTextChanged.connect(self.update_velocity_from_charge)
        self.charge_layout.addWidget(self.charge_combo)
        proj_layout.addLayout(self.charge_layout)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        # Projectile Parameters
        params_group = QGroupBox("Projectile Parameters")
        params_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        params_layout = QVBoxLayout()
        
        # Mass
        mass_layout = QHBoxLayout()
        mass_layout.addWidget(QLabel("Mass (kg):"))
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.001, 100)
        self.mass_input.setValue(3.2)
        self.mass_input.setSingleStep(0.1)
        mass_layout.addWidget(self.mass_input)
        params_layout.addLayout(mass_layout)
        
        # Diameter
        diam_layout = QHBoxLayout()
        diam_layout.addWidget(QLabel("Diameter (mm):"))
        self.diam_input = QDoubleSpinBox()
        self.diam_input.setRange(1, 300)
        self.diam_input.setValue(82)
        self.diam_input.setSingleStep(1)
        diam_layout.addWidget(self.diam_input)
        params_layout.addLayout(diam_layout)
        
        # Length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Length (mm):"))
        self.length_input = QDoubleSpinBox()
        self.length_input.setRange(1, 2000)
        self.length_input.setValue(400)
        self.length_input.setSingleStep(1)
        length_layout.addWidget(self.length_input)
        params_layout.addLayout(length_layout)
        
        # Velocity
        vel_layout = QHBoxLayout()
        vel_layout.addWidget(QLabel("Velocity (m/s):"))
        self.velocity_input = QDoubleSpinBox()
        self.velocity_input.setRange(1, 2000)
        self.velocity_input.setValue(200)
        self.velocity_input.setSingleStep(5)
        vel_layout.addWidget(self.velocity_input)
        params_layout.addLayout(vel_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Launch Parameters
        launch_group = QGroupBox("Launch Parameters")
        launch_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        launch_layout = QVBoxLayout()
        
        # Angle
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Elevation Angle (deg):"))
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(0, 90)
        self.angle_input.setValue(45)
        self.angle_input.setSingleStep(0.5)
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
        
        # Calculation buttons
        btn_layout = QHBoxLayout()
        
        self.calc_btn = QPushButton("Calculate Trajectory")
        self.calc_btn.setStyleSheet("background-color: #4CAF50; padding: 8px; font-weight: bold;")
        self.calc_btn.clicked.connect(self.calculate_solution)
        btn_layout.addWidget(self.calc_btn)
        
        self.opt_btn = QPushButton("Optimize Angle")
        self.opt_btn.setStyleSheet("background-color: #2196F3; padding: 8px;")
        self.opt_btn.clicked.connect(self.optimize_angle)
        btn_layout.addWidget(self.opt_btn)
        
        layout.addLayout(btn_layout)
        
        tab.setLayout(layout)
        return tab

    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Summary Group
        summary_group = QGroupBox("Firing Solution Summary")
        summary_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        summary_layout = QVBoxLayout()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("background-color: #252525;")
        summary_layout.addWidget(self.summary_text)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Trajectory Data
        data_group = QGroupBox("Trajectory Data")
        data_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        data_layout = QVBoxLayout()
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(7)
        self.data_table.setHorizontalHeaderLabels(["Time (s)", "Distance (m)", "Height (m)", "Velocity (m/s)", "Mach", "Wind Drift (m)", "TOF (s)"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setStyleSheet("QTableView { background-color: #252525; }")
        data_layout.addWidget(self.data_table)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        tab.setLayout(layout)
        return tab

    def create_payload_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Payload Parameters
        payload_group = QGroupBox("Payload Configuration")
        payload_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        payload_layout = QVBoxLayout()
        
        # Payload Type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Payload Type:"))
        self.payload_type_combo = QComboBox()
        self.payload_type_combo.addItems(["HE", "Smoke", "Illumination", "Training"])
        self.payload_type_combo.currentTextChanged.connect(self.update_payload_effects)
        type_layout.addWidget(self.payload_type_combo)
        payload_layout.addLayout(type_layout)
        
        # Payload Mass
        mass_layout = QHBoxLayout()
        mass_layout.addWidget(QLabel("Mass (kg):"))
        self.payload_mass_input = QDoubleSpinBox()
        self.payload_mass_input.setRange(0, 50)
        self.payload_mass_input.setValue(1.5)
        self.payload_mass_input.setSingleStep(0.1)
        self.payload_mass_input.valueChanged.connect(self.update_payload_effects)
        mass_layout.addWidget(self.payload_mass_input)
        payload_layout.addLayout(mass_layout)
        
        # Effects Table
        self.effects_table = QTableWidget()
        self.effects_table.setColumnCount(4)
        self.effects_table.setHorizontalHeaderLabels(["Range (m)", "Effect", "Value", "Units"])
        self.effects_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.effects_table.setStyleSheet("QTableView { background-color: #252525; }")
        payload_layout.addWidget(self.effects_table)
        
        payload_group.setLayout(payload_layout)
        layout.addWidget(payload_group)
        
        tab.setLayout(layout)
        return tab

    def create_environment_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Atmospheric Conditions
        atmo_group = QGroupBox("Atmospheric Conditions")
        atmo_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        atmo_layout = QVBoxLayout()
        
        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature (°C):"))
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(-50, 50)
        self.temp_input.setValue(15)
        self.temp_input.setSingleStep(1)
        temp_layout.addWidget(self.temp_input)
        atmo_layout.addLayout(temp_layout)
        
        # Pressure
        press_layout = QHBoxLayout()
        press_layout.addWidget(QLabel("Pressure (hPa):"))
        self.pressure_input = QDoubleSpinBox()
        self.pressure_input.setRange(800, 1100)
        self.pressure_input.setValue(1013.25)
        self.pressure_input.setSingleStep(1)
        press_layout.addWidget(self.pressure_input)
        atmo_layout.addLayout(press_layout)
        
        # Humidity
        hum_layout = QHBoxLayout()
        hum_layout.addWidget(QLabel("Humidity (%):"))
        self.humidity_input = QDoubleSpinBox()
        self.humidity_input.setRange(0, 100)
        self.humidity_input.setValue(50)
        self.humidity_input.setSingleStep(5)
        hum_layout.addWidget(self.humidity_input)
        atmo_layout.addLayout(hum_layout)
        
        atmo_group.setLayout(atmo_layout)
        layout.addWidget(atmo_group)
        
        # Wind Conditions
        wind_group = QGroupBox("Wind Conditions")
        wind_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        wind_layout = QVBoxLayout()
        
        # Wind Speed
        wind_speed_layout = QHBoxLayout()
        wind_speed_layout.addWidget(QLabel("Wind Speed (m/s):"))
        self.wind_speed_input = QDoubleSpinBox()
        self.wind_speed_input.setRange(0, 50)
        self.wind_speed_input.setValue(5)
        self.wind_speed_input.setSingleStep(0.5)
        wind_speed_layout.addWidget(self.wind_speed_input)
        wind_layout.addLayout(wind_speed_layout)
        
        # Wind Direction
        wind_dir_layout = QHBoxLayout()
        wind_dir_layout.addWidget(QLabel("Wind Direction (deg):"))
        self.wind_dir_input = QDoubleSpinBox()
        self.wind_dir_input.setRange(0, 359)
        self.wind_dir_input.setValue(90)
        self.wind_dir_input.setSingleStep(1)
        wind_dir_layout.addWidget(self.wind_dir_input)
        wind_layout.addLayout(wind_dir_layout)
        
        wind_group.setLayout(wind_layout)
        layout.addWidget(wind_group)
        
        # Advanced Options
        adv_group = QGroupBox("Advanced Options")
        adv_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 3px; margin-top: 6px; }")
        adv_layout = QVBoxLayout()
        
        # Coriolis Effect
        coriolis_layout = QHBoxLayout()
        self.coriolis_check = QCheckBox("Enable Coriolis Effect")
        self.coriolis_check.setChecked(True)
        coriolis_layout.addWidget(self.coriolis_check)
        adv_layout.addLayout(coriolis_layout)
        
        # Air Density
        density_layout = QHBoxLayout()
        self.density_check = QCheckBox("Use Altitude-Adjusted Air Density")
        self.density_check.setChecked(True)
        density_layout.addWidget(self.density_check)
        adv_layout.addLayout(density_layout)
        
        adv_group.setLayout(adv_layout)
        layout.addWidget(adv_group)
        
        tab.setLayout(layout)
        return tab

    def update_projectile_type(self, proj_type):
        try:
            is_mortar = (proj_type == "Mortar")
            self.drag_model_combo.clear()
            
            # Update caliber options
            self.caliber_combo.clear()
            calibers = list(ProjectileDatabase.PROJECTILES[proj_type].keys())
            self.caliber_combo.addItems(calibers)
            
            if is_mortar:
                self.drag_model_combo.addItems(["M1", "M2", "M3"])
                self.charge_combo.setEnabled(True)
                self.velocity_input.setEnabled(False)
                self.mass_input.setRange(1, 50)
                self.diam_input.setRange(60, 120)
            else:
                self.drag_model_combo.addItems(["G1", "G7"])
                self.charge_combo.setEnabled(False)
                self.velocity_input.setEnabled(True)
                self.mass_input.setRange(0.001, 1)
                self.diam_input.setRange(5, 20)
                self.velocity_input.setValue(800)  # Typical bullet velocity
            
            # Update with first caliber's default values
            self.update_caliber(self.caliber_combo.currentText())
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update projectile type: {str(e)}")

    def update_caliber(self, caliber):
        try:
            proj_type = self.type_combo.currentText()
            if not proj_type or not caliber:
                return
                
            # Get default values from database
            defaults = ProjectileDatabase.PROJECTILES[proj_type][caliber]
            self.mass_input.setValue(defaults["mass"])
            self.diam_input.setValue(defaults["diameter"])
            self.length_input.setValue(defaults["length"])
            self.drag_model_combo.setCurrentText(defaults["drag_model"])
            
            # Update charge options for mortars
            if proj_type == "Mortar":
                self.charge_combo.clear()
                charges = ProjectileDatabase.CHARGES.get(caliber, [])
                self.charge_combo.addItems(charges)
                if charges:
                    self.charge_combo.setCurrentIndex(1)  # Default to Charge 1
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update caliber: {str(e)}")

    def update_velocity_from_charge(self, charge):
        try:
            if self.type_combo.currentText() != "Mortar":
                return
                
            charge_data = ProjectileDatabase.CHARGE_DATA.get(charge, {"weight": 0, "velocity": 200})
            self.velocity_input.setValue(charge_data["velocity"])
            self.charge_weight_label.setText(f"{charge_data['weight']}g")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update velocity: {str(e)}")

    def handle_map_position(self, lat, lon):
        try:
            if self.fire_pos is None:
                self.fire_pos = (lat, lon)
                self.status_bar.showMessage(f"Firing position set to: {lat:.6f}, {lon:.6f}", 5000)
            else:
                self.target_pos = (lat, lon)
                self.calculate_geospatial()
                self.status_bar.showMessage(f"Target position set to: {lat:.6f}, {lon:.6f}", 5000)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to handle position: {str(e)}")

    def clear_map_positions(self):
        self.fire_pos = None
        self.target_pos = None
        self.map_widget.clear_positions()
        self.status_bar.showMessage("Positions cleared", 3000)

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
                f"Firing Alt: {alt1:.1f}m\n"
                f"Target Alt: {alt2:.1f}m\n"
                f"Wind: {self.wind_speed_input.value()} m/s from {self.wind_dir_input.value()}°"
            )
            
            # Auto-calculate if mortar
            if self.type_combo.currentText() == "Mortar":
                self.calculate_solution()
                
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Geospatial calculation failed: {str(e)}")

    def calculate_solution(self):
        if self.calculating:
            return
            
        self.calculating = True
        self.calc_btn.setEnabled(False)
        self.status_bar.showMessage("Calculating trajectory...")
        
        # Use a timer to allow UI to update before starting calculation
        QTimer.singleShot(100, self._calculate_solution)

    def _calculate_solution(self):
        try:
            # Get projectile parameters
            mass = self.mass_input.value()
            diameter = self.diam_input.value() / 1000  # Convert mm to m
            length = self.length_input.value() / 1000  # Convert mm to m
            drag_model = self.drag_model_combo.currentText()
            velocity = self.velocity_input.value()
            wind_speed = self.wind_speed_input.value()
            wind_dir = self.wind_dir_input.value()
            angle = self.angle_input.value()
            
            # Get environmental parameters
            temp = self.temp_input.value() + 273.15  # Convert to Kelvin
            pressure = self.pressure_input.value() * 100  # Convert hPa to Pa
            humidity = self.humidity_input.value()
            fire_alt = self.fire_alt_input.value()
            
            # Calculate trajectory with all effects
            self.trajectory = self.calculate_trajectory(
                velocity,
                angle,
                mass, diameter, length, drag_model,
                wind_speed, wind_dir,
                self.elevation_diff,
                temp, pressure, humidity, fire_alt
            )
            
            if not self.trajectory:
                raise ValueError("Trajectory calculation failed - no valid solution")
            
            # Update displays
            self.update_results_display()
            self.update_payload_effects()
            self.plot_trajectory()
            
            # Update map with trajectory
            if self.fire_pos:
                self.update_map_trajectory()
            
            self.status_bar.showMessage("Calculation complete", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Failed to calculate solution: {str(e)}")
            self.status_bar.showMessage("Calculation failed", 3000)
        finally:
            self.calculating = False
            self.calc_btn.setEnabled(True)

    def optimize_angle(self):
        if not hasattr(self, 'target_distance') or self.target_distance <= 0:
            QMessageBox.warning(self, "Warning", "No target distance available. Set target position first.")
            return
            
        self.status_bar.showMessage("Optimizing firing angle...")
        self.opt_btn.setEnabled(False)
        QTimer.singleShot(100, self._optimize_angle)

    def _optimize_angle(self):
        try:
            velocity = self.velocity_input.value()
            distance = self.target_distance
            elevation_diff = self.elevation_diff
            
            best_angle = 45
            best_error = float('inf')
            
            # Test angles between 35-85° in 0.5° steps
            for test_angle in range(350, 851, 5):
                angle = test_angle / 10
                traj = self.calculate_trajectory(
                    velocity, angle,
                    self.mass_input.value(),
                    self.diam_input.value() / 1000,
                    self.length_input.value() / 1000,
                    self.drag_model_combo.currentText(),
                    self.wind_speed_input.value(),
                    self.wind_dir_input.value(),
                    elevation_diff,
                    self.temp_input.value() + 273.15,
                    self.pressure_input.value() * 100,
                    self.humidity_input.value(),
                    self.fire_alt_input.value()
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
            
            self.angle_input.setValue(min(85, max(35, best_angle)))
            self.status_bar.showMessage(f"Optimal angle found: {best_angle:.1f}°", 3000)
            
            # Auto-calculate with new angle
            self.calculate_solution()
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Angle optimization failed: {str(e)}")
            self.status_bar.showMessage("Angle optimization failed", 3000)
        finally:
            self.opt_btn.setEnabled(True)

    def calculate_trajectory(self, velocity, angle, mass, diameter, length, drag_model, 
                           wind_speed, wind_dir, elevation_diff=0,
                           temp=288.15, pressure=101325, humidity=50, altitude=0):
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
            
            # Coriolis parameters
            latitude = self.fire_pos[0] * DEG_TO_RAD if self.fire_pos else 45 * DEG_TO_RAD
            coriolis_enabled = self.coriolis_check.isChecked()
            
            trajectory = []
            time = 0
            
            max_steps = 10000  # Prevent infinite loops
            step_count = 0
            
            while y >= -elevation_diff and step_count < max_steps:
                step_count += 1
                
                # Calculate current altitude for air density
                current_alt = altitude + y
                
                # Get air density (kg/m³)
                if self.density_check.isChecked():
                    rho = self.calculate_air_density(temp, pressure, humidity, current_alt)
                else:
                    rho = 1.225  # Standard sea level density
                
                # Relative velocity (accounting for wind)
                v_rel_x = vx - wind_x
                v_rel_y = vy - wind_y
                v_rel = math.hypot(v_rel_x, v_rel_y)
                mach = v_rel / self.calculate_sound_speed(temp)
                
                # Get drag coefficient
                cd = self.get_drag_coefficient(drag_model, mach)
                
                # Drag force
                drag_force = 0.5 * rho * v_rel**2 * cd * area
                
                # Acceleration components
                if v_rel > 0:
                    ax = -(drag_force * v_rel_x) / (mass * v_rel)
                    ay = -g - (drag_force * v_rel_y) / (mass * v_rel)
                else:
                    ax = 0
                    ay = -g
                
                # Add Coriolis effect if enabled
                if coriolis_enabled:
                    coriolis_x = 2 * EARTH_ROTATION_RATE * (vy * math.sin(latitude))
                    coriolis_y = 2 * EARTH_ROTATION_RATE * (-vx * math.sin(latitude))
                    ax += coriolis_x
                    ay += coriolis_y
                
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
                    mach, drift,
                    time_step
                ))
            
            return trajectory
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Trajectory calculation failed: {str(e)}")
            return []

    def calculate_air_density(self, temp, pressure, humidity, altitude):
        """Calculate air density using barometric formula with humidity correction"""
        try:
            # Temperature lapse rate (K/m)
            lapse_rate = 0.0065
            
            # Calculate temperature at altitude
            temp_alt = temp - lapse_rate * altitude
            
            # Calculate pressure at altitude (barometric formula)
            pressure_alt = pressure * (1 - (lapse_rate * altitude) / temp) ** (GRAVITY * AIR_MOLAR_MASS / (GAS_CONSTANT * lapse_rate))
            
            # Calculate vapor pressure
            saturation_pressure = 610.78 * math.exp((17.2694 * (temp_alt - 273.15)) / (temp_alt - 35.86))
            vapor_pressure = (humidity / 100) * saturation_pressure
            
            # Calculate air density (kg/m³)
            rho = ((pressure_alt - vapor_pressure) / (GAS_CONSTANT * temp_alt) + 
                  vapor_pressure / (461.495 * temp_alt))
            
            return max(0.1, rho)  # Prevent unrealistic values
            
        except:
            return 1.225  # Fallback to standard sea level density

    def calculate_sound_speed(self, temp):
        """Calculate speed of sound in air at given temperature (K)"""
        return 20.05 * math.sqrt(temp)

    def get_drag_coefficient(self, model, mach):
        """Get drag coefficient for given model and Mach number"""
        try:
            # Extended drag coefficient tables
            models = {
                'M1': [(0.0, 0.65), (0.8, 0.63), (1.0, 0.59), (1.2, 0.57), (2.0, 0.55)],
                'M2': [(0.0, 0.55), (0.8, 0.53), (1.0, 0.49), (1.2, 0.47), (2.0, 0.45)],
                'M3': [(0.0, 0.45), (0.8, 0.43), (1.0, 0.39), (1.2, 0.37), (2.0, 0.35)],
                'G1': [(0.0, 0.26), (0.8, 0.25), (1.0, 0.23), (1.2, 0.29), (2.0, 0.24), (3.0, 0.24)],
                'G7': [(0.0, 0.12), (0.8, 0.12), (1.0, 0.12), (1.2, 0.18), (2.0, 0.18), (3.0, 0.18)]
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
                f"Projectile: {self.type_combo.currentText()} {self.caliber_combo.currentText()}\n"
                f"Mass: {self.mass_input.value():.2f} kg\n"
                f"Diameter: {self.diam_input.value():.1f} mm\n"
                f"Velocity: {self.velocity_input.value():.1f} m/s\n"
                f"Angle: {self.angle_input.value():.1f}°\n\n"
                f"Flight Time: {flight_time:.1f} s\n"
                f"Impact Distance: {impact_dist:.1f} m\n"
                f"Impact Velocity: {impact_vel:.1f} m/s (Mach {impact_vel/self.calculate_sound_speed(self.temp_input.value()+273.15):.2f})\n"
                f"Max Height: {max_height:.1f} m\n"
                f"Wind Drift: {wind_drift:.1f} m\n"
            )
            
            if hasattr(self, 'target_distance'):
                summary += (
                    f"\nTarget Distance: {self.target_distance:.1f} m\n"
                    f"Error: {abs(impact_dist - self.target_distance):.1f} m\n"
                    f"Bearing: {self.target_bearing:.1f}°\n"
                )
            
            # Environmental conditions
            summary += (
                f"\nEnvironment:\n"
                f"Temperature: {self.temp_input.value():.1f}°C\n"
                f"Pressure: {self.pressure_input.value():.1f} hPa\n"
                f"Humidity: {self.humidity_input.value():.0f}%\n"
                f"Wind: {self.wind_speed_input.value():.1f} m/s from {self.wind_dir_input.value():.0f}°\n"
                f"Firing Alt: {self.fire_alt_input.value():.0f} m\n"
                f"Target Alt: {self.target_alt_input.value():.0f} m\n"
            )
            
            self.summary_text.setPlainText(summary)
            
            # Trajectory table
            self.data_table.setRowCount(min(100, len(self.trajectory)))  # Limit to 100 points for performance
            step = max(1, len(self.trajectory) // 100)
            
            for i, idx in enumerate(range(0, len(self.trajectory), step)):
                point = self.trajectory[idx]
                self.data_table.setItem(i, 0, QTableWidgetItem(f"{point[0]:.2f}"))
                self.data_table.setItem(i, 1, QTableWidgetItem(f"{point[1]:.1f}"))
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{point[2]:.1f}"))
                self.data_table.setItem(i, 3, QTableWidgetItem(f"{point[3]:.1f}"))
                self.data_table.setItem(i, 4, QTableWidgetItem(f"{point[4]:.2f}"))
                self.data_table.setItem(i, 5, QTableWidgetItem(f"{point[5]:.1f}"))
                self.data_table.setItem(i, 6, QTableWidgetItem(f"{point[6]:.3f}"))
                
        except Exception as e:
            QMessageBox.warning(self, "Display Error", f"Failed to update results: {str(e)}")

    def update_payload_effects(self):
        try:
            payload_type = self.payload_type_combo.currentText()
            mass = self.payload_mass_input.value()
            
            if payload_type == "HE":
                frag_count = min(1000, mass * 1500)
                lethal_radius = math.sqrt(mass) * 15
                crater_depth = math.pow(mass, 1/3) * 2
                
                ranges = [0, 100, 200, 500, 1000, 1500, 2000]
                self.effects_table.setRowCount(len(ranges))
                
                for i, r in enumerate(ranges):
                    if r == 0:
                        self.effects_table.setItem(i, 0, QTableWidgetItem("Ground Zero"))
                        self.effects_table.setItem(i, 1, QTableWidgetItem("Crater Depth"))
                        self.effects_table.setItem(i, 2, QTableWidgetItem(f"{crater_depth:.1f}"))
                        self.effects_table.setItem(i, 3, QTableWidgetItem("m"))
                    else:
                        eff_radius = max(0, lethal_radius - r*0.1)
                        self.effects_table.setItem(i, 0, QTableWidgetItem(f"{r}"))
                        self.effects_table.setItem(i, 1, QTableWidgetItem("Lethal Radius"))
                        self.effects_table.setItem(i, 2, QTableWidgetItem(f"{eff_radius:.1f}"))
                        self.effects_table.setItem(i, 3, QTableWidgetItem("m"))
                    
                    if i == 1:
                        self.effects_table.setItem(i, 1, QTableWidgetItem("Fragments"))
                        self.effects_table.setItem(i, 2, QTableWidgetItem(f"{frag_count}"))
                        self.effects_table.setItem(i, 3, QTableWidgetItem("count"))
            
            elif payload_type == "Smoke":
                self.effects_table.setRowCount(3)
                self.effects_table.setItem(0, 0, QTableWidgetItem("Coverage"))
                self.effects_table.setItem(0, 1, QTableWidgetItem("Area"))
                self.effects_table.setItem(0, 2, QTableWidgetItem(f"{mass * 500:.0f}"))
                self.effects_table.setItem(0, 3, QTableWidgetItem("m²"))
                
                self.effects_table.setItem(1, 0, QTableWidgetItem("Duration"))
                self.effects_table.setItem(1, 1, QTableWidgetItem("Time"))
                self.effects_table.setItem(1, 2, QTableWidgetItem(f"{mass * 60:.0f}"))
                self.effects_table.setItem(1, 3, QTableWidgetItem("s"))
                
                self.effects_table.setItem(2, 0, QTableWidgetItem("Wind Effect"))
                self.effects_table.setItem(2, 1, QTableWidgetItem("Drift Rate"))
                self.effects_table.setItem(2, 2, QTableWidgetItem(f"{mass * 0.2:.1f}"))
                self.effects_table.setItem(2, 3, QTableWidgetItem("m/s"))
            
            elif payload_type == "Illumination":
                self.effects_table.setRowCount(3)
                self.effects_table.setItem(0, 0, QTableWidgetItem("Brightness"))
                self.effects_table.setItem(0, 1, QTableWidgetItem("Intensity"))
                self.effects_table.setItem(0, 2, QTableWidgetItem(f"{mass * 200000:.0f}"))
                self.effects_table.setItem(0, 3, QTableWidgetItem("cd"))
                
                self.effects_table.setItem(1, 0, QTableWidgetItem("Duration"))
                self.effects_table.setItem(1, 1, QTableWidgetItem("Time"))
                self.effects_table.setItem(1, 2, QTableWidgetItem(f"{mass * 30:.0f}"))
                self.effects_table.setItem(1, 3, QTableWidgetItem("s"))
                
                self.effects_table.setItem(2, 0, QTableWidgetItem("Descent Rate"))
                self.effects_table.setItem(2, 1, QTableWidgetItem("Speed"))
                self.effects_table.setItem(2, 2, QTableWidgetItem(f"{5 + mass:.1f}"))
                self.effects_table.setItem(2, 3, QTableWidgetItem("m/s"))
                
        except Exception as e:
            QMessageBox.warning(self, "Payload Error", f"Failed to calculate payload effects: {str(e)}")

    def plot_trajectory(self):
        try:
            if not self.trajectory:
                raise ValueError("No trajectory data to plot")
            
            self.figure.clear()
            self.figure.set_facecolor('#353535')
            
            # Create subplots
            ax1 = self.figure.add_subplot(131)
            ax2 = self.figure.add_subplot(132)
            ax3 = self.figure.add_subplot(133)
            
            # Set dark background for plots
            for ax in [ax1, ax2, ax3]:
                ax.set_facecolor('#252525')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('#f0a830')
                ax.grid(True, color='#444')
                ax.spines['bottom'].set_color('#666')
                ax.spines['top'].set_color('#666')
                ax.spines['right'].set_color('#666')
                ax.spines['left'].set_color('#666')
            
            # Extract data
            times = [p[0] for p in self.trajectory]
            distances = [p[1] for p in self.trajectory]
            heights = [p[2] for p in self.trajectory]
            velocities = [p[3] for p in self.trajectory]
            drifts = [p[5] for p in self.trajectory]
            
            # Plot 1: Trajectory
            ax1.plot(distances, heights, color='#4CAF50')
            ax1.set_title('Trajectory Profile')
            ax1.set_xlabel('Distance (m)')
            ax1.set_ylabel('Height (m)')
            
            # Plot 2: Velocity
            ax2.plot(times, velocities, color='#2196F3')
            ax2.set_title('Velocity Decay')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (m/s)')
            
            # Plot 3: Wind Drift
            ax3.plot(times, drifts, color='#FF9800')
            ax3.set_title('Wind Drift')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Drift (m)')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Failed to plot trajectory: {str(e)}")

    def update_map_trajectory(self):
        """Convert trajectory coordinates to lat/lon for map display"""
        try:
            if not self.fire_pos or not self.trajectory:
                return
                
            lat1, lon1 = self.fire_pos
            bearing = self.target_bearing * DEG_TO_RAD
            earth_radius = 6371000  # meters
            
            # Convert trajectory distances to lat/lon coordinates
            lats = []
            lons = []
            
            for point in self.trajectory:
                distance = point[1]
                
                # Simple flat-earth approximation (good for short distances)
                lat2 = lat1 + (distance * math.sin(bearing)) / earth_radius * RAD_TO_DEG
                lon2 = lon1 + (distance * math.cos(bearing)) / (earth_radius * math.cos(lat1 * DEG_TO_RAD)) * RAD_TO_DEG
                
                lats.append(lat2)
                lons.append(lon2)
            
            self.map_widget.update_trajectory(lats, lons)
            
        except Exception as e:
            QMessageBox.warning(self, "Map Error", f"Failed to update map trajectory: {str(e)}")

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
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'software': "Tactical Ballistic Calculator",
                        'version': "2.0"
                    },
                    'trajectory': self.trajectory,
                    'parameters': {
                        'type': self.type_combo.currentText(),
                        'caliber': self.caliber_combo.currentText(),
                        'mass': self.mass_input.value(),
                        'diameter': self.diam_input.value(),
                        'length': self.length_input.value(),
                        'velocity': self.velocity_input.value(),
                        'angle': self.angle_input.value(),
                        'drag_model': self.drag_model_combo.currentText(),
                        'target_distance': getattr(self, 'target_distance', None),
                        'target_bearing': getattr(self, 'target_bearing', None),
                        'elevation_diff': getattr(self, 'elevation_diff', None),
                        'wind_speed': self.wind_speed_input.value(),
                        'wind_dir': self.wind_dir_input.value(),
                        'temperature': self.temp_input.value(),
                        'pressure': self.pressure_input.value(),
                        'humidity': self.humidity_input.value(),
                        'fire_alt': self.fire_alt_input.value(),
                        'target_alt': self.target_alt_input.value(),
                        'coriolis': self.coriolis_check.isChecked(),
                        'altitude_density': self.density_check.isChecked()
                    },
                    'payload': {
                        'type': self.payload_type_combo.currentText(),
                        'mass': self.payload_mass_input.value()
                    }
                }
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                with open(filename, 'w') as f:
                    f.write("Time(s),Distance(m),Height(m),Velocity(m/s),Mach,Drift(m),TOF(s)\n")
                    for point in self.trajectory:
                        f.write(f"{point[0]:.2f},{point[1]:.1f},{point[2]:.1f},{point[3]:.1f},{point[4]:.3f},{point[5]:.1f},{point[6]:.3f}\n")
            
            self.status_bar.showMessage(f"Data exported to {filename}", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")

    def reset_calculation(self):
        self.trajectory = []
        self.fire_pos = None
        self.target_pos = None
        self.target_distance = 0
        self.target_bearing = 0
        self.elevation_diff = 0
        
        self.summary_text.clear()
        self.data_table.setRowCount(0)
        self.effects_table.setRowCount(0)
        self.figure.clear()
        self.canvas.draw()
        self.map_widget.clear_positions()
        
        self.status_bar.showMessage("Calculation reset", 3000)

    def toggle_dark_mode(self):
        current_bg = self.palette().color(self.backgroundRole())
        if current_bg.lightness() > 127:
            TacticalStyle.apply(QApplication.instance())
        else:
            QApplication.instance().setStyle("Fusion")
            palette = QApplication.instance().palette()
            palette.setColor(palette.Window, Qt.white)
            palette.setColor(palette.WindowText, Qt.black)
            QApplication.instance().setPalette(palette)

    def show_about(self):
        about_text = """
        <h2>Tactical Ballistic Calculator</h2>
        <p>Version 2.0</p>
        <p>A comprehensive ballistic trajectory calculator for tactical applications.</p>
        <p>Features:</p>
        <ul>
            <li>Mortar and bullet trajectory calculations</li>
            <li>Multiple drag models (M1-M3, G1, G7)</li>
            <li>Environmental effects (wind, temperature, pressure)</li>
            <li>Coriolis effect for long-range shots</li>
            <li>Interactive map for position selection</li>
            <li>Payload effects simulation</li>
        </ul>
        <p>© 2023 Tactical Applications Group</p>
        """
        QMessageBox.about(self, "About", about_text)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 'Exit',
            'Are you sure you want to quit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    TacticalStyle.apply(app)
    
    # Set window icon if available
    try:
        app.setWindowIcon(QIcon('icon.png'))
    except:
        pass
    
    calculator = BallisticCalculator()
    calculator.show()
    sys.exit(app.exec_())
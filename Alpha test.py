#!/usr/bin/env python3
"""
Advanced Ballistic Calculator v5.0 — 3D Trajectory Engine

Comprehensive 3D physics simulation for bullets, rockets, and mortars with:
  - Full 3D state vector [x, y, z, vx, vy, vz]
  - 4th-order Runge-Kutta with adaptive stepping
  - Cubic-spline Mach-dependent drag (G1/G7/rocket/mortar, 19-point tables)
  - CIPM-2007 moist-air density (direct ideal gas law)
  - Altitude-corrected gravity g(h)
  - Full 3-component Coriolis in xyz downrange frame
  - Gyroscopic spin drift (3D cross-range, McCoy model)
  - Rocket thrust with linear propellant burnoff
  - ICAO standard atmosphere toggle
  - Ballistic coefficient (BC) alternative input mode
  - Crosswind modeling (u/v components)
  - Categorized presets from external JSON file
  - Input validation with visual feedback
  - QTableWidget trajectory data browser
  - Live preview graph on input tab
  - Dark/light theme toggle
  - Keyboard shortcuts
  - Save/load scenario .json files
  - Plot: side view, top view, speed/energy vs time

References: CIPM-2007, ISO 9613-1, IERS 2010, McCoy Modern Exterior Ballistics
"""
import sys, math, csv, json, logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
    QFileDialog, QMessageBox, QInputDialog, QScrollArea, QSizePolicy,
    QGridLayout, QTableWidget, QTableWidgetItem,
    QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ── constants ──────────────────────────────────────────────────────────────
GRAVITY = 9.80665
EARTH_RADIUS = 6371000
EARTH_ROTATION_RATE = 7.292115e-5
SOUND_SPEED_0C = 331.3
ABSOLUTE_ZERO = 273.15
GC_DRY = 287.058
GC_VAPOR = 461.495
MAX_FLIGHT_TIME = 600.0
MIN_HEIGHT = 1e-9
SPIN_DRIFT_COEFF = 0.03
PROGRESS_STRIDE = 500
PREV_TRAJ_LIMIT = 3
PRESETS_FILE = Path.home() / ".ballistic_presets.json"
SCENARIOS_DIR = Path.home() / ".ballistic_scenarios"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── drag curves ────────────────────────────────────────────────────────────
try:
    from scipy.interpolate import UnivariateSpline as _Spline
    _HAVE_SPLINE = True
except ImportError:
    _HAVE_SPLINE = False

class DragCurve:
    __slots__ = ('name', 'machs', 'cds', '_spline')
    def __init__(self, name: str, machs: List[float], cds: List[float]):
        self.name = name
        self.machs = machs
        self.cds = cds
        if _HAVE_SPLINE and len(machs) > 3:
            self._spline = _Spline(machs, cds, s=0, ext=3)
        else:
            self._spline = None
    def __call__(self, mach: float) -> float:
        if self._spline:
            return float(max(0.01, self._spline(float(mach))))
        # linear fallback
        m, c = self.machs, self.cds
        if mach <= m[0]: return c[0]
        if mach >= m[-1]: return c[-1]
        for i in range(1, len(m)):
            if mach <= m[i]:
                f = (mach - m[i-1]) / (m[i] - m[i-1])
                return c[i-1] + f * (c[i] - c[i-1])
        return c[-1]

class DragModels:
    G1 = DragCurve("G1", [
        0.00,0.20,0.40,0.60,0.80,0.90,0.95,1.00,1.05,1.10,
        1.20,1.40,1.60,1.80,2.00,2.50,3.00,4.00,5.00
    ], [
        0.263,0.261,0.273,0.295,0.342,0.377,0.407,0.440,0.466,0.476,
        0.480,0.471,0.451,0.441,0.433,0.415,0.396,0.356,0.324
    ])
    G7 = DragCurve("G7", [
        0.00,0.20,0.40,0.60,0.80,0.90,0.95,1.00,1.05,1.10,
        1.20,1.40,1.60,1.80,2.00,2.50,3.00,4.00,5.00
    ], [
        0.119,0.120,0.144,0.183,0.235,0.273,0.308,0.345,0.389,0.422,
        0.436,0.424,0.400,0.382,0.365,0.330,0.300,0.255,0.214
    ])
    ROCKET = DragCurve("Rocket", [0,0.8,1.0,1.5,2.0,3.0,5.0],
                       [0.25,0.30,0.35,0.40,0.45,0.50,0.50])
    MORTAR = DragCurve("Mortar", [0,0.8,1.0,1.5,3.0,5.0],
                       [0.40,0.45,0.50,0.55,0.55,0.55])
    @staticmethod
    def get(name: str) -> DragCurve:
        return {'G1':DragModels.G1,'G7':DragModels.G7,
                'rocket':DragModels.ROCKET,'mortar':DragModels.MORTAR}.get(name,DragModels.G7)

# ── utilities ──────────────────────────────────────────────────────────────
def sos(t_C: float) -> float:
    return SOUND_SPEED_0C * math.sqrt(1 + t_C / ABSOLUTE_ZERO)

def svp(t_C: float) -> float:
    if t_C >= 0:
        ex = ((18.678 - t_C/234.5)*t_C)/(234.5+t_C)
        return 6.1121 * math.exp(ex)
    ex = ((23.036 - t_C/333.7)*t_C)/(333.7+t_C)
    return 6.1115 * math.exp(ex)

def icao_atmosphere(alt_m: float) -> Tuple[float, float]:
    """Return (temp_C, pressure_hPa) at altitude using ICAO std atmosphere."""
    h = max(-500, min(20000, alt_m))
    if h <= 11000:
        T_K = 288.15 - 0.0065 * h
        p_Pa = 101325 * (T_K/288.15)**(9.80665/(287.058*0.0065))
    else:
        T_K = 216.65
        p_Pa = 22632 * math.exp(-9.80665*(h-11000)/(287.058*216.65))
    return (T_K - ABSOLUTE_ZERO, p_Pa/100)

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters (Haversine)."""
    R = 6371000
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (azimuth) from point 1 to point 2, 0–360°."""
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360

# ── physical models ────────────────────────────────────────────────────────
@dataclass
class Projectile:
    mass: float; diameter: float; drag_model: str; velocity: float
    projectile_type: str; thrust_curve: Dict[float,float]; burn_time: float
    propellant_mass: float; spin_rate: float = 0.0; bc: float = 0.0
    i_ratio: float = SPIN_DRIFT_COEFF
    def __post_init__(self):
        self.area = math.pi * (self.diameter/2)**2
        self.init_mass = self.mass
        self.curve = DragModels.get(self.drag_model)
        if self.projectile_type == 'rocket' and self.burn_time > 0 and self.propellant_mass <= 0:
            self.propellant_mass = 0.10 * self.mass
    def cd(self, v: float, sos_l: float) -> float:
        m = v / max(sos_l, 1)
        cd_ref = self.curve(m)
        if self.bc > 0 and self.projectile_type == 'bullet':
            return cd_ref / self.bc
        return cd_ref
    def thrust(self, t: float) -> float:
        if not self.thrust_curve or t > self.burn_time: return 0
        ks = sorted(self.thrust_curve)
        if t <= ks[0]: return self.thrust_curve[ks[0]]
        if t >= ks[-1]: return self.thrust_curve[ks[-1]]
        for i in range(1, len(ks)):
            if t <= ks[i]:
                f = (t-ks[i-1])/(ks[i]-ks[i-1]) if ks[i]!=ks[i-1] else 0
                return self.thrust_curve[ks[i-1]]+f*(self.thrust_curve[ks[i]]-self.thrust_curve[ks[i-1]])
        return 0
    def mass_at(self, t: float) -> float:
        if self.projectile_type != 'rocket' or self.burn_time <= 0: return self.mass
        if t >= self.burn_time: return self.init_mass - self.propellant_mass
        return self.init_mass - self.propellant_mass * t/self.burn_time

@dataclass
class Environment:
    altitude: float; temperature: float; pressure: float; humidity: float
    wind_speed: float; wind_angle: float; coriolis: bool; latitude: float
    azimuth: float = 90.0; use_icao: bool = False
    def __post_init__(self):
        if self.use_icao:
            self.temperature, self.pressure = icao_atmosphere(self.altitude)
        T = self.temperature + ABSOLUTE_ZERO
        sp = svp(self.temperature)*100
        pt = self.pressure*100; pv = self.humidity/100*sp; pd = pt-pv
        self.air_density = max(pd/(GC_DRY*T)+pv/(GC_VAPOR*T), MIN_HEIGHT)
        self.sos = sos(self.temperature)
        self.g = GRAVITY*(EARTH_RADIUS/(EARTH_RADIUS+self.altitude))**2
        # wind in x (downrange) and z (crossrange) components
        wr = math.radians(self.wind_angle - self.azimuth)
        self.wind_x = self.wind_speed * math.cos(wr)
        self.wind_z = self.wind_speed * math.sin(wr)

# ── 3D trajectory calculator ──────────────────────────────────────────────
class TrajectoryCalculator:
    """
    State: [x, y, z, vx, vy, vz]
      x – downrange (horizontal, azimuth direction)
      y – height above launch
      z – cross-range (right-hand positive)
    Forces: drag, gravity, thrust, Coriolis (3D), spin drift (3D), wind
    """
    def __init__(self, proj: Projectile, env: Environment,
                 launch_angle: float, enable_spin: bool = False,
                 rifling_twist: float = 10.0):
        self.p = proj; self.e = env
        self.la = math.radians(launch_angle)
        self.spin = enable_spin; self.twist = rifling_twist
        self.g = env.g
        # Coriolis params
        lr = math.radians(env.latitude); ar = math.radians(env.azimuth)
        self.w = EARTH_ROTATION_RATE
        self.wy = self.w * math.cos(lr)   # horizontal (North)
        self.wz = self.w * math.sin(lr)   # vertical (Up)
        self.sa, self.ca = math.sin(ar), math.cos(ar)
        # spin rate
        if enable_spin and proj.projectile_type == 'bullet':
            tm = rifling_twist * 0.0254
            self.spin_rps = (proj.velocity*2*math.pi)/tm if tm>0 else 0
        else:
            self.spin_rps = 0.0

    def derivs(self, s: List[float], t: float) -> List[float]:
        x, y, z, vx, vy, vz = s
        # wind-relative velocity
        ur = vx - self.e.wind_x
        vr = vz - self.e.wind_z
        wr = vy
        spd = math.hypot(ur, vr, wr) or 1e-9
        ax, ay, az = 0.0, -self.g, 0.0
        # drag
        cd = self.p.cd(spd, self.e.sos)
        Fd = 0.5 * self.e.air_density * spd**2 * cd * self.p.area
        m = self.p.mass_at(t)
        if m > 1e-9:
            ax -= Fd * ur / (m * spd)
            ay -= Fd * wr / (m * spd)
            az -= Fd * vr / (m * spd)
        # thrust
        if self.p.projectile_type == 'rocket' and t < self.p.burn_time:
            th = self.p.thrust(t)
            if m > 1e-9 and th > 0:
                ta = math.atan2(vy, vx) if spd > 1e-6 else self.la
                ax += th * math.cos(ta) / m
                ay += th * math.sin(ta) / m
        # Coriolis (full 3D in xyz frame)
        if self.e.coriolis:
            # a_cor_x = -2*wy*vy*sa - 2*wz*vz
            # a_cor_y =  2*wy*(vx*sa + vz*ca)
            # a_cor_z =  2*wz*vx - 2*wy*vy*ca
            ax += -2*self.wy*vy*self.sa - 2*self.wz*vz
            ay +=  2*self.wy*(vx*self.sa + vz*self.ca)
            az +=  2*self.wz*vx - 2*self.wy*vy*self.ca
        # spin drift (cross-range acceleration)
        if self.spin and self.spin_rps > 0 and spd > 10:
            a_spin = (self.p.i_ratio * self.spin_rps * self.p.diameter /
                      max(spd, 1) * self.g * (vx / max(spd, 1)))
            az += a_spin  # right-hand twist -> +z
        return [vx, vy, vz, ax, ay, az]

    def integrate(self, dt_max: float = 0.05, dt_min: float = 0.001,
                  progress: Callable = None, stop: Callable = None
                  ) -> List[Tuple[float,float,float,float,float,float,float,float]]:
        s = [0,0,0,
             self.p.velocity*math.cos(self.la), self.p.velocity*math.sin(self.la), 0]
        traj, tm, st = [], 0.0, 0
        while s[1] >= 0 and tm < MAX_FLIGHT_TIME:
            if stop and stop(): break
            sp = math.hypot(s[3], s[4], s[5])
            traj.append((s[0], s[1], s[2], tm, s[3], s[4], s[5], sp))
            vel = math.hypot(s[3], s[4], s[5])
            h = max(dt_min, min(dt_max*100/max(10,vel), dt_max*4))
            k1 = self.derivs(s, tm)
            s2 = [s[i]+.5*h*k1[i] for i in range(6)]
            k2 = self.derivs(s2, tm+.5*h)
            s3 = [s[i]+.5*h*k2[i] for i in range(6)]
            k3 = self.derivs(s3, tm+.5*h)
            s4 = [s[i]+h*k3[i] for i in range(6)]
            k4 = self.derivs(s4, tm+h)
            for i in range(6):
                s[i] += (h/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i])
            tm += h; st += 1
            if progress and st % PROGRESS_STRIDE == 0:
                progress(min(99, int(100*tm/MAX_FLIGHT_TIME)))
        sp = math.hypot(s[3], s[4], s[5])
        traj.append((s[0], max(s[1],MIN_HEIGHT), s[2], tm, s[3], s[4], s[5], sp))
        if progress: progress(100)
        return traj

    def spin_drift_total(self, flight_t: float) -> float:
        if not self.spin or self.spin_rps <= 0: return 0
        v_avg = self.p.velocity * 0.6
        if v_avg < 1: return 0
        a = (self.p.i_ratio * self.spin_rps * self.p.diameter / v_avg * self.g)
        return 0.5 * a * flight_t**2

# ── preset manager ─────────────────────────────────────────────────────────
class PresetManager:
    @staticmethod
    def load_defaults() -> Dict:
        p = Path(__file__).parent / "default_presets.json"
        if p.exists():
            try: return json.loads(p.read_text())
            except: pass
        return {}
    @classmethod
    def load_all(cls) -> Dict:
        merged = cls.load_defaults()
        if PRESETS_FILE.exists():
            try: merged.update(json.loads(PRESETS_FILE.read_text()))
            except: pass
        return merged
    @staticmethod
    def save_user(presets: Dict) -> None:
        defaults = PresetManager.load_defaults()
        user = {k:v for k,v in presets.items() if k not in defaults}
        PRESETS_FILE.write_text(json.dumps(user, indent=2))

SETTINGS_FILE = Path.home() / ".ballistic_settings.json"

class SettingsManager:
    DEFAULTS = {
        "api_provider": "Open-Meteo",
        "api_base_url": "https://api.open-meteo.com/v1/forecast",
        "api_key": "",
        "api_timeout": 10,
        "auto_fetch_weather": False,
        "default_lat": 45.0,
        "default_lon": 0.0,
        "default_drag": "G7",
        "dark_theme": False,
    }
    @classmethod
    def load(cls) -> Dict:
        if SETTINGS_FILE.exists():
            try:
                s = json.loads(SETTINGS_FILE.read_text())
                return {**cls.DEFAULTS, **s}
            except: pass
        return dict(cls.DEFAULTS)
    @staticmethod
    def save(settings: Dict) -> None:
        try: SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
        except: pass

# ── worker thread ──────────────────────────────────────────────────────────
class CalcThread(QThread):
    done = pyqtSignal(list); err = pyqtSignal(str); prog = pyqtSignal(int)
    def __init__(self, proj, env, angle, spin, twist):
        super().__init__()
        self.p = proj; self.e = env; self.a = angle; self.s = spin; self.t = twist
        self._run = True
    def run(self):
        try:
            c = TrajectoryCalculator(self.p,self.e,self.a,self.s,self.t)
            traj = c.integrate(progress=self.prog.emit, stop=lambda: not self._run)
            self.done.emit(traj)
        except Exception as e:
            logger.exception("calc"); self.err.emit(str(e))
    def stop(self): self._run = False

class WeatherFetcher(QThread):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    def __init__(self, lat: float, lon: float, base_url: str, timeout: int):
        super().__init__()
        self.lat = lat; self.lon = lon
        self.url = (f"{base_url}?latitude={lat}&longitude={lon}"
                    f"&current=temperature_2m,relative_humidity_2m,pressure_msl,"
                    f"wind_speed_10m,wind_direction_10m")
        self.timeout = timeout
    def run(self):
        try:
            from urllib.request import urlopen
            resp = urlopen(self.url, timeout=self.timeout)
            data = json.loads(resp.read().decode())
            c = data.get("current", {})
            self.done.emit({
                "temp": c.get("temperature_2m"),
                "humidity": c.get("relative_humidity_2m"),
                "pressure": c.get("pressure_msl"),
                "wind_speed": c.get("wind_speed_10m"),
                "wind_dir": c.get("wind_direction_10m"),
            })
        except Exception as e:
            self.failed.emit(str(e))

# ═══════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════
DARK_QSS = """
QMainWindow,QWidget{background-color:#2b2b2b;color:#ddd}
QGroupBox{border:1px solid #555;border-radius:4px;margin-top:10px;padding-top:15px;color:#ccc}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;color:#aaa}
QPushButton{background-color:#3c3c3c;border:1px solid #666;border-radius:3px;padding:5px 10px;color:#ddd}
QPushButton:hover{background-color:#505050}
QLineEdit,QTextEdit,QComboBox,QSpinBox,QDoubleSpinBox,QTableWidget{
    background-color:#3c3c3c;border:1px solid #555;border-radius:3px;padding:3px;color:#ddd;
    selection-background-color:#4a90d9
}
QTabWidget::pane{border:1px solid #555}
QHeaderView::section{background-color:#3c3c3c;color:#ddd;border:1px solid #555}
QProgressBar{background:#3c3c3c;border:1px solid #555;border-radius:3px;text-align:center;color:#ddd}
QProgressBar::chunk{background:#4a90d9}
QCheckBox{color:#ddd}
QLabel{color:#ddd}
"""

LIGHT_QSS = """
QMainWindow{background-color:#f5f5f5;font-family:Segoe UI,Arial}
QGroupBox{border:1px solid #ccc;border-radius:4px;margin-top:10px;padding-top:15px}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;color:#555}
QPushButton{background-color:#e0e0e0;border:1px solid #aaa;border-radius:3px;padding:5px 10px}
QPushButton:hover{background-color:#d0d0d0}
QLineEdit,QTextEdit,QComboBox,QSpinBox,QDoubleSpinBox,QTableWidget{
    border:1px solid #bbb;border-radius:3px;padding:3px
}
QTabWidget::pane{border:1px solid #ccc;margin-top:-1px}
QHeaderView::section{background-color:#e0e0e0;border:1px solid #bbb}
QProgressBar{background:#e0e0e0;border:1px solid #bbb;border-radius:3px;text-align:center}
QProgressBar::chunk{background:#4a90d9}
"""

class BallisticCalc(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ballistic Calculator v5.0 — 3D")
        self.setGeometry(80,80,1400,960)
        self.setMinimumSize(1000,700)
        self.trajectory = []; self.prev_trajs = []
        self.presets = PresetManager.load_all()
        self.settings = SettingsManager.load()
        self.thread = None; self._weather_thread = None
        self._dark = self.settings.get("dark_theme", False)
        self._icao_connected = False
        self._valid_states = {}
        self.init_ui()
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(DARK_QSS if self._dark else LIGHT_QSS)

    def toggle_theme(self):
        self._dark = not self._dark
        self.apply_theme()

    # ── input tab ──────────────────────────────────────────────────────
    def init_ui(self):
        cw = QWidget(); ml = QVBoxLayout(cw); ml.setContentsMargins(4,4,4,4)
        tabs = QTabWidget()
        tabs.addTab(self._inp_tab(), "Input")
        tabs.addTab(self._res_tab(), "Results")
        tabs.addTab(self._plot_tab(), "Graph")
        tabs.addTab(self._adv_tab(), "Advanced")
        tabs.addTab(self._set_tab(), "Settings")
        ml.addWidget(tabs, 1)
        self.setCentralWidget(cw)
        # menu
        mb = self.menuBar()
        fm = mb.addMenu('File')
        fm.addAction('Export CSV', self.export_csv)
        fm.addAction('Save Scenario', self.save_scenario)
        fm.addAction('Load Scenario', self.load_scenario)
        fm.addAction('Exit', self.close)
        hm = mb.addMenu('Help')
        hm.addAction('About', self.show_about)
        # status
        self.sb = self.statusBar()
        self.pb = QProgressBar(); self.pb.setFixedSize(200,16); self.pb.setVisible(False)
        self.sb.addPermanentWidget(self.pb)

    def _inp_tab(self):
        tab = QWidget(); sc = QScrollArea(); sc.setWidgetResizable(True)
        ct = QWidget(); lo = QVBoxLayout(ct)

        # type
        tg = QGroupBox("Projectile Type")
        tl = QHBoxLayout()
        self.type_cb = QComboBox(); self.type_cb.addItems(["Bullet","Rocket","Mortar"])
        self.type_cb.currentTextChanged.connect(self._on_type_change)
        tl.addWidget(QLabel("Type:")); tl.addWidget(self.type_cb)
        tg.setLayout(tl); lo.addWidget(tg)

        # presets (categorized)
        pg = QGroupBox("Ammunition Presets")
        pl = QHBoxLayout()
        self.preset_cb = QComboBox()
        self._rebuild_presets()
        self.preset_cb.currentIndexChanged.connect(self._on_preset)
        pl.addWidget(self.preset_cb)
        sv = QPushButton("Save"); sv.clicked.connect(self._save_preset)
        pl.addWidget(sv)
        pg.setLayout(pl); lo.addWidget(pg)

        # projectile params
        pj = QGroupBox("Projectile Parameters")
        pjl = QGridLayout()
        self.mass_sb = self._mk_spin(0.1,2e6,10,1,"Mass (g)")
        pjl.addWidget(QLabel("Mass (g):"),0,0); pjl.addWidget(self.mass_sb,0,1)
        self.diam_sb = self._mk_spin(0.1,1000,7.62,0.1,"Diameter (mm)")
        pjl.addWidget(QLabel("Diameter (mm):"),0,2); pjl.addWidget(self.diam_sb,0,3)
        self.drag_cb = QComboBox(); self.drag_cb.addItems(['G1','G7','rocket','mortar'])
        pjl.addWidget(QLabel("Drag Model:"),1,0); pjl.addWidget(self.drag_cb,1,1)
        self.bc_sb = self._mk_spin(0,5,0,0.01,"Ballistic Coefficient (0 = disabled)")
        self.bc_sb.setDecimals(3)
        pjl.addWidget(QLabel("BC (G1):"),1,2); pjl.addWidget(self.bc_sb,1,3)
        pj.setLayout(pjl); lo.addWidget(pj)

        # rocket group
        self.rg = QGroupBox("Rocket Parameters")
        rl = QVBoxLayout()
        bl = QHBoxLayout(); bl.addWidget(QLabel("Burn (s):"))
        self.burn_sb = self._mk_spin(0,60,1,0.1)
        bl.addWidget(self.burn_sb); rl.addLayout(bl)
        tl2 = QHBoxLayout(); tl2.addWidget(QLabel("Peak Thrust (N):"))
        self.thrust_sb = self._mk_spin(0,5e5,1000,100)
        tl2.addWidget(self.thrust_sb); rl.addLayout(tl2)
        pl2 = QHBoxLayout(); pl2.addWidget(QLabel("Propellant (g):"))
        self.prop_sb = self._mk_spin(0,1e6,0,10,"0 = 10% auto")
        pl2.addWidget(self.prop_sb); rl.addLayout(pl2)
        self.rg.setLayout(rl); self.rg.setVisible(False); lo.addWidget(self.rg)

        # launch
        lg = QGroupBox("Launch Parameters")
        ll = QGridLayout()
        self.vel_sb = self._mk_spin(1,5000,800,10,"Muzzle velocity")
        ll.addWidget(QLabel("Muzzle Vel (m/s):"),0,0); ll.addWidget(self.vel_sb,0,1)
        self.ang_sb = self._mk_spin(-90,90,15,1)
        self.ang_sb.setDecimals(4)
        ll.addWidget(QLabel("Angle (deg):"),0,2); ll.addWidget(self.ang_sb,0,3)
        lg.setLayout(ll); lo.addWidget(lg)

        # environment
        eg = QGroupBox("Environmental Parameters")
        el = QGridLayout(); el.setHorizontalSpacing(10)
        self.icao_cb = QCheckBox("ICAO Standard Atmosphere")
        self.icao_cb.stateChanged.connect(self._on_icao)
        el.addWidget(self.icao_cb,0,0,1,2)
        el.addWidget(QLabel("Altitude (m):"),1,0)
        self.alt_sb = self._mk_spin(-500,20000,0,10)
        el.addWidget(self.alt_sb,1,1)
        el.addWidget(QLabel("Temp (°C):"),1,2)
        self.temp_sb = self._mk_spin(-80,60,15,1)
        el.addWidget(self.temp_sb,1,3)
        el.addWidget(QLabel("Pressure (hPa):"),2,0)
        self.pres_sb = self._mk_spin(500,1100,1013.25,1)
        el.addWidget(self.pres_sb,2,1)
        el.addWidget(QLabel("Humidity (%):"),2,2)
        self.hum_sb = self._mk_spin(0,100,50,5)
        el.addWidget(self.hum_sb,2,3)
        el.addWidget(QLabel("Wind (m/s):"),3,0)
        self.ws_sb = self._mk_spin(0,100,0,0.5)
        el.addWidget(self.ws_sb,3,1)
        el.addWidget(QLabel("Wind From (°):"),3,2)
        self.wa_sb = QSpinBox(); self.wa_sb.setRange(0,359); self.wa_sb.setValue(0)
        el.addWidget(self.wa_sb,3,3)
        self.cor_cb = QCheckBox("Coriolis"); el.addWidget(self.cor_cb,4,0)
        el.addWidget(QLabel("Lat (°):"),4,1)
        self.lat_sb = self._mk_spin(-90,90,45,1); self.lat_sb.setEnabled(False)
        el.addWidget(self.lat_sb,4,2)
        el.addWidget(QLabel("Azim (°):"),4,3)
        self.az_sb = self._mk_spin(0,359,90,1); self.az_sb.setEnabled(False)
        el.addWidget(self.az_sb,4,4)
        def _tc(e):
            self.lat_sb.setEnabled(e); self.az_sb.setEnabled(e)
        self.cor_cb.stateChanged.connect(lambda: _tc(self.cor_cb.isChecked()))

        self.weather_btn = QPushButton("Get Live Weather")
        self.weather_btn.setToolTip("Fetch weather from API using GPS coordinates")
        self.weather_btn.clicked.connect(self._fetch_weather)
        self.weather_lbl = QLabel("")
        el.addWidget(self.weather_btn,5,0,1,2)
        el.addWidget(self.weather_lbl,5,2,1,2)
        eg.setLayout(el); lo.addWidget(eg)

        # GPS targeting
        gg = QGroupBox("GPS Targeting")
        gl = QGridLayout()
        gl.addWidget(QLabel("Launch Lat:"),0,0)
        self.gps_la_lat = self._mk_spin(-90,90,0,0.01,"Decimal degrees (e.g. 48.8566)")
        self.gps_la_lat.setDecimals(6)
        gl.addWidget(self.gps_la_lat,0,1)
        gl.addWidget(QLabel("Lon:"),0,2)
        self.gps_la_lon = self._mk_spin(-180,180,0,0.01)
        self.gps_la_lon.setDecimals(6)
        gl.addWidget(self.gps_la_lon,0,3)
        gl.addWidget(QLabel("Target Lat:"),1,0)
        self.gps_tg_lat = self._mk_spin(-90,90,0,0.01)
        self.gps_tg_lat.setDecimals(6)
        gl.addWidget(self.gps_tg_lat,1,1)
        gl.addWidget(QLabel("Lon:"),1,2)
        self.gps_tg_lon = self._mk_spin(-180,180,0,0.01)
        self.gps_tg_lon.setDecimals(6)
        gl.addWidget(self.gps_tg_lon,1,3)
        self.gps_btn = QPushButton("Compute Firing Solution")
        self.gps_btn.clicked.connect(self._on_gps)
        self.gps_info = QLabel("")
        gl.addWidget(self.gps_btn,2,0,1,2)
        gl.addWidget(self.gps_info,2,2,1,2)
        gg.setLayout(gl); lo.addWidget(gg)

        # calc button
        bl2 = QHBoxLayout()
        self.calc_btn = QPushButton("Calculate Trajectory")
        self.calc_btn.setMinimumHeight(38)
        self.calc_btn.setStyleSheet(
            "QPushButton{background:#4a90d9;color:white;font-weight:bold;"
            "font-size:14px;border:none;border-radius:5px;padding:8px 24px}"
            "QPushButton:hover{background:#357abd}"
            "QPushButton:disabled{background:#888}")
        self.calc_btn.clicked.connect(self._on_calc)
        bl2.addStretch(); bl2.addWidget(self.calc_btn); bl2.addStretch()
        lo.addLayout(bl2)

        # live preview
        self._preview_fig = Figure(figsize=(6,2),dpi=80)
        self._preview_ax = self._preview_fig.add_subplot(111)
        self._preview_canvas = FigureCanvas(self._preview_fig)
        self._preview_canvas.setFixedHeight(140)
        self._preview_ax.set_title("Live Preview")
        lo.addWidget(self._preview_canvas)

        sc.setWidget(ct); tab.setLayout(QVBoxLayout()); tab.layout().addWidget(sc)
        return tab

    def _mk_spin(self, lo, hi, val, step, tip=""):
        s = QDoubleSpinBox(); s.setRange(lo,hi); s.setValue(val); s.setSingleStep(step)
        if tip: s.setToolTip(tip)
        s.valueChanged.connect(self._validate_inputs)
        return s

    def _validate_inputs(self):
        """Quick sanity check — show hint in status bar for edge cases."""
        m = self.mass_sb.value(); v = self.vel_sb.value()
        d = self.diam_sb.value(); t = self.type_cb.currentText().lower()
        if t == 'bullet' and m > 200:
            self.sb.showMessage("Bullet mass >200g? Verify input", 3000)
        elif t == 'mortar' and v > 600:
            self.sb.showMessage("Mortar velocity >600m/s? Verify input", 3000)
        elif v < 50:
            self.sb.showMessage("Very low velocity — trajectory may be short", 3000)
        elif d < 2:
            self.sb.showMessage("Diameter <2mm? Verify input", 3000)

    def _rebuild_presets(self):
        self.preset_cb.clear()
        self.preset_cb.addItem("— Custom —", "CUSTOM")
        cats = {'bullet':[], 'rocket':[], 'mortar':[]}
        for k,v in self.presets.items():
            t = v.get('type','bullet')
            cats.setdefault(t,[]).append(k)
        for cat, items in [('Bullet',cats['bullet']),('Rocket',cats['rocket']),('Mortar',cats['mortar'])]:
            if not items: continue
            for name in sorted(items):
                d = f"[{cat}] {name}"
                self.preset_cb.addItem(d, name)

    def _on_preset(self, idx):
        key = self.preset_cb.itemData(idx)
        if not key or key == "CUSTOM": return
        p = self.presets.get(key)
        if not p: return
        self.mass_sb.blockSignals(True)
        self.diam_sb.blockSignals(True)
        self.vel_sb.blockSignals(True)
        self.mass_sb.setValue(p['mass'])
        self.diam_sb.setValue(p['diameter'])
        t = p.get('type','bullet')
        self.type_cb.setCurrentText(t.capitalize())
        self.drag_cb.setCurrentText(p.get('drag_model','G7'))
        self.vel_sb.setValue(p['velocity'])
        if 'bc' in p: self.bc_sb.setValue(p['bc'])
        else: self.bc_sb.setValue(0)
        if t == 'rocket':
            self.burn_sb.setValue(p.get('burn_time',1))
            crv = p.get('thrust_curve',{})
            self.thrust_sb.setValue(max(crv.values()) if crv else 1000)
            self.prop_sb.setValue(p.get('propellant_mass',0))
        if t == 'bullet' and hasattr(self,'ir_sb'):
            self.ir_sb.setValue(p.get('i_ratio', SPIN_DRIFT_COEFF))
        self.mass_sb.blockSignals(False)
        self.diam_sb.blockSignals(False)
        self.vel_sb.blockSignals(False)

    def _save_preset(self):
        n, ok = QInputDialog.getText(self,"Save Preset","Name:")
        if not (ok and n.strip()): return
        n = n.strip()
        d = {
            'mass': self.mass_sb.value(), 'diameter': self.diam_sb.value(),
            'drag_model': self.drag_cb.currentText(),
            'velocity': self.vel_sb.value(), 'type': self.type_cb.currentText().lower()
        }
        if self.type_cb.currentText().lower() == 'rocket':
            d.update({'burn_time':self.burn_sb.value(),'propellant_mass':self.prop_sb.value(),
                      'thrust_curve':{0:self.thrust_sb.value(),self.burn_sb.value():0}})
        self.presets[n] = d
        self._rebuild_presets()
        PresetManager.save_user(self.presets)
        QMessageBox.information(self,"Saved",f"Preset '{n}' saved.")

    def _on_type_change(self, s):
        t = s.lower()
        self.rg.setVisible(t == 'rocket')
        self.drag_cb.clear()
        if t == 'bullet': self.drag_cb.addItems(['G1','G7'])
        elif t == 'rocket': self.drag_cb.addItems(['rocket'])
        else: self.drag_cb.addItems(['mortar'])

    def _on_icao(self, en):
        en = bool(en)
        self.temp_sb.setEnabled(not en); self.pres_sb.setEnabled(not en)
        if en and not self._icao_connected:
            self.alt_sb.valueChanged.connect(self._update_icao)
            self._icao_connected = True
            self._update_icao()
        elif not en and self._icao_connected:
            try: self.alt_sb.valueChanged.disconnect(self._update_icao)
            except: pass
            self._icao_connected = False

    def _update_icao(self):
        t, p = icao_atmosphere(self.alt_sb.value())
        self.temp_sb.setValue(round(t,1))
        self.pres_sb.setValue(round(p,1))

    def _build_env(self):
        return Environment(
            altitude=self.alt_sb.value(), temperature=self.temp_sb.value(),
            pressure=self.pres_sb.value(), humidity=self.hum_sb.value(),
            wind_speed=self.ws_sb.value(), wind_angle=self.wa_sb.value(),
            coriolis=self.cor_cb.isChecked(), latitude=self.lat_sb.value(),
            azimuth=self.az_sb.value(), use_icao=self.icao_cb.isChecked()
        )

    def _build_proj(self):
        return Projectile(
            mass=self.mass_sb.value()/1000, diameter=self.diam_sb.value()/1000,
            drag_model=self.drag_cb.currentText(), velocity=self.vel_sb.value(),
            projectile_type=self.type_cb.currentText().lower(),
            thrust_curve=self._thrust_curve(),
            burn_time=self.burn_sb.value() if self.type_cb.currentText().lower()=='rocket' else 0,
            propellant_mass=self.prop_sb.value()/1000 if self.type_cb.currentText().lower()=='rocket' else 0,
            bc=self.bc_sb.value()
        )

    def _thrust_curve(self):
        if self.type_cb.currentText().lower() != 'rocket': return {}
        pk = self.preset_cb.currentData()
        if pk and pk != "CUSTOM" and pk in self.presets:
            c = self.presets[pk].get('thrust_curve')
            if c: return {float(k):v for k,v in c.items()}
        return {0:self.thrust_sb.value(), self.burn_sb.value():0}

    def _on_calc(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop(); self.calc_btn.setText("Stopping…"); self.calc_btn.setEnabled(False)
            return
        if self.trajectory:
            self.prev_trajs.append(self.trajectory)
            if len(self.prev_trajs) > PREV_TRAJ_LIMIT: self.prev_trajs.pop(0)
        self.calc_btn.setText("Cancel"); self.pb.setVisible(True); self.pb.setValue(0)
        try:
            env = self._build_env()
            proj = self._build_proj()
            self.thread = CalcThread(proj, env, self.ang_sb.value(),
                                     self.spin_cb.isChecked() if hasattr(self,'spin_cb') else False,
                                     self.twist_sb.value() if hasattr(self,'twist_sb') else 10)
            self.thread.done.connect(self._on_done)
            self.thread.err.connect(self._on_err)
            self.thread.prog.connect(self.pb.setValue)
            self.thread.start()
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))
            self.calc_btn.setText("Calculate Trajectory")

    def _on_done(self, traj):
        self.trajectory = traj
        self.calc_btn.setText("Calculate Trajectory"); self.pb.setVisible(False)
        if traj:
            self._update_results()
            self._plot()
            self._live_preview()
            last = traj[-1]
            self.sb.showMessage(
                f"Range: {last[0]:.0f}m | Cross:{last[2]:.1f}m | "
                f"Time:{last[3]:.1f}s | Impact:{last[7]:.0f}m/s", 10000)
        else:
            QMessageBox.warning(self,"Warning","No data generated.")

    def _on_err(self, msg):
        self.calc_btn.setText("Calculate Trajectory"); self.pb.setVisible(False)
        QMessageBox.critical(self,"Error",f"Calculation failed:\n{msg}")

    def _on_gps(self):
        """Compute firing solution from GPS coordinates."""
        la_lat = self.gps_la_lat.value(); la_lon = self.gps_la_lon.value()
        tg_lat = self.gps_tg_lat.value(); tg_lon = self.gps_tg_lon.value()
        dist = haversine(la_lat, la_lon, tg_lat, tg_lon)
        az = bearing(la_lat, la_lon, tg_lat, tg_lon)
        self.az_sb.setValue(round(az, 1))
        self.zr_sb.setValue(round(dist, 1))
        self.gps_info.setText(f"Range: {dist:.0f}m  Bearing: {az:.1f}°")
        self._zero()
        cur_ang = self.ang_sb.value()
        self.sb.showMessage(
            f"GPS: {dist:.0f}m @ {az:.1f}° → angle {cur_ang:.4f}°", 12000)
        if self.settings.get("auto_fetch_weather", False):
            self._fetch_weather()

    def _fetch_weather(self):
        """Fetch live weather from API for the launch location."""
        lat = self.gps_la_lat.value() if abs(self.gps_la_lat.value()) > 1 else self.settings.get("default_lat", 45)
        lon = self.gps_la_lon.value() if abs(self.gps_la_lon.value()) > 1 else self.settings.get("default_lon", 0)
        base = self.settings.get("api_base_url", "https://api.open-meteo.com/v1/forecast")
        timeout = self.settings.get("api_timeout", 10)
        self.weather_btn.setEnabled(False)
        self.weather_btn.setText("Fetching…")
        self.weather_lbl.setText("Contacting API…")
        self._weather_thread = WeatherFetcher(lat, lon, base, timeout)
        self._weather_thread.done.connect(self._on_weather)
        self._weather_thread.failed.connect(self._on_weather_fail)
        self._weather_thread.start()

    def _on_weather(self, data):
        self.weather_btn.setEnabled(True)
        self.weather_btn.setText("Get Live Weather")
        t = data.get("temp")
        h = data.get("humidity")
        p = data.get("pressure")
        ws = data.get("wind_speed")
        wd = data.get("wind_dir")
        if t is not None: self.temp_sb.setValue(round(t, 1))
        if h is not None: self.hum_sb.setValue(round(h))
        if p is not None: self.pres_sb.setValue(round(p, 1))
        if ws is not None: self.ws_sb.setValue(round(ws, 1))
        if wd is not None: self.wa_sb.setValue(round(wd))
        self.weather_lbl.setText(f"Live ✓ {t:.1f}°C {p:.0f}hPa {ws:.1f}m/s")
        self.sb.showMessage("Weather data applied from API", 5000)

    def _on_weather_fail(self, msg):
        self.weather_btn.setEnabled(True)
        self.weather_btn.setText("Get Live Weather")
        self.weather_lbl.setText("API unavailable — using manual")
        self.sb.showMessage(f"Weather API failed: {msg}", 8000)

    # ── results tab ────────────────────────────────────────────────────
    def _res_tab(self):
        tab = QWidget(); lo = QVBoxLayout(tab); lo.setContentsMargins(4,4,4,4)
        self.summary_txt = QTextEdit(); self.summary_txt.setReadOnly(True)
        self.summary_txt.setFont(QFont("Courier New",10)); self.summary_txt.setMaximumHeight(300)
        lo.addWidget(self.summary_txt)
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ["Time(s)","Range(m)","Height(m)","Cross(m)","Vx","Vy","Vz","Speed"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        lo.addWidget(self.table, 1)
        return tab

    def _update_results(self):
        if not self.trajectory: return
        tr = self.trajectory
        max_h = max(p[1] for p in tr)
        last = tr[-1]
        dist = last[0]; cdist = last[2]; ft = last[3]
        iv = last[7]; ivx,ivy,ivz = last[4],last[5],last[6]
        ia = abs(math.degrees(math.atan2(ivy,ivx)))
        ptype = self.type_cb.currentText().lower()
        mk = self.mass_sb.value()/1000
        mf = mk - self.prop_sb.value()/1000 if ptype=='rocket' and self.prop_sb.value()>0 else mk
        if ptype == 'rocket' and self.prop_sb.value()<=0: mf = mk*0.9
        mf = max(mf,1e-6)
        ie = 0.5*mf*iv**2
        s = (f"PROJECTILE: {self.type_cb.currentText()}  |  "
             f"Mass:{self.mass_sb.value():.0f}g  |  "
             f"Diam:{self.diam_sb.value():.1f}mm  |  "
             f"Drag:{self.drag_cb.currentText()}\n"
             f"Muzzle:{self.vel_sb.value():.0f}m/s @ {self.ang_sb.value():.1f}°")
        if ptype == 'rocket':
            s += f"  |  Burn:{self.burn_sb.value():.1f}s  Thrust:{self.thrust_sb.value():.0f}N"
        s += (f"\nENV: Alt:{self.alt_sb.value():.0f}m  "
              f"T:{self.temp_sb.value():.1f}°C  P:{self.pres_sb.value():.0f}hPa  "
              f"Hum:{self.hum_sb.value():.0f}%  Wind:{self.ws_sb.value():.1f}@{self.wa_sb.value():.0f}°")
        if self.cor_cb.isChecked():
            s += f"  Coriolis:On @{self.lat_sb.value():.1f}° Lat {self.az_sb.value():.0f}° Az"
        s += (f"\nRESULTS:  MaxH:{max_h:.0f}m  "
              f"Range:{dist:.0f}m  Cross:{cdist:.1f}m  "
              f"Time:{ft:.1f}s  Impact:{iv:.0f}m/s @{ia:.1f}°  "
              f"E:{ie/1000:.2f}kJ")
        # spin drift
        if hasattr(self,'spin_cb') and self.spin_cb.isChecked() and ptype == 'bullet':
            twist = self.twist_sb.value()*0.0254 if hasattr(self,'twist_sb') else 0.254
            if twist > 0:
                sr = self.vel_sb.value()*2*math.pi/twist
                va = self.vel_sb.value()*0.6
                gl = GRAVITY*(EARTH_RADIUS/(EARTH_RADIUS+self.alt_sb.value()))**2
                asp = SPIN_DRIFT_COEFF*sr*self.diam_sb.value()/1000/max(va,1)*gl
                drift = 0.5*asp*ft**2
                if drift > 0.01: s += f"\nSpin Drift ~{drift:.1f}m (right, est.)"
        self.summary_txt.setPlainText(s)
        # table
        self.table.setRowCount(len(tr))
        fmts = ["{:.3f}","{:.1f}","{:.1f}","{:.2f}","{:.2f}","{:.2f}","{:.2f}","{:.2f}"]
        for i,p in enumerate(tr):
            for j,v in enumerate([p[3],p[0],p[1],p[2],p[4],p[5],p[6],p[7]]):
                self.table.setItem(i,j,QTableWidgetItem(fmts[j].format(v)))
        self.table.resizeColumnsToContents()

    # ── plot tab ───────────────────────────────────────────────────────
    def _plot_tab(self):
        tab = QWidget(); lo = QVBoxLayout(tab); lo.setContentsMargins(4,4,4,4)
        self.fig = Figure(figsize=(10,7),dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tb = NavigationToolbar(self.canvas, self)
        lo.addWidget(tb); lo.addWidget(self.canvas,1)
        bl = QHBoxLayout()
        self.comp_cb = QCheckBox("Overlay previous")
        bl.addWidget(self.comp_cb); bl.addStretch()
        lo.addLayout(bl)
        return tab

    def _plot(self):
        if not self.trajectory: return
        self.fig.clear()
        ax1 = self.fig.add_subplot(221)  # side
        ax2 = self.fig.add_subplot(222)  # top
        ax3 = self.fig.add_subplot(223)  # speed
        ax4 = self.fig.add_subplot(224)  # energy
        tr = self.trajectory
        x = [p[0] for p in tr]; y = [p[1] for p in tr]
        z = [p[2] for p in tr]; t = [p[3] for p in tr]
        sp = [p[7] for p in tr]
        # side
        ax1.plot(x,y,'b-',lw=2,label='Current')
        if self.comp_cb.isChecked():
            col = ['r','g','m']
            for i,pt in enumerate(self.prev_trajs):
                ax1.plot([q[0] for q in pt],[q[1] for q in pt],'--',
                        color=col[i%3],lw=1,alpha=.6)
        # apogee
        mi = y.index(max(y))
        ax1.plot(x[mi],y[mi],'ro',ms=6)
        ax1.annotate(f"Apo {y[mi]:.0f}m",(x[mi],y[mi]),xytext=(5,5),
                     textcoords='offset points',fontsize=8)
        ax1.set_xlabel("Range (m)"); ax1.set_ylabel("Height (m)")
        ax1.grid(True,ls='--',alpha=.4); ax1.set_title("Side View")
        # top
        ax2.plot(x,z,'b-',lw=2)
        ax2.set_xlabel("Range (m)"); ax2.set_ylabel("Cross-Range (m)")
        ax2.grid(True,ls='--',alpha=.4); ax2.set_title("Top View")
        # speed
        ax3.plot(t,sp,'r-',lw=1.5)
        ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Speed (m/s)")
        ax3.grid(True,ls='--',alpha=.4)
        # energy
        mk = self.mass_sb.value()/2000
        ke = [0.5*mk*v**2/1000 for v in sp]
        ax4.plot(t,ke,'g-',lw=1.5)
        ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Kinetic Energy (kJ)")
        ax4.grid(True,ls='--',alpha=.4)
        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    def _live_preview(self):
        if not self.trajectory: return
        self._preview_ax.clear()
        tr = self.trajectory
        x = [p[0] for p in tr]; y = [p[1] for p in tr]
        self._preview_ax.plot(x,y,'b-',lw=2)
        self._preview_ax.set_xlabel("Range (m)"); self._preview_ax.set_ylabel("Ht (m)")
        self._preview_ax.grid(True,ls='--',alpha=.3)
        self._preview_canvas.draw()

    # ── advanced tab ───────────────────────────────────────────────────
    def _adv_tab(self):
        tab = QWidget(); sc = QScrollArea(); sc.setWidgetResizable(True)
        ct = QWidget(); lo = QVBoxLayout(ct)
        # sight
        sg = QGroupBox("Sight")
        sl = QHBoxLayout(); sl.addWidget(QLabel("Sight Height (mm):"))
        self.sh_sb = self._mk_spin(0,200,50,1); sl.addWidget(self.sh_sb); sg.setLayout(sl); lo.addWidget(sg)
        # zero
        zg = QGroupBox("Zeroing")
        zl = QHBoxLayout(); zl.addWidget(QLabel("Zero Range (m):"))
        self.zr_sb = self._mk_spin(10,5000,100,10); zl.addWidget(self.zr_sb)
        zb = QPushButton("Calc Zero Angle"); zb.clicked.connect(self._zero); zl.addWidget(zb)
        zg.setLayout(zl); lo.addWidget(zg)
        # spin
        sg2 = QGroupBox("Spin Drift (3D Cross-Range)")
        sl2 = QHBoxLayout()
        self.spin_cb = QCheckBox("Enable Spin Drift (requires 3D)")
        self.spin_cb.setToolTip("Models gyroscopic precession as lateral acceleration in z-axis")
        sl2.addWidget(self.spin_cb)
        sl2.addWidget(QLabel("Twist (in/rev):"))
        self.twist_sb = self._mk_spin(1,50,10,0.5); self.twist_sb.setEnabled(False)
        sl2.addWidget(self.twist_sb)
        sl2.addWidget(QLabel("I_x/I_y:"))
        self.ir_sb = self._mk_spin(0.001,0.5,0.03,0.005,"Moment of inertia ratio")
        sl2.addWidget(self.ir_sb)
        self.spin_cb.stateChanged.connect(lambda: self.twist_sb.setEnabled(self.spin_cb.isChecked()))
        sg2.setLayout(sl2); lo.addWidget(sg2)
        sc.setWidget(ct); tab.setLayout(QVBoxLayout()); tab.layout().addWidget(sc)
        return tab

    # ── settings tab ───────────────────────────────────────────────────
    def _set_tab(self):
        tab = QWidget(); sc = QScrollArea(); sc.setWidgetResizable(True)
        ct = QWidget(); lo = QVBoxLayout(ct)

        ag = QGroupBox("Weather API")
        al = QGridLayout()
        al.addWidget(QLabel("Provider:"),0,0)
        self.api_provider_cb = QComboBox()
        self.api_provider_cb.addItems(["Open-Meteo","OpenWeatherMap","Custom"])
        self.api_provider_cb.setCurrentText(self.settings.get("api_provider","Open-Meteo"))
        al.addWidget(self.api_provider_cb,0,1)

        al.addWidget(QLabel("API Base URL:"),1,0)
        self.api_url_le = QLineEdit(self.settings.get("api_base_url","https://api.open-meteo.com/v1/forecast"))
        al.addWidget(self.api_url_le,1,1)

        al.addWidget(QLabel("API Key (if required):"),2,0)
        self.api_key_le = QLineEdit(self.settings.get("api_key",""))
        al.addWidget(self.api_key_le,2,1)

        al.addWidget(QLabel("Timeout (s):"),3,0)
        self.api_timeout_sb = QSpinBox(); self.api_timeout_sb.setRange(1,60)
        self.api_timeout_sb.setValue(self.settings.get("api_timeout",10))
        al.addWidget(self.api_timeout_sb,3,1)

        self.auto_fetch_cb = QCheckBox("Auto-fetch weather on GPS compute")
        self.auto_fetch_cb.setChecked(self.settings.get("auto_fetch_weather",False))
        al.addWidget(self.auto_fetch_cb,4,0,1,2)
        ag.setLayout(al); lo.addWidget(ag)

        dg = QGroupBox("Program Defaults")
        dl = QGridLayout()
        dl.addWidget(QLabel("Default Latitude:"),0,0)
        self.def_lat_sb = self._mk_spin(-90,90,self.settings.get("default_lat",45),1)
        dl.addWidget(self.def_lat_sb,0,1)
        dl.addWidget(QLabel("Default Longitude:"),0,2)
        self.def_lon_sb = self._mk_spin(-180,180,self.settings.get("default_lon",0),1)
        dl.addWidget(self.def_lon_sb,0,3)
        dl.addWidget(QLabel("Default Drag Model:"),1,0)
        self.def_drag_cb = QComboBox(); self.def_drag_cb.addItems(["G1","G7","rocket","mortar"])
        self.def_drag_cb.setCurrentText(self.settings.get("default_drag","G7"))
        dl.addWidget(self.def_drag_cb,1,1)
        dl.addWidget(QLabel("Default Theme:"),1,2)
        self.def_theme_cb = QComboBox(); self.def_theme_cb.addItems(["Light","Dark"])
        self.def_theme_cb.setCurrentText("Dark" if self.settings.get("dark_theme",False) else "Light")
        dl.addWidget(self.def_theme_cb,1,3)
        dg.setLayout(dl); lo.addWidget(dg)

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        lo.addWidget(save_btn)

        self._set_status_lbl = QLabel("")
        lo.addWidget(self._set_status_lbl)

        sc.setWidget(ct); tab.setLayout(QVBoxLayout()); tab.layout().addWidget(sc)
        return tab

    def _save_settings(self):
        self.settings["api_provider"] = self.api_provider_cb.currentText()
        self.settings["api_base_url"] = self.api_url_le.text()
        self.settings["api_key"] = self.api_key_le.text()
        self.settings["api_timeout"] = self.api_timeout_sb.value()
        self.settings["auto_fetch_weather"] = self.auto_fetch_cb.isChecked()
        self.settings["default_lat"] = self.def_lat_sb.value()
        self.settings["default_lon"] = self.def_lon_sb.value()
        self.settings["default_drag"] = self.def_drag_cb.currentText()
        new_dark = self.def_theme_cb.currentText() == "Dark"
        if new_dark != self._dark:
            self._dark = new_dark
            self.settings["dark_theme"] = self._dark
            self.apply_theme()
        SettingsManager.save(self.settings)
        self._set_status_lbl.setText("Settings saved ✓")

    # ── zero angle ─────────────────────────────────────────────────────
    def _zero(self):
        zr = self.zr_sb.value(); sh = self.sh_sb.value()/1000
        bp = Projectile(mass=self.mass_sb.value()/1000,diameter=self.diam_sb.value()/1000,
                        drag_model=self.drag_cb.currentText(),velocity=self.vel_sb.value(),
                        projectile_type=self.type_cb.currentText().lower(),
                        thrust_curve={},burn_time=0,propellant_mass=0)
        be = Environment(altitude=self.alt_sb.value(),temperature=self.temp_sb.value(),
                         pressure=self.pres_sb.value(),humidity=self.hum_sb.value(),
                         wind_speed=0,wind_angle=0,coriolis=False,latitude=self.lat_sb.value())
        def h_at(a):
            try:
                c = TrajectoryCalculator(bp,be,a)
                tr = c.integrate()
                if not tr: return None
                for i in range(1,len(tr)):
                    if tr[i-1][0] <= zr <= tr[i][0]:
                        f = (zr-tr[i-1][0])/(tr[i][0]-tr[i-1][0]) if tr[i][0]!=tr[i-1][0] else 0
                        return tr[i-1][1]+f*(tr[i][1]-tr[i-1][1])-sh
                return tr[-1][1]-sh
            except: return None
        lo, hi = 0.0, 85.0
        fl = h_at(lo); fh = h_at(hi)
        if fl is None or fh is None:
            QMessageBox.warning(self,"Zero Angle","Cannot compute.")
            return
        if fl*fh > 0:
            QMessageBox.warning(self,"Zero",f"No solution 0-85° for range {zr}m.")
            return
        for _ in range(50):
            m = (lo+hi)/2; fm = h_at(m)
            if fm is None: break
            if fl*fm <= 0: hi,fh = m,fm
            else: lo,fl = m,fm
            if hi-lo < 1e-6: break
        a = (lo+hi)/2
        self.ang_sb.setValue(round(a,4))
        QMessageBox.information(self,"Zero Angle",f"Zero angle: {a:.4f}° @ {zr}m")

    # ── file i/o ───────────────────────────────────────────────────────
    def export_csv(self):
        if not self.trajectory: return
        fn,_ = QFileDialog.getSaveFileName(self,"Save CSV","","CSV (*.csv)")
        if not fn: return
        if not fn.endswith('.csv'): fn += '.csv'
        with open(fn,'w',newline='') as f:
            w = csv.writer(f)
            w.writerow(["# Ballistic Calc v5.0",datetime.now().isoformat()])
            w.writerow(["# Type",self.type_cb.currentText()])
            w.writerow(["# Mass(g)",self.mass_sb.value(),"Diam(mm)",self.diam_sb.value()])
            w.writerow(["# Drag Model",self.drag_cb.currentText()])
            w.writerow(["# Vel(m/s)",self.vel_sb.value(),"Angle(deg)",self.ang_sb.value()])
            w.writerow(["# Alt(m)",self.alt_sb.value(),"Temp(C)",self.temp_sb.value(),"Press(hPa)",self.pres_sb.value()])
            w.writerow(["# Wind(m/s)",self.ws_sb.value(),"Wind From(deg)",self.wa_sb.value()])
            w.writerow([])
            w.writerow(["Time(s)","Range(m)","Height(m)","Cross(m)","Vx","Vy","Vz","Speed"])
            for p in self.trajectory:
                w.writerow([f"{v:.3f}" for v in (p[3],p[0],p[1],p[2],p[4],p[5],p[6],p[7])])
        QMessageBox.information(self,"Exported",f"Saved to {fn}")

    def save_scenario(self):
        d = {
            'type':self.type_cb.currentText(),'mass':self.mass_sb.value(),
            'diameter':self.diam_sb.value(),'drag':self.drag_cb.currentText(),
            'velocity':self.vel_sb.value(),'angle':self.ang_sb.value(),
            'altitude':self.alt_sb.value(),'temp':self.temp_sb.value(),
            'pressure':self.pres_sb.value(),'humidity':self.hum_sb.value(),
            'wind_speed':self.ws_sb.value(),'wind_angle':self.wa_sb.value(),
            'coriolis':self.cor_cb.isChecked(),'latitude':self.lat_sb.value(),
            'azimuth':self.az_sb.value(),'icao':self.icao_cb.isChecked(),
            'bc':self.bc_sb.value(),'spin':self.spin_cb.isChecked() if hasattr(self,'spin_cb') else False,
            'twist':self.twist_sb.value() if hasattr(self,'twist_sb') else 10,
            'i_ratio':self.ir_sb.value() if hasattr(self,'ir_sb') else 0.03
        }
        if self.type_cb.currentText().lower() == 'rocket':
            d.update({'burn_time':self.burn_sb.value(),'thrust':self.thrust_sb.value(),
                      'propellant':self.prop_sb.value()})
        SCENARIOS_DIR.mkdir(exist_ok=True)
        fn,_ = QFileDialog.getSaveFileName(self,"Save Scenario",
            str(SCENARIOS_DIR),"JSON (*.json)")
        if not fn: return
        if not fn.endswith('.json'): fn += '.json'
        Path(fn).write_text(json.dumps(d,indent=2))
        QMessageBox.information(self,"Saved",f"Scenario saved.")

    def load_scenario(self):
        fn,_ = QFileDialog.getOpenFileName(self,"Load Scenario",
            str(SCENARIOS_DIR),"JSON (*.json)")
        if not fn: return
        try:
            d = json.loads(Path(fn).read_text())
        except: QMessageBox.critical(self,"Error","Invalid scenario file."); return
        self.type_cb.setCurrentText(d.get('type','Bullet'))
        self.mass_sb.setValue(d.get('mass',10))
        self.diam_sb.setValue(d.get('diameter',7.62))
        self.drag_cb.setCurrentText(d.get('drag','G7'))
        self.vel_sb.setValue(d.get('velocity',800))
        self.ang_sb.setValue(d.get('angle',15))
        self.alt_sb.setValue(d.get('altitude',0))
        self.temp_sb.setValue(d.get('temp',15))
        self.pres_sb.setValue(d.get('pressure',1013.25))
        self.hum_sb.setValue(d.get('humidity',50))
        self.ws_sb.setValue(d.get('wind_speed',0))
        self.wa_sb.setValue(d.get('wind_angle',0))
        self.cor_cb.setChecked(d.get('coriolis',False))
        self.lat_sb.setValue(d.get('latitude',45))
        self.az_sb.setValue(d.get('azimuth',90))
        self.icao_cb.setChecked(d.get('icao',False))
        self.bc_sb.setValue(d.get('bc',0))
        if hasattr(self,'spin_cb'): self.spin_cb.setChecked(d.get('spin',False))
        if hasattr(self,'twist_sb'): self.twist_sb.setValue(d.get('twist',10))
        if hasattr(self,'ir_sb'): self.ir_sb.setValue(d.get('i_ratio',0.03))
        if d.get('type','').lower() == 'rocket':
            self.burn_sb.setValue(d.get('burn_time',1))
            self.thrust_sb.setValue(d.get('thrust',1000))
            self.prop_sb.setValue(d.get('propellant',0))
        QMessageBox.information(self,"Loaded","Scenario loaded.")

    def show_about(self):
        QMessageBox.about(self,"About",
            "Ballistic Calculator v5.0 — 3D Trajectory Engine\n\n"
            "Features:\n"
            "  • 3D state vector [x,y,z] with full 6-DOF physics\n"
            "  • Cubic-spline drag curves (19-point G1/G7)\n"
            "  • Full 3-component Coriolis with azimuth\n"
            "  • 3D gyroscopic spin drift (McCoy model)\n"
            "  • ICAO standard atmosphere toggle\n"
            "  • Ballistic coefficient (BC) input mode\n"
            "  • Crosswind modeling\n"
            "  • 56 built-in presets (external JSON)\n"
            "  • GPS coordinate targeting & auto firing solution\n"
            "  • Zero-angle solver for target range\n"
            "  • Dark/light theme\n"
            "  • Scenario save/load\n"
            "  • CSV export with metadata\n"
            "  • Live input preview\n"
            "License: MIT")

# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    bc = BallisticCalc()
    bc.show()
    sys.exit(app.exec_())

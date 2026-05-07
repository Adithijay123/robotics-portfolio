"""
PID Controller Tuner — Adithi Jayaraman
========================================
Interactive simulator for tuning PID gains on a pendulum or drone system.
Visualises step response, error, and control signal in real time.

Usage:
    python pid_tuner.py

Dependencies:
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# PHYSICS MODELS
# ─────────────────────────────────────────────

def simulate_pendulum(Kp, Ki, Kd, dt=0.01, t_end=5.0, setpoint=0.0, disturbance=True):
    """
    Simulate a damped pendulum with PID control.
    State: [theta (rad), theta_dot (rad/s)]
    
    Nonlinear pendulum: theta'' = -(g/L)*sin(theta) - b*theta' + u
    """
    g = 9.81
    L = 1.0       # pendulum length (m)
    b = 0.3       # damping coefficient
    m = 1.0       # mass (kg)
    
    # Initial condition: 30-degree displacement
    theta0 = np.radians(30)
    
    t = np.arange(0, t_end, dt)
    n = len(t)
    
    theta    = np.zeros(n)
    theta_d  = np.zeros(n)  # theta_dot
    u_hist   = np.zeros(n)
    e_hist   = np.zeros(n)
    
    theta[0] = theta0
    integral  = 0.0
    prev_err  = setpoint - theta0
    
    for i in range(1, n):
        error     = setpoint - theta[i-1]
        integral += error * dt
        # Clamp integral (anti-windup)
        integral  = np.clip(integral, -10, 10)
        derivative = (error - prev_err) / dt
        
        u = Kp * error + Ki * integral + Kd * derivative
        u = np.clip(u, -20, 20)   # actuator saturation
        
        # Add disturbance at t=2.5s
        dist = 0.0
        if disturbance and 2.45 <= t[i] <= 2.55:
            dist = 5.0
        
        # Nonlinear pendulum dynamics (Euler integration)
        alpha = -(g / L) * np.sin(theta[i-1]) - (b / (m * L**2)) * theta_d[i-1] + u / (m * L**2) + dist
        theta_d[i] = theta_d[i-1] + alpha * dt
        theta[i]   = theta[i-1]  + theta_d[i] * dt
        
        e_hist[i]  = error
        u_hist[i]  = u
        prev_err   = error
    
    return t, np.degrees(theta), e_hist, u_hist


def simulate_drone(Kp, Ki, Kd, dt=0.01, t_end=5.0, setpoint=5.0, disturbance=True):
    """
    Simulate a 1-D drone altitude controller.
    State: [altitude (m), velocity (m/s)]
    
    Dynamics: z'' = (T/m) - g + noise
    Setpoint: 5 m hover
    """
    g   = 9.81
    m   = 0.5     # drone mass (kg)
    
    t = np.arange(0, t_end, dt)
    n = len(t)
    
    z      = np.zeros(n)   # altitude
    v      = np.zeros(n)   # velocity
    u_hist = np.zeros(n)
    e_hist = np.zeros(n)
    
    integral  = 0.0
    prev_err  = setpoint - 0.0
    
    for i in range(1, n):
        error     = setpoint - z[i-1]
        integral += error * dt
        integral  = np.clip(integral, -20, 20)
        derivative = (error - prev_err) / dt
        
        u = Kp * error + Ki * integral + Kd * derivative
        u = np.clip(u, 0, 30)  # thrust can't be negative
        
        # Wind gust disturbance at t=3s
        dist = 0.0
        if disturbance and 2.95 <= t[i] <= 3.05:
            dist = -3.0
        
        accel     = (u / m) - g + dist
        v[i]      = v[i-1] + accel * dt
        z[i]      = z[i-1] + v[i]  * dt
        z[i]      = max(0.0, z[i])  # ground constraint
        
        e_hist[i] = error
        u_hist[i] = u
        prev_err  = error
    
    return t, z, e_hist, u_hist


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(t, y, setpoint, dt):
    """Compute rise time, settling time, overshoot, steady-state error."""
    steady_band = 0.02 * abs(setpoint) if setpoint != 0 else 0.05
    sp = setpoint
    y0 = y[0]

    # Rise time: 10% → 90% of setpoint
    if sp != y0:
        ten_pct  = y0 + 0.10 * (sp - y0)
        ninety   = y0 + 0.90 * (sp - y0)
        t10 = next((t[i] for i in range(len(y)) if (sp > y0 and y[i] >= ten_pct) or (sp < y0 and y[i] <= ten_pct)), None)
        t90 = next((t[i] for i in range(len(y)) if (sp > y0 and y[i] >= ninety)  or (sp < y0 and y[i] <= ninety)),  None)
        rise_time = (t90 - t10) if (t10 is not None and t90 is not None) else float('nan')
    else:
        rise_time = 0.0

    # Settling time: last time outside ±2% band
    settled = [i for i in range(len(y)) if abs(y[i] - sp) > steady_band]
    settle_time = t[settled[-1]] if settled else 0.0

    # Overshoot
    if sp > y0:
        peak = np.max(y)
        overshoot = max(0, (peak - sp) / abs(sp - y0) * 100) if sp != y0 else 0
    else:
        peak = np.min(y)
        overshoot = max(0, (sp - peak) / abs(sp - y0) * 100) if sp != y0 else 0

    # Steady-state error (mean of last 10%)
    tail = y[int(0.9 * len(y)):]
    ss_error = abs(sp - np.mean(tail))

    return rise_time, settle_time, overshoot, ss_error


# ─────────────────────────────────────────────
# DEFAULT GAINS  (good starting points)
# ─────────────────────────────────────────────
DEFAULTS = {
    "Pendulum": {"Kp": 25.0, "Ki": 2.0,  "Kd": 8.0,  "setpoint": 0.0,  "unit": "°",   "label": "Angle (°)"},
    "Drone":    {"Kp": 4.0,  "Ki": 0.8,  "Kd": 2.5,  "setpoint": 5.0,  "unit": "m",   "label": "Altitude (m)"},
}

current_system = "Pendulum"


# ─────────────────────────────────────────────
# FIGURE LAYOUT
# ─────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0d0f14",
    "axes.facecolor":   "#13161e",
    "axes.edgecolor":   "#2a2d3a",
    "axes.labelcolor":  "#c8ccd8",
    "text.color":       "#c8ccd8",
    "xtick.color":      "#6b7080",
    "ytick.color":      "#6b7080",
    "grid.color":       "#1e2230",
    "grid.linewidth":   0.8,
    "font.family":      "monospace",
    "lines.linewidth":  2.0,
})

fig = plt.figure(figsize=(16, 9))
fig.suptitle("PID CONTROLLER TUNER", fontsize=14, fontweight="bold",
             color="#00e5ff", y=0.98)

gs = gridspec.GridSpec(
    3, 2,
    left=0.07, right=0.62,
    top=0.90, bottom=0.12,
    hspace=0.45, wspace=0.35
)

ax_resp  = fig.add_subplot(gs[0, :])   # full-width: response
ax_err   = fig.add_subplot(gs[1, 0])   # error
ax_ctrl  = fig.add_subplot(gs[1, 1])   # control signal
ax_phase = fig.add_subplot(gs[2, :])   # phase portrait / error integral

for ax in (ax_resp, ax_err, ax_ctrl, ax_phase):
    ax.grid(True, alpha=0.5)

# ─── Slider panel ───
slider_color   = "#13161e"
slider_ax_Kp   = fig.add_axes([0.68, 0.78, 0.28, 0.025], facecolor=slider_color)
slider_ax_Ki   = fig.add_axes([0.68, 0.70, 0.28, 0.025], facecolor=slider_color)
slider_ax_Kd   = fig.add_axes([0.68, 0.62, 0.28, 0.025], facecolor=slider_color)

s_Kp = Slider(slider_ax_Kp, "Kp", 0.0, 60.0, valinit=DEFAULTS["Pendulum"]["Kp"],  color="#00e5ff")
s_Ki = Slider(slider_ax_Ki, "Ki", 0.0, 20.0, valinit=DEFAULTS["Pendulum"]["Ki"],  color="#ff6b6b")
s_Kd = Slider(slider_ax_Kd, "Kd", 0.0, 30.0, valinit=DEFAULTS["Pendulum"]["Kd"],  color="#a8ff78")

for s in (s_Kp, s_Ki, s_Kd):
    s.label.set_fontsize(11)
    s.label.set_color("#c8ccd8")
    s.valtext.set_color("#ffffff")

# ─── Radio buttons: system select ───
radio_ax = fig.add_axes([0.68, 0.47, 0.28, 0.12], facecolor="#0d0f14")
radio_ax.set_title("System", color="#c8ccd8", fontsize=9, pad=4)
radio = RadioButtons(radio_ax, ("Pendulum", "Drone"),
                     activecolor="#00e5ff")
for label in radio.labels:
    label.set_color("#c8ccd8")
    label.set_fontsize(10)

# ─── Reset button ───
btn_ax  = fig.add_axes([0.68, 0.38, 0.13, 0.05], facecolor="#1e2230")
btn_reset = Button(btn_ax, "Reset Gains", color="#1e2230", hovercolor="#2a2d3a")
btn_reset.label.set_color("#00e5ff")
btn_reset.label.set_fontsize(9)

# ─── Disturbance toggle ───
dist_ax   = fig.add_axes([0.83, 0.38, 0.13, 0.05], facecolor="#1e2230")
btn_dist  = Button(dist_ax, "Disturbance: ON", color="#1e2230", hovercolor="#2a2d3a")
btn_dist.label.set_color("#ff9f43")
btn_dist.label.set_fontsize(9)
disturbance_on = [True]

# ─── Metrics text box ───
metrics_ax = fig.add_axes([0.67, 0.10, 0.30, 0.25], facecolor="#0d1117")
metrics_ax.axis("off")
metrics_text = metrics_ax.text(
    0.05, 0.95, "", transform=metrics_ax.transAxes,
    verticalalignment="top", fontsize=9,
    color="#c8ccd8", fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#13161e", edgecolor="#2a2d3a")
)

# ─── Decorative label ───
fig.text(0.675, 0.98, "Adithi Jayaraman · Portfolio Project",
         fontsize=8, color="#3a4055", ha="left")


# ─────────────────────────────────────────────
# PLOT UPDATE
# ─────────────────────────────────────────────

def update(_=None):
    Kp = s_Kp.val
    Ki = s_Ki.val
    Kd = s_Kd.val
    sys = current_system
    cfg = DEFAULTS[sys]

    if sys == "Pendulum":
        t, y, e, u = simulate_pendulum(Kp, Ki, Kd,
                                        setpoint=cfg["setpoint"],
                                        disturbance=disturbance_on[0])
    else:
        t, y, e, u = simulate_drone(Kp, Ki, Kd,
                                     setpoint=cfg["setpoint"],
                                     disturbance=disturbance_on[0])

    sp     = cfg["setpoint"]
    unit   = cfg["unit"]
    ylabel = cfg["label"]

    # ── Response plot ──
    ax_resp.cla()
    ax_resp.plot(t, y,  color="#00e5ff", lw=2,   label=f"Output ({unit})")
    ax_resp.axhline(sp, color="#ff6b6b", lw=1.2, ls="--", label=f"Setpoint ({sp}{unit})")
    ax_resp.fill_between(t, sp * 0.98, sp * 1.02,
                          alpha=0.08, color="#ff6b6b", label="±2% band")
    if disturbance_on[0]:
        d_t = 2.5 if sys == "Pendulum" else 3.0
        ax_resp.axvline(d_t, color="#ff9f43", lw=1, ls=":", alpha=0.8)
        ax_resp.text(d_t + 0.05, ax_resp.get_ylim()[0] if ax_resp.get_ylim()[0] != 0 else -1,
                     "disturbance", color="#ff9f43", fontsize=7, alpha=0.8)
    ax_resp.set_ylabel(ylabel)
    ax_resp.set_title(f"Step Response  —  {sys}  |  Kp={Kp:.1f}  Ki={Ki:.2f}  Kd={Kd:.2f}",
                      color="#c8ccd8", fontsize=10)
    ax_resp.legend(fontsize=8, loc="upper right",
                   facecolor="#0d0f14", edgecolor="#2a2d3a", labelcolor="#c8ccd8")
    ax_resp.grid(True, alpha=0.4)

    # ── Error plot ──
    ax_err.cla()
    ax_err.plot(t, e, color="#ff6b6b", lw=1.5)
    ax_err.axhline(0, color="#2a2d3a", lw=1)
    ax_err.fill_between(t, e, alpha=0.15, color="#ff6b6b")
    ax_err.set_ylabel(f"Error ({unit})")
    ax_err.set_xlabel("Time (s)")
    ax_err.set_title("Error", color="#c8ccd8", fontsize=9)
    ax_err.grid(True, alpha=0.4)

    # ── Control signal ──
    ax_ctrl.cla()
    ax_ctrl.plot(t, u, color="#a8ff78", lw=1.5)
    ax_ctrl.axhline(0, color="#2a2d3a", lw=1)
    ax_ctrl.fill_between(t, u, alpha=0.12, color="#a8ff78")
    ax_ctrl.set_ylabel("Control Output (u)")
    ax_ctrl.set_xlabel("Time (s)")
    ax_ctrl.set_title("Control Signal", color="#c8ccd8", fontsize=9)
    ax_ctrl.grid(True, alpha=0.4)

    # ── Phase portrait (error vs de/dt) ──
    ax_phase.cla()
    de = np.gradient(e, t)
    # Fading trail: plot in segments with increasing alpha
    n_segs = 30
    seg_len = max(1, len(t) // n_segs)
    for s_i in range(n_segs):
        start = s_i * seg_len
        end   = min(start + seg_len + 1, len(t))
        alpha = 0.15 + 0.85 * (s_i / n_segs)
        ax_phase.plot(e[start:end], de[start:end], color="#c77dff", lw=1.2, alpha=alpha)
    ax_phase.scatter(e[0],  de[0],  color="#00e5ff", s=50, zorder=5, label="Start")
    ax_phase.scatter(e[-1], de[-1], color="#ff6b6b", s=50, zorder=5, label="End")
    ax_phase.axhline(0, color="#2a2d3a", lw=0.8)
    ax_phase.axvline(0, color="#2a2d3a", lw=0.8)
    # Show goal origin star when system is settled
    if abs(e[-1]) < 1.0 and abs(de[-1]) < 2.0:
        ax_phase.scatter(0, 0, color="#a8ff78", s=100, marker="*", zorder=6, label="Goal (origin)")
    ax_phase.set_xlabel(f"Error ({unit})")
    ax_phase.set_ylabel("d(Error)/dt")
    ax_phase.set_title("Phase Portrait  —  well-tuned = spiral into origin",
                        color="#c8ccd8", fontsize=9)
    ax_phase.legend(fontsize=8, facecolor="#0d0f14",
                    edgecolor="#2a2d3a", labelcolor="#c8ccd8")
    ax_phase.grid(True, alpha=0.4)

    # ── Metrics ──
    rt, st, os_, sse = compute_metrics(t, y, sp, t[1] - t[0])
    tail_std  = np.std(y[-200:])
    tail_err  = abs(np.mean(y[-200:]) - sp)
    threshold = max(0.5, 0.05 * abs(sp - y[0])) if sp != y[0] else 0.5
    stability = "STABLE ✓" if tail_err < threshold and tail_std < threshold else "UNSTABLE ✗"
    stab_color = "#a8ff78" if "STABLE" in stability else "#ff6b6b"

    metrics_str = (
        f"  ┌─ PERFORMANCE METRICS ─────────────┐\n"
        f"  │  Rise Time    : {rt:.3f} s\n"
        f"  │  Settling Time: {st:.3f} s\n"
        f"  │  Overshoot    : {os_:.1f} %\n"
        f"  │  SS Error     : {sse:.4f} {unit}\n"
        f"  │  Status       : {stability}\n"
        f"  └───────────────────────────────────┘"
    )
    metrics_text.set_text(metrics_str)
    metrics_text.set_color(stab_color if "UNSTABLE" in stability else "#c8ccd8")

    fig.canvas.draw_idle()


def on_system_change(label):
    global current_system
    current_system = label
    cfg = DEFAULTS[label]
    s_Kp.set_val(cfg["Kp"])
    s_Ki.set_val(cfg["Ki"])
    s_Kd.set_val(cfg["Kd"])
    update()


def on_reset(_):
    cfg = DEFAULTS[current_system]
    s_Kp.set_val(cfg["Kp"])
    s_Ki.set_val(cfg["Ki"])
    s_Kd.set_val(cfg["Kd"])
    update()


def on_dist_toggle(_):
    disturbance_on[0] = not disturbance_on[0]
    state = "ON" if disturbance_on[0] else "OFF"
    btn_dist.label.set_text(f"Disturbance: {state}")
    update()


# ─── Wire up callbacks ───
s_Kp.on_changed(update)
s_Ki.on_changed(update)
s_Kd.on_changed(update)
radio.on_clicked(on_system_change)
btn_reset.on_clicked(on_reset)
btn_dist.on_clicked(on_dist_toggle)

# ─── Initial draw ───
update()

plt.show()
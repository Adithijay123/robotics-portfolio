# PID Controller Tuner

An interactive PID gain tuning simulator built in Python. Tune proportional, integral, and derivative gains in real time against two physical systems, a nonlinear pendulum and a drone altitude controller, and observe the effect on step response, error, control signal, and phase portrait simultaneously.

Built as part of a robotics engineering portfolio to demonstrate practical understanding of closed-loop control, system dynamics, and stability analysis.

---

## Pendulum — Stable Tuning (Kp=25, Ki=2, Kd=8)

![Pendulum demo](demo_pendulum.gif)

## Drone — Gain Exploration

![Drone demo](demo_drone.gif)

---

## Overview

PID control is the dominant control strategy across robotics, aerospace, and industrial automation. Every robotic joint, drone flight controller, CNC axis, and thermal regulator relies on some form of PID. This tool makes the theory tangible: adjust a gain and immediately see how rise time, overshoot, settling time, and steady-state error respond — including the system's reaction to a mid-run disturbance.

The project demonstrates:

- Physical modelling of nonlinear dynamical systems from equations of motion
- Closed-loop simulation with Euler integration, actuator saturation, and integral anti-windup
- Real-time visualisation of four diagnostic views simultaneously
- Quantitative performance metrics calculated from the simulated response
- Phase portrait analysis — a well-tuned controller produces a spiral into the origin; an unstable one produces a limit cycle or divergence

---

## Physics Models

### Pendulum

Nonlinear damped pendulum starting at 30 degrees, controlled back to vertical (0 degrees).

```
theta'' = -(g/L) * sin(theta) - (b / mL^2) * theta' + u / mL^2
```

Parameters: g = 9.81 m/s², L = 1.0 m, b = 0.3 (damping), m = 1.0 kg. Impulse disturbance at t = 2.5 s. Actuator output clamped to [-20, 20].

### Drone (1D Altitude)

Single-axis quadrotor altitude controller. Hover setpoint at 5 m.

```
z'' = T/m - g
```

Parameters: m = 0.5 kg, g = 9.81 m/s². Thrust constrained to [0, 30] — physically cannot push downward. Wind gust disturbance at t = 3.0 s.

Both models use Euler integration at dt = 0.01 s with integral anti-windup clamped to prevent windup during saturation.

---

## Interface

| Control | Description |
|---|---|
| Kp slider | Proportional gain — 0 to 60 |
| Ki slider | Integral gain — 0 to 20 |
| Kd slider | Derivative gain — 0 to 30 |
| System selector | Switch between Pendulum and Drone |
| Reset Gains | Restore well-tuned defaults for the active system |
| Disturbance toggle | Inject a mid-run impulse to test rejection |

### Default gains

| System | Kp | Ki | Kd |
|---|---|---|---|
| Pendulum | 25.0 | 2.0 | 8.0 |
| Drone | 4.0 | 0.8 | 2.5 |

---

## What Each Gain Does

**Kp (Proportional)** — drives output proportional to current error. Higher values speed up response but increase overshoot and can cause oscillation.

**Ki (Integral)** — eliminates steady-state error by accumulating error over time. Too high causes integral windup and sustained oscillation. Anti-windup is implemented here.

**Kd (Derivative)** — damps the response by reacting to rate of change of error. Reduces overshoot and improves settling. Too high amplifies noise and causes instability.

### Suggested experiments

- Set Ki = 12, Kd = 0 — observe sustained oscillation from integral windup with no damping
- Set Kp = 3 — sluggish response, long settling time, poor disturbance rejection
- Set Kd = 25 — aggressive derivative action, strong overshoot suppression
- Toggle disturbance on and observe how quickly different gain sets recover

---

## Performance Metrics

Calculated automatically from the simulated response:

- **Rise time** — time for output to go from 10% to 90% of the setpoint
- **Settling time** — last time the output exceeds a 2% band around setpoint
- **Overshoot** — peak exceedance beyond setpoint as a percentage of total displacement
- **Steady-state error** — mean absolute error over the final 10% of the simulation
- **Status** — STABLE or UNSTABLE, determined by tail variance and mean error relative to a 5% threshold

---

## Installation

```bash
git clone https://github.com/adithijayaraman/pid-controller-tuner
cd pid-controller-tuner
pip install matplotlib numpy
python pid_tuner.py
```

Python 3.8 or higher. No dependencies beyond the standard scientific stack.

---

## Project Structure

```
pid-controller-tuner/
├── pid_tuner.py
├── README.md
├── demo_pendulum.gif
└── demo_drone.gif
```

---

MIT Licence

# Adithi Jayaraman — Portfolio

First Class Honours graduate in Robotics (Falmouth University, July 2026), building end-to-end robotics systems, from mechanical design through control software to statistically validated, hardware-grounded results — at the intersection of perception, sensor fusion, reinforcement learning, and autonomous systems.

[LinkedIn](https://www.linkedin.com/in/adithi-jayaraman-a21583153/)

---

## Projects

### Dissertation — PPO vs. Q-Learning on a 6-DOF Industrial Arm
A controlled, statistically powered, hardware-referenced benchmark of PPO against Q-Learning for autonomous robotic arm navigation across constrained maze geometries. Implemented from scratch in MuJoCo and grounded against a physical UR7e arm running a deterministic hardcoded controller, instrumented with a custom 3D-printed end effector and piezoelectric collision sensing.

**Stack:** MuJoCo, Python, Reinforcement Learning (PPO, Q-Learning), UR7e (physical hardware)
**Focus:** Sim-to-real grounding, statistical validation, RL vs. classical/human control benchmarking

[View project](./Dissertation/)

---

### Warthog-Inspired Mobile Robot Platform
A ruggedised, animatronic field robot built in collaboration with a PhD researcher from the University of Exeter studying warthog–mongoose interaction dynamics, intended for field deployment in Uganda. Drives locomotion via high-torque ODrive actuators with Raspberry Pi control and radio remote teleoperation, engineered for stability under real, unpredictable field conditions. The project is set to be published, with the author and two Falmouth collaborators as co-authors.

**Stack:** Raspberry Pi, ODrive, ROS2, Teleoperation, Data collection
**Includes:** Powered scissor-lift for payload transport, modular architecture for future semi-autonomous upgrades

[View project](./Warthog/)

---

### LLM Arm Planner
Natural language to robotic arm task planner: converts plain-English instructions into validated pick-and-place action sequences via the Gemini API, executed through a real ROS2 action server simulation.

**Stack:** Python, ROS2, Gemini API
**Includes:** Mock in-process and full ROS2 action-based execution modes

[View project](./llm_arm_planner/)

---

### 2D Occupancy Grid SLAM Simulation
Simulated robot and lidar in a 2D environment, illustrating the core SLAM problem via side-by-side occupancy grid maps built from ground-truth pose vs. drifted odometry.

**Stack:** Python, NumPy, Matplotlib
**Demonstrates:** Log-odds occupancy grid mapping, the localisation–mapping problem

[View project](./slam_2d_sim/)

---

### HRI Companion Robot (Smushie)
A soft, personalised companion robot for elderly and assistive care, adapting movement, sound, and haptic responses over time. Uses a Hebbian reinforcement learning model to learn desired responses from a user's voice-perceived emotion.

**Stack:** Python, Reinforcement Learning (Hebbian), Embedded Sensing
**Focus:** Human-robot interaction, behaviour policy design, touch/proximity sensing

[View project](./HRI_smushie/)

---

### IMU Activity Classifier
Classifies human activities from raw accelerometer and gyroscope data using hand-crafted time-domain and frequency-domain features. Covers the full pipeline from raw sensor windows to trained model inference.

**Stack:** Python, scikit-learn, pandas, matplotlib
**Models:** Random Forest, SVM (RBF kernel)
**Dataset:** UCI HAR — 7,352 labelled sensor windows at 50 Hz
**Result:** ~93–95% test accuracy across 6 activity classes

[View project](./imu-activity-classifier/)

---

## About

I am a First Class Honours graduate in Robotics from Falmouth University (July 2026), with hands-on experience across the full robotics stack, from autonomous systems and reinforcement learning to hardware integration and sensor fusion.

My final-year dissertation provides a controlled, statistically powered, hardware-referenced benchmark of PPO against Q-Learning on a six-DOF industrial arm across constrained maze geometries, implemented from scratch in MuJoCo and grounded against a physical UR7e arm running a deterministic hardcoded controller I instrumented myself with a custom 3D-printed end effector and piezoelectric collision sensing.

My most significant project to date is a ruggedised, animatronic robot warthog, built in collaboration with a PhD researcher from the University of Exeter studying warthog–mongoose interaction dynamics, intended for field deployment in Uganda. I integrated high-torque ODrive actuators with Raspberry Pi control to keep the platform stable and controllable under real, unpredictable field conditions. The project is set to be published, with myself and two Falmouth collaborators as co-authors.

Beyond these two projects, I have also built SOLARIS, a 3-DOF robotic arm controlled through real-time hand-gesture tracking, converting live MediaPipe and OpenCV pose data into low-latency, physically reliable motor commands, with the full mechanical structure engineered in Fusion 360; a companion robot for elderly care that uses a Hebbian reinforcement learning model to adapt its behaviour to individual users over time; a natural-language robotic arm task planner that converts plain-English instructions into validated action sequences via an LLM, executed through a real ROS2 action server; a 2D occupancy-grid SLAM simulation illustrating how odometry drift corrupts mapping; and a PID controller tuning tool for closed-loop system control.

Across all of these systems, my instinct has been the same, do not trust simulation or theory alone, ground every claim in real, physically validated hardware behaviour. I have taken full ownership from mechanical design through to control software and statistical validation.

Professionally, I have worked as a development and prototyping intern at Ungraded, contributing to a confidential imaging and automation system, and continuing part-time alongside my studies. I am comfortable working across the hardware-software boundary, maintaining technical documentation, and iterating quickly in R&D environments.

**Core skills:** Python · C++ · ROS2 · MuJoCo · Gazebo · OpenCV · MediaPipe · Reinforcement Learning · Raspberry Pi · Fusion 360 · Git . Scrum . CAD . PCB/Electronic Development 

Open to full-time roles and internships in robotics, autonomy, and systems engineering  and other research opportunities.

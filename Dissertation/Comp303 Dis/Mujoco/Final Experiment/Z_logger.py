"""
Z-Maze (Lego staircase)
Hardware:     UR7e arm + Arduino Nano 

HOW TO USE:
  - Run this script once per trial
  - Auto-increments: trial_1.csv, trial_2.csv, trial_3.csv 
  - Each run traverses all 7 waypoints then returns to HOME

Path:
  WP1 -> WP2 -> WP3 -> WP4 -> WP5 -> WP6 -> WP7 -> HOME

Waypoints manually calibrated from teach pendant, Z staircase maze
  WP1  base=-92.26  shoulder=-72.33  elbow=-142.38  w1=-55.29  w2=89.94  w3=-2.25
  WP2  base=-73.45  shoulder=-66.09  elbow=-145.74  w1=-58.19  w2=89.94  w3=16.56
  WP3  base=-75.02  shoulder=-71.99  elbow=-142.58  w1=-55.46  w2=89.92  w3=14.98
  WP4  base=-78.56  shoulder=-78.56  elbow=-138.23  w1=-53.22  w2=89.92  w3=11.44
  WP5  base=-80.88  shoulder=-83.03  elbow=-134.73  w1=-52.25  w2=89.91  w3=9.11
  WP6  base=-83.37  shoulder=-91.31  elbow=-127.02  w1=-51.69  w2=89.91  w3=6.60
  WP7  base=-75.38  shoulder=-90.28  elbow=-128.05  w1=-51.68  w2=89.90  w3=14.59
  HOME base=-90.00  shoulder=-50.00  elbow=-140.00  w1=-80.00  w2=90.00  w3=0.00

Dependencies:
    pip install ur-rtde pyserial

Safety:
    - JOINT_SPEED = 0.15 rad/s 
    - e-stop within reach at all times
    - Script pauses before starting to make sure the workspace is clear
"""

import rtde_control
import rtde_receive
import serial
import math
import time
import csv
import os
import threading
from datetime import datetime

#   Configuration 
ROBOT_IP = "192.168.0.100"
ARDUINO_PORT  = "COM6"
BAUD_RATE     = 9600

JOINT_SPEED   = 0.15              # rad/s
JOINT_ACCEL   = 0.15              # rad/s square
DWELL_TIME    = 0.4               # seconds pause at each waypoint
TRIAL_TIMEOUT = 120.0             # seconds max — longer for 7-point path

OUTPUT_DIR    = "."
ALGORITHM     = "hardcoded"


#   Waypoints (degrees)  
# [base, shoulder, elbow, wrist1, wrist2, wrist3]

HOME_DEG = [-90.00, -50.00, -140.00, -80.00,  90.00,   0.00]

WAYPOINT_DEGS = [
    [-92.26, -72.33, -142.38, -55.29,  89.94,  -2.25],  # WP1 — start
    [-73.45, -66.09, -145.74, -58.19,  89.94,  16.56],  # WP2
    [-75.02, -71.99, -142.58, -55.46,  89.92,  14.98],  # WP3
    [-78.56, -78.56, -138.23, -53.22,  89.92,  11.44],  # WP4
    [-80.88, -83.03, -134.73, -52.25,  89.91,   9.11],  # WP5
    [-83.37, -91.31, -127.02, -51.69,  89.91,   6.60],  # WP6
    [-75.38, -90.28, -128.05, -51.68,  89.90,  14.59],  # WP7 — goal
]

WAYPOINT_NAMES = ["WP1 start", "WP2", "WP3", "WP4", "WP5", "WP6", "WP7 goal"]

def deg_to_rad(deg):
    return [math.radians(d) for d in deg]

HOME      = deg_to_rad(HOME_DEG)
WAYPOINTS = [deg_to_rad(wp) for wp in WAYPOINT_DEGS]


#   Auto trial number   
def get_next_trial_number(directory="."):
    i = 1
    while os.path.exists(os.path.join(directory, f"trial_{i}.csv")):
        i += 1
    return i


#   CSV 
HEADERS = [
    "trial",
    "timestamp",
    "algorithm",
    "maze",
    "collisions",
    "completion_time_s",
    "goal_reached",
    "timeout",
    "notes",
]

def write_csv(filepath, row):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerow(row)
    print(f"\n  Saved: {filepath}")


#   Arduino Piezo Reader        
class PiezoCollisionReader:
    def __init__(self, port, baud):
        self.port       = port
        self.baud       = baud
        self.collisions = []
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._ser       = None
        self._thread    = None
        self._ready     = threading.Event()

    def start(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"  Serial open on {self.port} at {self.baud} baud")
        except serial.SerialException as e:
            print(f"  [WARNING] Could not open Arduino: {e}")
            print("  Collision count will be 0.")
            return False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        if self._ready.wait(timeout=5):
            print("  Arduino READY.")
        else:
            print("  [WARNING] No READY from Arduino within 5s.")
        return True

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                if self._ser and self._ser.in_waiting:
                    line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                    if line == "READY":
                        self._ready.set()
                    elif line.startswith("HIT,"):
                        parts = line.split(",")
                        if len(parts) == 4:
                            with self._lock:
                                self.collisions.append({
                                    "index":     int(parts[1]),
                                    "elapsed_s": float(parts[2]),
                                    "raw_val":   int(parts[3]),
                                })
                            print(f"    [COLLISION] #{parts[1]} "
                                  f"at {parts[2]}s  raw={parts[3]}")
            except Exception:
                pass
            time.sleep(0.005)

    def get_count(self):
        with self._lock:
            return len(self.collisions)

    def stop(self):
        self._stop.set()
        if self._ser:
            self._ser.close()


#   Main    
def main():
    trial_num = get_next_trial_number(OUTPUT_DIR)
    csv_path  = os.path.join(OUTPUT_DIR, f"trial_{trial_num}.csv")

    print("=" * 55)
    print(f"  UR7e Z-Maze — Trial {trial_num}")
    print(f"  Waypoints: {len(WAYPOINTS)}  |  Speed: {JOINT_SPEED} rad/s")
    print(f"  Saving to: trial_{trial_num}.csv")
    print("=" * 55)

    #   Arduino 
    piezo      = PiezoCollisionReader(ARDUINO_PORT, BAUD_RATE)
    arduino_ok = piezo.start()
    if not arduino_ok:
        ans = input("\n  No Arduino. Continue without collision sensing? [y/N]: ")
        if ans.strip().lower() != "y":
            print("  Aborted.")
            return

    #   Robot 
    print(f"\nConnecting to UR7e at {ROBOT_IP}...")
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    print("Connected.")

    input("\n  Workspace clear? E-stop in hand? Press Enter to start\n")

    goal_reached = False
    timed_out    = False

    #   Move to WP1 first, then start clock          
    print(f"  [1/{len(WAYPOINTS)}] Moving to {WAYPOINT_NAMES[0]}...")
    rtde_c.moveJ(WAYPOINTS[0], JOINT_SPEED, JOINT_ACCEL)
    time.sleep(DWELL_TIME)
    print("  At WP1. Starting trial clock.")
    trial_start = time.time()

    try:
        #   WP2 through WP7                 
        for i in range(1, len(WAYPOINTS)):
            name = WAYPOINT_NAMES[i]
            print(f"  [{i+1}/{len(WAYPOINTS)}] Moving to {name}...")
            rtde_c.moveJ(WAYPOINTS[i], JOINT_SPEED, JOINT_ACCEL)

            elapsed = round(time.time() - trial_start, 3)
            print(f"        Reached {name} at {elapsed}s")
            time.sleep(DWELL_TIME)

            if time.time() - trial_start > TRIAL_TIMEOUT:
                timed_out = True
                raise TimeoutError(f"Timeout at {name}")

        goal_reached = True

    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Stopping robot.")
        rtde_c.stopJ(2.0)

    except TimeoutError as e:
        print(f"  [TIMEOUT] {e}")

    completion_time = round(time.time() - trial_start, 3)
    collisions      = piezo.get_count()

    #   Print result                   ─
    tcp = rtde_r.getActualTCPPose()
    print(f"\n  TCP: x={tcp[0]*1000:.1f}mm  y={tcp[1]*1000:.1f}mm  "
          f"z={tcp[2]*1000:.1f}mm")
    print(f"\n    Trial {trial_num} result          ")
    print(f"  Collisions      : {collisions}")
    print(f"  Completion time : {completion_time}s")
    print(f"  Goal reached    : {'YES' if goal_reached else 'NO'}")
    print(f"  Timeout         : {'YES' if timed_out else 'NO'}")

    #   Save CSV   
    write_csv(csv_path, {
        "trial":             trial_num,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm":         ALGORITHM,
        "maze":              "Z",
        "collisions":        collisions,
        "completion_time_s": completion_time,
        "goal_reached":      int(goal_reached),
        "timeout":           int(timed_out),
        "notes":             "",
    })

    #   Return home directly from WP7       
    print("\n  Returning to HOME from WP7...")
    rtde_c.moveJ(HOME, JOINT_SPEED, JOINT_ACCEL)
    print("  Arm is home. Safe to leave.")

    rtde_c.stopScript()
    piezo.stop()
    print(f"\n  Done. Run again for trial {trial_num + 1}.")
    print("=" * 55)


if __name__ == "__main__":
    main()
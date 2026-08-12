"""
Square U-Maze
Hardware:     UR7e arm + Arduino Nano 

HOW TO USE:
  - Each run does ONE traversal: WP1 -> WP2 -> WP3 -> WP4 -> HOME
  - After reaching WP4 (goal) the arm returns to WP1 (home)

Waypoints manually calibrated from from teach pendant
  WP1 HOME/START  base=-93.84  shoulder=-73.27  elbow=-141.51  w1=-55.21  w2=89.94  w3=-3.84
  WP2 corner-L    base=-92.91  shoulder=-87.44  elbow=-130.52  w1=-52.04  w2=89.92  w3=-2.93
  WP3 corner-R    base=-73.35  shoulder=-83.07  elbow=-134.39  w1=-52.57  w2=89.91  w3=16.63
  WP4 GOAL        base=-67.37  shoulder=-64.78  elbow=-146.08  w1=-59.18  w2=89.94  w3=22.64

Dependencies:
    pip install ur-rtde pyserial

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
DWELL_TIME    = 0.5               # seconds pause at each waypoint
TRIAL_TIMEOUT = 90.0              # seconds max before timeout flag

OUTPUT_DIR    = "."
ALGORITHM     = "hardcoded"


#   Waypoints (degrees -> radians)                       
HOME_DEG = [-90.00, -50.00, -140.00, -80.00,  90.00,   0.00]   # TRUE HOME (from pendant)
WP1_DEG = [-93.84, -73.27, -141.51, -55.21,  89.94,  -3.84]   # maze entry (start)
WP2_DEG = [-92.91, -87.44, -130.52, -52.04,  89.92,  -2.93]   # corner bottom-left
WP3_DEG = [-73.35, -83.07, -134.39, -52.57,  89.91,  16.63]   # corner bottom-right
WP4_DEG = [-67.37, -64.78, -146.08, -59.18,  89.94,  22.64]   # GOAL

def deg_to_rad(deg):
    return [math.radians(d) for d in deg]

HOME = deg_to_rad(HOME_DEG)
WP1 = deg_to_rad(WP1_DEG)
WP2 = deg_to_rad(WP2_DEG)
WP3 = deg_to_rad(WP3_DEG)
WP4 = deg_to_rad(WP4_DEG)


#   Auto trial number                              
def get_next_trial_number(directory="."):
    """Scans for existing trial_N.csv files and returns the next number."""
    i = 1
    while os.path.exists(os.path.join(directory, f"trial_{i}.csv")):
        i += 1
    return i


#   CSV Logger                                 ─
HEADERS = [
    "trial",
    "timestamp",
    "algorithm",
    "collisions",
    "completion_time_s",
    "wp2_time_s",
    "wp3_time_s",
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


#   Arduino Piezo Reader                            ─
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
            print(f"  Could not open Arduino: {e}")
            print("  Collision count will be 0.")
            return False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        if self._ready.wait(timeout=5):
            print("  Arduino READY.")
        else:
            print("  No READY from Arduino within 5s.")
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


#   Main                                    ─
def main():
    trial_num = get_next_trial_number(OUTPUT_DIR)
    csv_path  = os.path.join(OUTPUT_DIR, f"trial_{trial_num}.csv")

    print("=" * 55)
    print(f"  UR7e Square U-Maze — Trial {trial_num}")
    print(f"  File will save as: trial_{trial_num}.csv")
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

    input("\n  Workspace clear? E-stop in hand? Press Enter to start trial\n")

    goal_reached = False
    timed_out    = False
    wp2_time     = None
    wp3_time     = None

    #   Move to WP1 (home/start), then start clock            
    print(" Moving to WP1 HOME (start position)...")
    rtde_c.moveJ(WP1, JOINT_SPEED, JOINT_ACCEL)
    time.sleep(DWELL_TIME)
    print("  At WP1. Starting trial clock now.")
    trial_start = time.time()

    try:
        # WP2                                
        rtde_c.moveJ(WP2, JOINT_SPEED, JOINT_ACCEL)
        wp2_time = round(time.time() - trial_start, 3)
        print(f"        Reached WP2 at {wp2_time}s")
        time.sleep(DWELL_TIME)

        if time.time() - trial_start > TRIAL_TIMEOUT:
            timed_out = True
            raise TimeoutError

        # WP3                                
        rtde_c.moveJ(WP3, JOINT_SPEED, JOINT_ACCEL)
        wp3_time = round(time.time() - trial_start, 3)
        print(f"        Reached WP3 at {wp3_time}s")
        time.sleep(DWELL_TIME)

        if time.time() - trial_start > TRIAL_TIMEOUT:
            timed_out = True
            raise TimeoutError

        # WP4 (goal)                            
        rtde_c.moveJ(WP4, JOINT_SPEED, JOINT_ACCEL)
        time.sleep(DWELL_TIME)

        elapsed = time.time() - trial_start
        if elapsed > TRIAL_TIMEOUT:
            timed_out = True
        else:
            goal_reached = True

    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Stopping robot.")
        rtde_c.stopJ(2.0)

    except TimeoutError:
        print(f"  [TIMEOUT] Trial exceeded {TRIAL_TIMEOUT}s")

    completion_time = round(time.time() - trial_start, 3)
    collisions      = piezo.get_count()

    tcp = rtde_r.getActualTCPPose()
    print(f"\n  TCP: x={tcp[0]*1000:.1f}mm  y={tcp[1]*1000:.1f}mm  "
          f"z={tcp[2]*1000:.1f}mm")
    print(f"\n    Trial {trial_num} result              ")
    print(f"  Collisions      : {collisions}")
    print(f"  Completion time : {completion_time}s")
    print(f"  WP2 split time  : {wp2_time}s")
    print(f"  WP3 split time  : {wp3_time}s")
    print(f"  Goal reached    : {'YES' if goal_reached else 'NO'}")
    print(f"  Timeout         : {'YES' if timed_out else 'NO'}")

    #   Save CSV                              ─
    write_csv(csv_path, {
        "trial":             trial_num,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm":         ALGORITHM,
        "collisions":        collisions,
        "completion_time_s": completion_time,
        "wp2_time_s":        wp2_time if wp2_time is not None else "",
        "wp3_time_s":        wp3_time if wp3_time is not None else "",
        "goal_reached":      int(goal_reached),
        "timeout":           int(timed_out),
        "notes":             "",
    })

    #   Return to HOME (direct from WP4, no maze re-entry)         
    print("\n  Returning arm directly to HOME from WP4")
    rtde_c.moveJ(HOME, JOINT_SPEED, JOINT_ACCEL)

    rtde_c.stopScript()
    piezo.stop()
    print(f"  Done. Run the script again for trial {trial_num + 1}.")
    print("=" * 55)


if __name__ == "__main__":
    main()
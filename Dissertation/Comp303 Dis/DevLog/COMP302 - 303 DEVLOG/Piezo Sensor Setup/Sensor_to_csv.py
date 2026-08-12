import serial
import csv
from datetime import datetime
import os

PORT = "COM26"  # change to your port
BAUD = 9600
CSV_PATH = "piezo_log.csv"

def main():
    print("Current working directory:", os.getcwd())
    print("Writing CSV to:", os.path.abspath(CSV_PATH))

    # Use a SHORT timeout so readline() returns regularly and Ctrl+C is processed
    ser = serial.Serial(PORT, BAUD, timeout=0.1)

    try:
        with open(CSV_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)

            if f.tell() == 0:
                writer.writerow(["pc_time_iso", "arduino_value"])

            print("Logging; press Ctrl+C to stop.")
            while True:
                try:
                    line = ser.readline().decode(errors="ignore").strip()
                except KeyboardInterrupt:
                    # In case interrupt happens exactly during readline()
                    break

                if not line:
                    continue

                if not line.isdigit():
                    print("TEXT:", line)
                    continue

                value = int(line)
                pc_time = datetime.now().isoformat(timespec="milliseconds")
                writer.writerow([pc_time, value])
                f.flush()
                print(pc_time, value)

    except KeyboardInterrupt:
        # Ctrl+C while in the main loop
        print("Stopped by user.")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()

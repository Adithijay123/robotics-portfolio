/*
  Piezo collision logger for UR7e maze runs
  Hardware: Arduino Nano, piezo sensor on A0, COM6
  Output:   Serial lines read by companion Python script then converted to CSV
*/

#define PIEZO_PIN     A5
#define THRESHOLD     50     
#define DEBOUNCE_MS   250     

unsigned long lastHit   = 0;
unsigned long startTime = 0;
int           hitCount  = 0;

void setup() {
  Serial.begin(9600);
  startTime = millis();
  Serial.println("READY");
}

void loop() {
  int val = analogRead(PIEZO_PIN);

  if (val > THRESHOLD) {
    unsigned long now = millis();
    if (now - lastHit > DEBOUNCE_MS) {
      lastHit = now;
      hitCount++;
      float elapsed = (now - startTime) / 1000.0;
      //collision_index, elapsed_seconds, raw_sensor_value
      Serial.print("HIT,");
      Serial.print(hitCount);
      Serial.print(",");
      Serial.print(elapsed, 3);
      Serial.print(",");
      Serial.println(val);
    }
  }
}

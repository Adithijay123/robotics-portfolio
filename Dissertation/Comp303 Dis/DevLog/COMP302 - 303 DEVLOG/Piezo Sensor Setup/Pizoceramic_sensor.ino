// Highly sensitive collision detection with analog piezo module on A5

const int PIEZO_PIN   = A5;   // analog input from S pin
const int THRESHOLD   = 40;   // MUCH lower; tune 10–80 range
const int QUIET_BAND  = 5;    // treat values within +/- this as noise around baseline
const int DEBOUNCE_MS = 80;   // longer debounce removes multiple triggers per hit

unsigned long lastTriggerTime = 0;

void setup() {
  Serial.begin(9600);
  pinMode(PIEZO_PIN, INPUT);
}

void loop() {
  int value = analogRead(PIEZO_PIN);
  unsigned long now = millis();

  if (value < QUIET_BAND) {
    value = 0;
  }

  bool collision = false;

  if (value >= THRESHOLD && (now - lastTriggerTime) > DEBOUNCE_MS) {
    collision = true;
    lastTriggerTime = now;
    Serial.println("collision");   // optional text line for human reading
  }

  // ALWAYS print a numeric line for Python to parse
  int plotValue = collision ? 800 : value;
  Serial.println(plotValue);       // Python will read this line
  delay(200);
}
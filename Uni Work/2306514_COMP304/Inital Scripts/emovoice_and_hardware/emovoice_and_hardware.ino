#include <FastLED.h>

#define TOUCH_PIN 2      // TTP223 OUT to D2
#define MOTOR_PIN 3      // S8550 base, via 220Ω, to D3
#define LED_PIN 6        // WS2812 data in to D6
#define NUM_LEDS 8

CRGB leds[NUM_LEDS];
bool ledOn = false;
unsigned long ledTimer = 0; // Track duration

void setup() {
  Serial.begin(9600);
  pinMode(TOUCH_PIN, INPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, HIGH); // Motor OFF initially
  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.clear(true);
}

void loop() {
  // Vibration motor on when sensor activated
  if (digitalRead(TOUCH_PIN) == HIGH) {
    digitalWrite(MOTOR_PIN, LOW);
  } else {
    digitalWrite(MOTOR_PIN, HIGH);
  }

  // Listen for serial commands from python voice emotion script
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "ON") {
      ledOn = true;
      ledTimer = millis(); // start 5s timer
    }
    if (cmd == "OFF") {
      ledOn = false;
      FastLED.clear(true);
    }
  }

  // LED 10 second timer
  if (ledOn) {
    fill_solid(leds, NUM_LEDS, CRGB::Red);
    FastLED.show();
    if (millis() - ledTimer >= 10000) { // 5 seconds passed
      ledOn = false;
      FastLED.clear(true);
    }
  } else {
    FastLED.clear(true);
  }
}

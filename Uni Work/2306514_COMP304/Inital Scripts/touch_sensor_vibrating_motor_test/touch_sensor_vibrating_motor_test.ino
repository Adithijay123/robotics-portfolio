#include <FastLED.h>

#define TOUCH_PIN 2     // TTP223 OUT to D2
#define MOTOR_PIN 3     // S8550 base, via 220Ω, to D3
#define LED_PIN 6       // WS2812 data in to D6
#define NUM_LEDS 8      // Set to the actual number in your strip

CRGB leds[NUM_LEDS];

void setup() {
  pinMode(TOUCH_PIN, INPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, HIGH); // Motor OFF (PNP logic)
  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.clear(true);            // Make sure LEDs are off at start
}

void loop() {
  if (digitalRead(TOUCH_PIN) == HIGH) {
    digitalWrite(MOTOR_PIN, LOW);   // Motor ON
    for (int i = 0; i < NUM_LEDS; i++) {
      leds[i] = CRGB::Red;          // Change to any color you want
    }
    FastLED.show();
  } else {
    digitalWrite(MOTOR_PIN, HIGH);  // Motor OFF
    FastLED.clear(true);            // LEDs OFF
  }
}

#include <FastLED.h>

#define TOUCH_PIN 2      // TTP223 OUT to D2
#define MOTOR_PIN 3      // S8550 base, via 220Ω, to D3
#define LED_PIN   6      
#define NUM_LEDS  8

CRGB leds[NUM_LEDS];

// current state
bool ledOn = false;
unsigned long ledTimer = 0;
const unsigned long LED_DURATION = 10000; // 10 seconds

// which emotion pattern did Python choose
char currentEmotion = 'N';   // N = neutral
int  currentPattern = 0;

void setup() {
  Serial.begin(9600);
  pinMode(TOUCH_PIN, INPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, HIGH); // Motor OFF initially
  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.clear(true);
}



// happy / calm patterns (H, C)
void happyPattern0() {        // warm yellow
  fill_solid(leds, NUM_LEDS, CRGB::Orange);
}

void happyPattern1() {        // rainbow
  static uint8_t hue = 0;
  for (int i = 0; i < NUM_LEDS; i++) {
    leds[i] = CHSV(hue + i * 16, 255, 255);
  }
  hue++;
}

void happyPattern2() {   // pink/blue 
  static uint8_t offset = 0;
  fill_solid(leds, NUM_LEDS, CRGB::Black);
  leds[offset % NUM_LEDS] = CRGB::HotPink;
  leds[(offset + 4) % NUM_LEDS] = CRGB::Aqua;
  offset++;
}

// sad patterns (S)
void sadPattern0() {          // dim blue
  fill_solid(leds, NUM_LEDS, CRGB(0, 0, 40));
}

void sadPattern1() {          // slow blue breathing
  static uint8_t b = 0;
  static int dir = 1;
  b += dir;
  if (b == 0 || b == 140) dir = -dir;
  fill_solid(leds, NUM_LEDS, CRGB(0, 0, b));
}

// angry patterns (A)
void angryPattern0() {        // solid red
  fill_solid(leds, NUM_LEDS, CRGB::Red);
}

void angryPattern1() {        // red strobe
  static bool on = false;
  on = !on;
  if (on) fill_solid(leds, NUM_LEDS, CRGB::Red);
  else    fill_solid(leds, NUM_LEDS, CRGB::Black);
}

// neutral patterns (N)
void neutralPattern0() {      // soft white
  fill_solid(leds, NUM_LEDS, CRGB(60, 60, 60));
}

void neutralPattern1() {      // white chase
  static uint8_t pos = 0;
  fill_solid(leds, NUM_LEDS, CRGB::Black);
  leds[pos % NUM_LEDS] = CRGB::White;
  pos++;
}

// choose pattern based on currentEmotion/currentPattern
void runCurrentPattern() {
  switch (currentEmotion) {

    case 'H': // happy
    case 'C': // calm
      switch (currentPattern) {
        case 0: happyPattern0(); break;
        case 1: happyPattern1(); break;
        case 2: happyPattern2(); break;
        default: happyPattern0(); break;
      }
      break;

    case 'S': // sad
      switch (currentPattern) {
        case 0: sadPattern0(); break;
        case 1: sadPattern1(); break;
        default: sadPattern0(); break;
      }
      break;

    case 'A': // angry
      switch (currentPattern) {
        case 0: angryPattern0(); break;
        case 1: angryPattern1(); break;
        default: angryPattern0(); break;
      }
      break;

    case 'N': // neutral / default
    default:
      switch (currentPattern) {
        case 0: neutralPattern0(); break;
        case 1: neutralPattern1(); break;
        default: neutralPattern0(); break;
      }
      break;
  }
}

void loop() {
  // Vibration motor on when sensor activated
  if (digitalRead(TOUCH_PIN) == HIGH) {
    digitalWrite(MOTOR_PIN, LOW);
  } else {
    digitalWrite(MOTOR_PIN, HIGH);
  }

  // wait for for serial commands from Python
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');  
    cmd.trim();
    if (cmd.length() >= 2) {
      currentEmotion = cmd.charAt(0);          // h,c,s,a,n
      currentPattern = cmd.substring(1).toInt(); // 0,1,2,

      ledOn = true;
      ledTimer = millis();                     // restart 10s timer
    }
  }

  // Run pattern while timer is active
  if (ledOn) {
    runCurrentPattern();
    FastLED.show();

    if (millis() - ledTimer >= LED_DURATION) {
      ledOn = false;
      FastLED.clear(true);
    }
  } else {
    FastLED.clear(true);
  }
}

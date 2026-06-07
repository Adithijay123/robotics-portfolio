#include <FastLED.h>

//LED configuration
#define LED_PIN     15
#define NUM_LEDS    16
#define LED_TYPE    SK6812
#define COLOR_ORDER GRB

CRGB leds[NUM_LEDS];

// Motor configuration
#define MOTOR_PIN   16

//Touch configuration
#define TOUCH1_PIN  2
#define TOUCH2_PIN  4
#define TOUCH3_PIN  5

//Emotion and pattern
char currentEmotion = 'N';    // 'H','C','S','A','N'
int  currentPatternIndex = 0; // 0 or 1 for all emotions, since only 2 patterns
bool emotionActive  = false;

unsigned long emotionStartTime       = 0;
const unsigned long EMOTION_DURATION = 8000;  // 8 s

// motor pattern state
bool motorOn              = false;
unsigned long motorTimer  = 0;
unsigned long motorOnDuration  = 0;
unsigned long motorOffDuration = 0;

// feedback state
bool feedbackSent = false;

void setup() {
  Serial.begin(115200);

  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
  FastLED.clear(true);

  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, HIGH);   // OFF at startup (active-low driver)

  pinMode(TOUCH1_PIN, INPUT);
  pinMode(TOUCH2_PIN, INPUT);
  pinMode(TOUCH3_PIN, INPUT);
}

void fillAll(CRGB c) {
  for (int i = 0; i < NUM_LEDS; i++) {
    leds[i] = c;
  }
}

void showEmotionColor() {
  switch (currentEmotion) {
    case 'H': fillAll(CRGB(0, 0, 255));  break;   // happy: blue
    case 'C': fillAll(CRGB(0, 255, 0));  break;   // calm: green
    case 'S': fillAll(CRGB(255, 128, 0)); break;  // sad: orange
    case 'A': fillAll(CRGB(255, 0, 0));  break;   // angry: red
    case 'N':
    default:  fillAll(CRGB(60, 60, 60)); break;   // neutral: dim white
  }
}

// Motor helpers (active-low)

void setMotorPatternForEmotion() {
  // default: motor fully off
  motorOnDuration  = 0;
  motorOffDuration = 0;
  motorOn = false;
  digitalWrite(MOTOR_PIN, HIGH);   // OFF

  switch (currentEmotion) {
    case 'H': // happy: lively vs softer joy
      if (currentPatternIndex == 0) {
        motorOnDuration  = 150;
        motorOffDuration = 150;
        Serial.println("PATTERN: happy_0");
      } else {
        motorOnDuration  = 200;
        motorOffDuration = 300;
        Serial.println("PATTERN: happy_1");
      }
      break;

    case 'C': // calm
      if (currentPatternIndex == 0) {
        motorOnDuration  = 200;
        motorOffDuration = 800;
        Serial.println("PATTERN: calm_0");
      } else {
        motorOnDuration  = 150;
        motorOffDuration = 600;
        Serial.println("PATTERN: calm_1");
      }
      break;

    case 'S': // sad
      if (currentPatternIndex == 0) {
        motorOnDuration  = 150;
        motorOffDuration = 850;
        Serial.println("PATTERN: sad_0");
      } else {
        motorOnDuration  = 120;
        motorOffDuration = 600;
        Serial.println("PATTERN: sad_1");
      }
      break;

    case 'A': // angry
      if (currentPatternIndex == 0) {
        motorOnDuration  = 80;
        motorOffDuration = 80;
        Serial.println("PATTERN: angry_0");
      } else {
        motorOnDuration  = 200;
        motorOffDuration = 100;
        Serial.println("PATTERN: angry_1");
      }
      break;

    case 'N': // neutral: motor stays OFF
    default:
      Serial.println("PATTERN: neutral_0");
      break;
  }
}

void updateMotorPattern() {
  // If no emotion is active, motor must be OFF
  if (!emotionActive) {
    motorOn = false;
    digitalWrite(MOTOR_PIN, HIGH);   // OFF
    return;
  }

  // If current pattern has zero durations, keep motor OFF
  if (motorOnDuration == 0 && motorOffDuration == 0) {
    motorOn = false;
    digitalWrite(MOTOR_PIN, HIGH);   // OFF
    return;
  }

  unsigned long now = millis();

  if (motorOn) {
    if (now - motorTimer >= motorOnDuration) {
      motorOn = false;
      motorTimer = now;
      digitalWrite(MOTOR_PIN, HIGH);  // OFF
    }
  } else {
    if (now - motorTimer >= motorOffDuration) {
      motorOn = true;
      motorTimer = now;
      digitalWrite(MOTOR_PIN, LOW);   // ON (active-low)
    }
  }
}

//Touch feedback input

void checkTouchFeedback() {
  if (!emotionActive || feedbackSent) return;

  int t1 = digitalRead(TOUCH1_PIN);
  int t2 = digitalRead(TOUCH2_PIN);
  int t3 = digitalRead(TOUCH3_PIN);

  if (t1 == HIGH || t2 == HIGH || t3 == HIGH) {
    Serial.println("FEEDBACK_POS");
    feedbackSent = true;
  }
}

//Serial: emotion + optional pattern index

void checkSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r' || c == ' ') continue; // || means or

    if (c == 'H' || c == 'C' || c == 'S' || c == 'A' || c == 'N') {
      currentEmotion = c;
      currentPatternIndex = 0;

      delayMicroseconds(200);
      if (Serial.available() > 0) {
        char d = Serial.peek();
        if (d >= '0' && d <= '1') {       // only 0 or 1 patterns
          Serial.read();
          currentPatternIndex = d - '0';
        }
      }

      emotionActive    = true;
      emotionStartTime = millis();

      setMotorPatternForEmotion();

      // start pattern from OFF
      motorOn    = false;
      motorTimer = millis();
      feedbackSent = false;
    }
  }
}

void loop() {
  checkSerial();

  unsigned long now = millis();

  // end of emotion window, neutral, motor off
  if (emotionActive && (now - emotionStartTime > EMOTION_DURATION)) {
    emotionActive  = false;
    currentEmotion = 'N';
    motorOn = false;
    digitalWrite(MOTOR_PIN, HIGH);   // OFF
  }

  showEmotionColor();
  FastLED.show();

  updateMotorPattern();
  checkTouchFeedback();

  delay(5);
}

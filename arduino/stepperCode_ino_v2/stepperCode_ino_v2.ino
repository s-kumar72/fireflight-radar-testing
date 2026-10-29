#include <Arduino.h>

#define PIN_PUL 13
// #define RPS 2
#define PUL_PER_REV 3200  // This is a physical configuration on the stepper motor driver

const float RPS = 1.697; // maps to 102 rpm for PanoRadar

const unsigned long PUL_FREQ = RPS * PUL_PER_REV;                  // 1600 Hz
const unsigned long PUL_PERIOD = 1000000UL / PUL_FREQ;             // 625 us
const unsigned long PUL_HIGH_TIME = PUL_PERIOD / 2;                // 312 us
const unsigned long PUL_LOW_TIME = PUL_PERIOD - PUL_HIGH_TIME;     // 313 us

const unsigned long REPORT_PERIOD = 10000; // print step count every 10ms (100 Hz)

unsigned long lastToggle = 0;
unsigned long lastReport = 0;
bool pulState = LOW;
unsigned long ticks = 0; // to sync motor with radar recording

void setup() {
  Serial.begin(115200);
  pinMode(PIN_PUL, OUTPUT);
  while (!Serial);

  Serial.print("Pulse frequency: ");
  Serial.print(PUL_FREQ);
  Serial.println(" Hz");
  Serial.print("Pulse period: ");
  Serial.print(PUL_PERIOD);
  Serial.println(" us");

  lastToggle = micros();
  lastReport = lastToggle;
}

void loop() {
  unsigned long now = micros();
  unsigned long interval = pulState ? PUL_HIGH_TIME : PUL_LOW_TIME;

  if (now - lastToggle >= interval) {
    lastToggle += interval;  // += keeps timing from drifting
    pulState = !pulState;
    digitalWrite(PIN_PUL, pulState);
    if (pulState == HIGH) {
      ticks++; // one step per rising edge
    }
  }

  if (now - lastReport >= REPORT_PERIOD) {
    lastREport += REPORT_PERIOD;
    Serial.println(ticks); // one number per line to be read by motor logger
  }
}
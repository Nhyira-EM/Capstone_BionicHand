#if defined(ARDUINO) && ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

#include "EMGFilters.h"

#define TIMING_DEBUG 1

// Define your 3 sensor input pins
const int SensorInputPins[3] = { 34, 35, 32 };  // adjust as needed

// Create 3 filter objects
EMGFilters myFilters[3];

SAMPLE_FREQUENCY sampleRate = SAMPLE_FREQ_1000HZ;
NOTCH_FREQUENCY humFreq = NOTCH_FREQ_50HZ;

static int Threshold = 600000;
unsigned long timeBudget;
unsigned long timeStamp;

void setup() {
  Serial.begin(115200);

  // Initialize filters for each sensor
  for (int i = 0; i < 3; i++) {
    myFilters[i].init(sampleRate, humFreq, true, true, true);
  }

  timeBudget = 1e6 / 1000;  // for 1000 Hz
}

void loop() {
  timeStamp = micros();
  int envelopes[3];  // Store all envelope values

  for (int i = 0; i < 3; i++) {
    int rawValue = analogRead(SensorInputPins[i]);
    int filteredValue = myFilters[i].update(rawValue);
    int envelope = sq(filteredValue);
    envelopes[i] = (envelope > Threshold) ? envelope : 0;
  }

  if (TIMING_DEBUG) {
    Serial.print("Sensor0:");
    Serial.print(envelopes[0]);
    Serial.print("\tSensor1:");
    Serial.print(envelopes[1]);
    Serial.print("\tSensor2:");
    Serial.println(envelopes[2]);
  }

  timeStamp = micros() - timeStamp;
  delayMicroseconds(500);
}

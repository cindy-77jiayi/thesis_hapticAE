#include <Wire.h>

constexpr uint8_t DRV2605_ADDR = 0x5A;
constexpr int SDA_PIN = 21;
constexpr int SCL_PIN = 22;
constexpr unsigned long SERIAL_BAUD_RATE = 115200;

constexpr uint8_t REG_MODE = 0x01;
constexpr uint8_t REG_RTP_INPUT = 0x02;
constexpr uint8_t REG_GO = 0x0C;
constexpr uint8_t REG_FEEDBACK = 0x1A;
constexpr uint8_t REG_CONTROL3 = 0x1D;

constexpr uint8_t MODE_INTERNAL_TRIGGER = 0x00;
constexpr uint8_t MODE_REALTIME_PLAYBACK = 0x05;

struct PatternStep {
  uint8_t amplitude;
  uint16_t durationMs;
};

struct PatternDefinition {
  const PatternStep *steps;
  uint8_t stepCount;
};

const PatternStep WAVEFORM_1[] = {
  {72, 70}, {0, 50}, {92, 90},
};
const PatternStep WAVEFORM_2[] = {
  {110, 120}, {0, 50}, {70, 90},
};
const PatternStep WAVEFORM_3[] = {
  {80, 45}, {0, 40}, {80, 45}, {0, 40}, {112, 110},
};
const PatternStep WAVEFORM_4[] = {
  {96, 200},
};
const PatternStep WAVEFORM_5[] = {
  {48, 80}, {0, 50}, {78, 80}, {0, 50}, {108, 120},
};
const PatternStep WAVEFORM_6[] = {
  {122, 60}, {0, 35}, {122, 60}, {0, 35}, {122, 60},
};
const PatternStep WAVEFORM_7[] = {
  {58, 180}, {0, 40}, {88, 150},
};
const PatternStep WAVEFORM_8[] = {
  {100, 70}, {0, 40}, {65, 70}, {0, 40}, {100, 120},
};
const PatternStep WAVEFORM_9[] = {
  {118, 45}, {0, 35}, {92, 45}, {0, 35}, {72, 120},
};
const PatternStep WAVEFORM_10[] = {
  {60, 100}, {0, 60}, {120, 180},
};
const PatternStep WAVEFORM_11[] = {
  {85, 90}, {0, 50}, {105, 90}, {0, 50}, {85, 90},
};
const PatternStep WAVEFORM_12[] = {
  {52, 70}, {0, 35}, {52, 70}, {0, 35}, {125, 150},
};
const PatternStep WAVEFORM_13[] = {
  {126, 90}, {0, 40}, {96, 120},
};
const PatternStep WAVEFORM_14[] = {
  {70, 55}, {0, 35}, {88, 55}, {0, 35}, {106, 55}, {0, 35}, {124, 120},
};
const PatternStep WAVEFORM_15[] = {
  {90, 260},
};

const PatternDefinition WAVEFORMS[15] = {
  {WAVEFORM_1, sizeof(WAVEFORM_1) / sizeof(WAVEFORM_1[0])},
  {WAVEFORM_2, sizeof(WAVEFORM_2) / sizeof(WAVEFORM_2[0])},
  {WAVEFORM_3, sizeof(WAVEFORM_3) / sizeof(WAVEFORM_3[0])},
  {WAVEFORM_4, sizeof(WAVEFORM_4) / sizeof(WAVEFORM_4[0])},
  {WAVEFORM_5, sizeof(WAVEFORM_5) / sizeof(WAVEFORM_5[0])},
  {WAVEFORM_6, sizeof(WAVEFORM_6) / sizeof(WAVEFORM_6[0])},
  {WAVEFORM_7, sizeof(WAVEFORM_7) / sizeof(WAVEFORM_7[0])},
  {WAVEFORM_8, sizeof(WAVEFORM_8) / sizeof(WAVEFORM_8[0])},
  {WAVEFORM_9, sizeof(WAVEFORM_9) / sizeof(WAVEFORM_9[0])},
  {WAVEFORM_10, sizeof(WAVEFORM_10) / sizeof(WAVEFORM_10[0])},
  {WAVEFORM_11, sizeof(WAVEFORM_11) / sizeof(WAVEFORM_11[0])},
  {WAVEFORM_12, sizeof(WAVEFORM_12) / sizeof(WAVEFORM_12[0])},
  {WAVEFORM_13, sizeof(WAVEFORM_13) / sizeof(WAVEFORM_13[0])},
  {WAVEFORM_14, sizeof(WAVEFORM_14) / sizeof(WAVEFORM_14[0])},
  {WAVEFORM_15, sizeof(WAVEFORM_15) / sizeof(WAVEFORM_15[0])},
};

const uint8_t anonymousIdToWaveformSlot[60] = {
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

String serialBuffer;

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(DRV2605_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(DRV2605_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(static_cast<int>(DRV2605_ADDR), 1);

  if (Wire.available()) {
    return Wire.read();
  }

  return 0;
}

void stopMotor() {
  writeRegister(REG_RTP_INPUT, 0);
  writeRegister(REG_GO, 0);
}

void setRealtimeAmplitude(uint8_t amplitude) {
  writeRegister(REG_RTP_INPUT, amplitude);
}

void initDrv2605() {
  delay(10);
  writeRegister(REG_MODE, MODE_INTERNAL_TRIGGER);
  delay(10);

  // Clear bit 7 so the device stays in ERM mode.
  uint8_t feedback = readRegister(REG_FEEDBACK);
  feedback &= 0x7F;
  writeRegister(REG_FEEDBACK, feedback);

  // Open-loop ERM tuning is a safe starting point for placeholder patterns.
  uint8_t control3 = readRegister(REG_CONTROL3);
  control3 |= 0x20;
  writeRegister(REG_CONTROL3, control3);

  writeRegister(REG_MODE, MODE_REALTIME_PLAYBACK);
  stopMotor();
}

void playWaveformSlot(uint8_t waveformSlot) {
  if (waveformSlot < 1 || waveformSlot > 15) {
    Serial.println("ERR:WAVEFORM_SLOT");
    return;
  }

  const PatternDefinition &pattern = WAVEFORMS[waveformSlot - 1];

  for (uint8_t index = 0; index < pattern.stepCount; index += 1) {
    setRealtimeAmplitude(pattern.steps[index].amplitude);
    delay(pattern.steps[index].durationMs);
  }

  stopMotor();
}

void handleAnonymousId(long anonymousId) {
  if (anonymousId < 1 || anonymousId > 60) {
    Serial.println("ERR:INVALID_ID");
    return;
  }

  uint8_t waveformSlot = anonymousIdToWaveformSlot[anonymousId - 1];
  playWaveformSlot(waveformSlot);
  Serial.print("PLAYED:");
  Serial.println(anonymousId);
}

void processSerialLine(const String &line) {
  if (line.length() == 0) {
    return;
  }

  char *endPtr = nullptr;
  long anonymousId = strtol(line.c_str(), &endPtr, 10);

  if (endPtr == line.c_str() || *endPtr != '\0') {
    Serial.println("ERR:PARSE");
    return;
  }

  handleAnonymousId(anonymousId);
}

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  Wire.begin(SDA_PIN, SCL_PIN);
  initDrv2605();
  serialBuffer.reserve(16);
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char incoming = static_cast<char>(Serial.read());

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      processSerialLine(serialBuffer);
      serialBuffer = "";
      continue;
    }

    serialBuffer += incoming;
  }
}

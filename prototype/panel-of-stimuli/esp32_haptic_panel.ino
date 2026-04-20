#include <Wire.h>

constexpr uint8_t DRV2605_ADDR = 0x5A;
constexpr int SDA_PIN = 21;
constexpr int SCL_PIN = 22;
constexpr unsigned long SERIAL_BAUD_RATE = 115200;
constexpr uint16_t DEFAULT_SAMPLE_INTERVAL_MS = 5;

constexpr uint8_t REG_MODE = 0x01;
constexpr uint8_t REG_RTP_INPUT = 0x02;
constexpr uint8_t REG_GO = 0x0C;
constexpr uint8_t REG_FEEDBACK = 0x1A;
constexpr uint8_t REG_CONTROL3 = 0x1D;

constexpr uint8_t MODE_INTERNAL_TRIGGER = 0x00;
constexpr uint8_t MODE_REALTIME_PLAYBACK = 0x05;

struct WaveformDefinition {
  const uint8_t *samples;
  uint16_t sampleCount;
  uint16_t sampleIntervalMs;
};

// Success block: anonymous IDs 1-15
// RANDOM NOISE
const uint8_t STIMULUS_01[] = {0};
const uint8_t STIMULUS_02[] = {0};
const uint8_t STIMULUS_03[] = {0};

//RANDOM VECTOR - VAE
const uint8_t STIMULUS_04[] = {0};
const uint8_t STIMULUS_05[] = {0};
const uint8_t STIMULUS_06[] = {0};

// HAPTIC GEN
const uint8_t STIMULUS_07[] = 
{
};
const uint8_t STIMULUS_08[] = {0};
const uint8_t STIMULUS_09[] = {0};

// LLM DIRECT
const uint8_t STIMULUS_10[] = {0};
const uint8_t STIMULUS_11[] = {0};
const uint8_t STIMULUS_12[] = {0};

// MY VAE
const uint8_t STIMULUS_13[] = {0};
const uint8_t STIMULUS_14[] = {0};
const uint8_t STIMULUS_15[] = {0};

// Error block: anonymous IDs 16-30
// RANDOM NOISE
const uint8_t STIMULUS_16[] = {0};
const uint8_t STIMULUS_17[] = {0};
const uint8_t STIMULUS_18[] = {0};

//RANDOM VECTOR - VAE
const uint8_t STIMULUS_19[] = {0};
const uint8_t STIMULUS_20[] = {0};
const uint8_t STIMULUS_21[] = {0};

// HAPTIC GEN
const uint8_t STIMULUS_22[] = {0};
const uint8_t STIMULUS_23[] = {0};
const uint8_t STIMULUS_24[] = {0};

// LLM DIRECT
const uint8_t STIMULUS_25[] = {0};
const uint8_t STIMULUS_26[] = {0};
const uint8_t STIMULUS_27[] = {0};

// MY VAE
const uint8_t STIMULUS_28[] = {0};
const uint8_t STIMULUS_29[] = {0};
const uint8_t STIMULUS_30[] = {0};

// Notification block: anonymous IDs 31-45
// RANDOM NOISE
const uint8_t STIMULUS_31[] = {0};
const uint8_t STIMULUS_32[] = {0};
const uint8_t STIMULUS_33[] = {0};

//RANDOM VECTOR - VAE
const uint8_t STIMULUS_34[] = {0};
const uint8_t STIMULUS_35[] = {0};
const uint8_t STIMULUS_36[] = {0};

// HAPTIC GEN
const uint8_t STIMULUS_37[] = {0};
const uint8_t STIMULUS_38[] = {0};
const uint8_t STIMULUS_39[] = {0};

// LLM DIRECT
const uint8_t STIMULUS_40[] = {0};
const uint8_t STIMULUS_41[] = {0};
const uint8_t STIMULUS_42[] = {0};

// MY VAE
const uint8_t STIMULUS_43[] = {0};
const uint8_t STIMULUS_44[] = {0};
const uint8_t STIMULUS_45[] = {0};

// Loading block: anonymous IDs 46-60
// RANDOM NOISE
const uint8_t STIMULUS_46[] = {0};
const uint8_t STIMULUS_47[] = {0};
const uint8_t STIMULUS_48[] = {0};

//RANDOM VECTOR - VAE
const uint8_t STIMULUS_49[] = {0};
const uint8_t STIMULUS_50[] = {0};
const uint8_t STIMULUS_51[] = {0};

// HAPTIC GEN
const uint8_t STIMULUS_52[] = {0};
const uint8_t STIMULUS_53[] = {0};
const uint8_t STIMULUS_54[] = {0};

// LLM DIRECT
const uint8_t STIMULUS_55[] = {0};
const uint8_t STIMULUS_56[] = {0};
const uint8_t STIMULUS_57[] = {0};

// MY VAE
const uint8_t STIMULUS_58[] = {0};
const uint8_t STIMULUS_59[] = {0};
const uint8_t STIMULUS_60[] = {0};

const WaveformDefinition STIMULI[60] = {
  // Success block: anonymous IDs 1-15
  {STIMULUS_01, sizeof(STIMULUS_01) / sizeof(STIMULUS_01[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_02, sizeof(STIMULUS_02) / sizeof(STIMULUS_02[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_03, sizeof(STIMULUS_03) / sizeof(STIMULUS_03[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_04, sizeof(STIMULUS_04) / sizeof(STIMULUS_04[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_05, sizeof(STIMULUS_05) / sizeof(STIMULUS_05[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_06, sizeof(STIMULUS_06) / sizeof(STIMULUS_06[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_07, sizeof(STIMULUS_07) / sizeof(STIMULUS_07[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_08, sizeof(STIMULUS_08) / sizeof(STIMULUS_08[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_09, sizeof(STIMULUS_09) / sizeof(STIMULUS_09[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_10, sizeof(STIMULUS_10) / sizeof(STIMULUS_10[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_11, sizeof(STIMULUS_11) / sizeof(STIMULUS_11[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_12, sizeof(STIMULUS_12) / sizeof(STIMULUS_12[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_13, sizeof(STIMULUS_13) / sizeof(STIMULUS_13[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_14, sizeof(STIMULUS_14) / sizeof(STIMULUS_14[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_15, sizeof(STIMULUS_15) / sizeof(STIMULUS_15[0]), DEFAULT_SAMPLE_INTERVAL_MS},

  // Error block: anonymous IDs 16-30
  {STIMULUS_16, sizeof(STIMULUS_16) / sizeof(STIMULUS_16[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_17, sizeof(STIMULUS_17) / sizeof(STIMULUS_17[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_18, sizeof(STIMULUS_18) / sizeof(STIMULUS_18[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_19, sizeof(STIMULUS_19) / sizeof(STIMULUS_19[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_20, sizeof(STIMULUS_20) / sizeof(STIMULUS_20[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_21, sizeof(STIMULUS_21) / sizeof(STIMULUS_21[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_22, sizeof(STIMULUS_22) / sizeof(STIMULUS_22[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_23, sizeof(STIMULUS_23) / sizeof(STIMULUS_23[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_24, sizeof(STIMULUS_24) / sizeof(STIMULUS_24[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_25, sizeof(STIMULUS_25) / sizeof(STIMULUS_25[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_26, sizeof(STIMULUS_26) / sizeof(STIMULUS_26[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_27, sizeof(STIMULUS_27) / sizeof(STIMULUS_27[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_28, sizeof(STIMULUS_28) / sizeof(STIMULUS_28[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_29, sizeof(STIMULUS_29) / sizeof(STIMULUS_29[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_30, sizeof(STIMULUS_30) / sizeof(STIMULUS_30[0]), DEFAULT_SAMPLE_INTERVAL_MS},

  // Notification block: anonymous IDs 31-45
  {STIMULUS_31, sizeof(STIMULUS_31) / sizeof(STIMULUS_31[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_32, sizeof(STIMULUS_32) / sizeof(STIMULUS_32[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_33, sizeof(STIMULUS_33) / sizeof(STIMULUS_33[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_34, sizeof(STIMULUS_34) / sizeof(STIMULUS_34[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_35, sizeof(STIMULUS_35) / sizeof(STIMULUS_35[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_36, sizeof(STIMULUS_36) / sizeof(STIMULUS_36[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_37, sizeof(STIMULUS_37) / sizeof(STIMULUS_37[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_38, sizeof(STIMULUS_38) / sizeof(STIMULUS_38[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_39, sizeof(STIMULUS_39) / sizeof(STIMULUS_39[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_40, sizeof(STIMULUS_40) / sizeof(STIMULUS_40[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_41, sizeof(STIMULUS_41) / sizeof(STIMULUS_41[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_42, sizeof(STIMULUS_42) / sizeof(STIMULUS_42[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_43, sizeof(STIMULUS_43) / sizeof(STIMULUS_43[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_44, sizeof(STIMULUS_44) / sizeof(STIMULUS_44[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_45, sizeof(STIMULUS_45) / sizeof(STIMULUS_45[0]), DEFAULT_SAMPLE_INTERVAL_MS},

  // Loading block: anonymous IDs 46-60
  {STIMULUS_46, sizeof(STIMULUS_46) / sizeof(STIMULUS_46[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_47, sizeof(STIMULUS_47) / sizeof(STIMULUS_47[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_48, sizeof(STIMULUS_48) / sizeof(STIMULUS_48[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_49, sizeof(STIMULUS_49) / sizeof(STIMULUS_49[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_50, sizeof(STIMULUS_50) / sizeof(STIMULUS_50[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_51, sizeof(STIMULUS_51) / sizeof(STIMULUS_51[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_52, sizeof(STIMULUS_52) / sizeof(STIMULUS_52[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_53, sizeof(STIMULUS_53) / sizeof(STIMULUS_53[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_54, sizeof(STIMULUS_54) / sizeof(STIMULUS_54[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_55, sizeof(STIMULUS_55) / sizeof(STIMULUS_55[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_56, sizeof(STIMULUS_56) / sizeof(STIMULUS_56[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_57, sizeof(STIMULUS_57) / sizeof(STIMULUS_57[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_58, sizeof(STIMULUS_58) / sizeof(STIMULUS_58[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_59, sizeof(STIMULUS_59) / sizeof(STIMULUS_59[0]), DEFAULT_SAMPLE_INTERVAL_MS},
  {STIMULUS_60, sizeof(STIMULUS_60) / sizeof(STIMULUS_60[0]), DEFAULT_SAMPLE_INTERVAL_MS},
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

  uint8_t feedback = readRegister(REG_FEEDBACK);
  feedback &= 0x7F;
  writeRegister(REG_FEEDBACK, feedback);

  uint8_t control3 = readRegister(REG_CONTROL3);
  control3 |= 0x20;
  writeRegister(REG_CONTROL3, control3);

  writeRegister(REG_MODE, MODE_REALTIME_PLAYBACK);
  stopMotor();
}

void playAnonymousStimulus(uint8_t anonymousId) {
  if (anonymousId < 1 || anonymousId > 60) {
    Serial.println("ERR:INVALID_ID");
    return;
  }

  const WaveformDefinition &waveform = STIMULI[anonymousId - 1];

  for (uint16_t index = 0; index < waveform.sampleCount; index += 1) {
    setRealtimeAmplitude(waveform.samples[index]);
    delay(waveform.sampleIntervalMs);
  }

  stopMotor();
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

  if (anonymousId < 1 || anonymousId > 60) {
    Serial.println("ERR:INVALID_ID");
    return;
  }

  playAnonymousStimulus(static_cast<uint8_t>(anonymousId));
  Serial.print("PLAYED:");
  Serial.println(anonymousId);
}

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
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

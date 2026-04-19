## Arduino Integration Contract

This prototype now uses anonymous vibration IDs `1..60` instead of exposing the underlying waveform label to participants.

### Browser -> ESP32

- Transport: Web Serial
- Baud rate: `115200`
- Payload format: newline-terminated ASCII integer
- Examples:
  - `1\n`
  - `27\n`
  - `60\n`

### Anonymous ID ranges

The browser sends one anonymous ID per replay or selection:

- Success block: `1..15`
- Error block: `16..30`
- Notification block: `31..45`
- Loading block: `46..60`

The included sketch now defines 60 independent placeholders using raw `uint8_t` waveform arrays:

- `STIMULUS_01`
- `STIMULUS_02`
- ...
- `STIMULUS_60`

You can paste your own exported arrays directly into each placeholder without changing the browser protocol.

Example format:

```cpp
const uint8_t STIMULUS_01[] = {
  236,248,238,239,248,238,236,245,
  239,235,247,237,223,237,255,213,
  107,138,87,112,85,81,87,88,
  0,0,0,0
};
```

The sketch replays each sample at `DEFAULT_SAMPLE_INTERVAL_MS`, which is currently set to `5 ms`.
If your exported data was generated with a different interval, update that constant in:

- [esp32_haptic_panel.ino](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\esp32_haptic_panel.ino)

### Included sketch

Use:

- [esp32_haptic_panel.ino](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\esp32_haptic_panel.ino)

The sketch includes:

- ESP32 serial parsing for anonymous IDs
- DRV2605L initialization on `SDA=21`, `SCL=22`, `0x5A`
- ERM configuration
- RTP playback mode
- 60 placeholder stimulus arrays
- Safe stop after playback
- Optional serial debug output:
  - `READY`
  - `PLAYED:<id>`
  - `ERR:INVALID_ID`
  - `ERR:PARSE`

### If you want to keep your own firmware

You can still use your own Arduino code as long as it follows this contract:

1. Read newline-terminated integers from serial.
2. Accept values `1..60`.
3. Map each anonymous ID to the real waveform you want to play, or store one array per ID.
4. Trigger the motor.
5. Stop the motor cleanly after playback.

### Where to change the browser side if needed

If you change the baud rate or serial payload format, update:

- [config.ts](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\src\app\config.ts)
- [useSerialConnection.ts](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\src\app\hooks\useSerialConnection.ts)

## Arduino Integration Contract

This prototype now uses anonymous vibration IDs `1..60` instead of exposing the underlying waveform slot to participants.

### Browser -> ESP32

- Transport: Web Serial
- Baud rate: `115200`
- Payload format: newline-terminated ASCII integer
- Examples:
  - `1\n`
  - `27\n`
  - `60\n`

### Anonymous ID mapping

The browser sends one anonymous ID per replay or selection:

- Success block: `1..15`
- Error block: `16..30`
- Notification block: `31..45`
- Loading block: `46..60`

The included sketch maps those 60 IDs back to 15 waveform slots with a repeated lookup table:

- `1..15 -> waveform 1..15`
- `16..30 -> waveform 1..15`
- `31..45 -> waveform 1..15`
- `46..60 -> waveform 1..15`

### Included sketch

Use:

- [esp32_haptic_panel.ino](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\esp32_haptic_panel.ino)

The sketch includes:

- ESP32 serial parsing for anonymous IDs
- DRV2605L initialization on `SDA=21`, `SCL=22`, `0x5A`
- ERM configuration
- RTP playback mode
- 15 placeholder waveform arrays
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
3. Map each anonymous ID to the real waveform slot you want to play.
4. Trigger the motor.
5. Stop the motor cleanly after playback.

### Where to change the browser side if needed

If you change the baud rate or serial payload format, update:

- [config.ts](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\src\app\config.ts)
- [useSerialConnection.ts](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\src\app\hooks\useSerialConnection.ts)

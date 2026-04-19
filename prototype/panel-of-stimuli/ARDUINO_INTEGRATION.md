## Arduino Integration Contract

This prototype does not require any specific firmware implementation, but the browser app expects the ESP32 sketch to follow this serial contract.

### Browser -> ESP32

- Transport: Web Serial
- Recommended baud rate: `115200`
- Payload format: newline-terminated ASCII integer
- Examples:
  - `1\n`
  - `7\n`
  - `15\n`

### Expected firmware behavior

When your ESP32 receives a valid integer:

1. Parse the integer value.
2. Validate that it is in the range `1..15`.
3. Map that value to your corresponding preloaded waveform.
4. Play the haptic on the DRV2605L / motor.
5. Stop the motor cleanly at the end of playback.

### Recommended parser shape

- Ignore carriage returns.
- Use newline as the end-of-message marker.
- Reject empty lines or out-of-range IDs.
- It is okay if the firmware is blocking during playback, as long as the motor is safely stopped after the waveform finishes.

### Optional acknowledgements

The current browser app does not require a response, but these optional serial prints are useful during debugging:

- `READY`
- `PLAYED:7`
- `ERR:INVALID_STIMULUS`
- `ERR:PARSE`

### Hardware assumptions from the study design

- ESP32
- DRV2605L
- ERM motor
- SDA = `21`
- SCL = `22`
- I2C address = `0x5A`

### Where to change the browser config if needed

If your firmware uses a different baud rate or message format, update:

- [config.ts](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\src\app\config.ts)
- [useSerialConnection.ts](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\src\app\hooks\useSerialConnection.ts)

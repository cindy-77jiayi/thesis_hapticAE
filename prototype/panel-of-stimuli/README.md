## Panel of Stimuli v2

This directory contains the local-first user-study prototype for the haptic thesis project.

### What it does

- Runs a block-based study with 4 fixed UI flows:
  - success
  - error
  - notification
  - loading
- Shows 15 anonymous vibration cards per flow in a `3 x 5` comparison grid
- Keeps the UI flow viewport fixed in size and position
- Sends anonymous IDs `1..60` to an ESP32 through Web Serial
- Captures 2 Likert ratings per anonymous ID
- Captures one overview per flow:
  - Best match selections, 1 to 3 anonymous IDs
  - Worst match selections, 1 to 3 anonymous IDs
  - Optional open-text reasons
- Exports two CSV files:
  - detailed block ratings
  - flow overview selections

### Run locally

1. `npm install`
2. `npm run dev`
3. Open the local Vite URL in Chrome or Edge

Web Serial requires a browser with serial support and a local secure context such as `http://localhost:5173`.

### Configure Google Forms

Edit [config.ts](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\src\app\config.ts) and set:

- `googleFormUrl`
- `participantPrefillEntry`

If these stay blank, the prototype still works and CSV export is unaffected.

### Arduino integration

Use the included sketch:

- [esp32_haptic_panel.ino](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\esp32_haptic_panel.ino)

Browser payload format:

- baud rate `115200`
- newline-terminated anonymous integer
- valid messages are `1..60`
- example: `27\n`

The ESP32 sketch maps each anonymous ID back to one of 15 waveform slots.

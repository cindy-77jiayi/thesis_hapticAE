## Panel of Stimuli Study Prototype

This directory contains a local-first user-study prototype for the haptic thesis project.

### What it does

- Runs a 60-trial haptic study locally in the browser
- Uses Web Serial to send stimulus IDs `1-15` to an ESP32
- Reuses the existing Figma/React UI flow visuals for success, error, notification, and loading
- Captures five Likert ratings per trial
- Exports CSV with the schema:
  `participant_id,trial_index,stimulus_id,ui_flow,rating_match,rating_pleasant,rating_clarity,rating_quality,rating_preference,timestamp`
- Opens Google Forms at the end when configured

### Run locally

1. `npm install`
2. `npm run dev`
3. Open the printed local URL in Chrome or Edge

Web Serial requires a secure context, so use the Vite dev server URL such as `http://localhost:5173`.

### Configure Google Forms

Edit [config.ts](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\src\app\config.ts) and fill in:

- `googleFormUrl`
- `participantPrefillEntry`

If those values are left blank, the study still runs and CSV export still works.

### Arduino integration

Use your own firmware. The front end only assumes:

- baud rate `115200`
- newline-terminated ASCII integers
- valid messages are `1` through `15`
- one stimulus trigger per line, for example `7\n`

See [ARDUINO_INTEGRATION.md](C:\Users\11604\Desktop\thesis\prototype\panel-of-stimuli\ARDUINO_INTEGRATION.md) for the exact browser-side expectations.

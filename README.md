# Sentinel — On-device Visual Security Console

Sentinel turns a single webcam (or, with light changes, any browser-accessible
camera feed) into a live security monitoring console. Everything runs
client-side in the browser:

- **Object detection** (TensorFlow.js / COCO-SSD) — tracks people, bags,
  vehicles, etc. in real time.
- **Zone-breach detection** — draw a restricted area on the feed; get an
  alert when a person enters it.
- **Loitering detection** — flags a person who lingers in frame past a
  configurable time threshold.
- **Unattended-object detection** — flags a bag/case left with no one
  nearby for too long.
- **Watchlist face matching** (face-api.js) — enroll a photo of a person you
  don't want to let in; get alerted (with a snapshot) if they appear on
  camera. Face signatures are computed and stored on-device only.
- **Incident log** — every alert is timestamped with a cropped snapshot, and
  optionally rewritten as a one-line plain-English note via a short,
  text-only Gemini call (no image/video data is ever sent).

No video frame is uploaded anywhere. Model weights are fetched once from
public CDNs on first load; detection and face matching run locally after
that.

## Run Locally

**Prerequisites:** Node.js

1. Install dependencies:
   `npm install`
2. Set `GEMINI_API_KEY` in `.env.local` to your Gemini API key (optional —
   only used for the one-line incident narratives; the app works without it).
3. Run the app:
   `npm run dev`

## Production build

```
npm run build
npm start   # serves dist/ via the included Express server
```

## Notes for deployment

- Face recognition (the watchlist feature) is legally regulated in many
  jurisdictions (e.g. Illinois' BIPA, EU GDPR's biometric-data rules).
  Confirm you have a legal basis before enabling it for a given site, and
  keep a human reviewing matches before any action is taken.
- The bundled `WATCHLIST_MATCH_DISTANCE` threshold is a reasonable default,
  not a certified accuracy guarantee — tune it and validate it for your
  camera/lighting conditions before relying on it.

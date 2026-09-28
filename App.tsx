import React, { useEffect, useRef, useState, useCallback } from 'react';
import { GoogleGenAI } from '@google/genai';
import {
  Camera, Square, Play, Loader2, AlertCircle, Info, X, ShieldAlert, Shield,
  Users, Crosshair, Plus, Trash2, Bell, CheckCircle2, Upload, Clock, ScanFace,
  UserX, PackageX,
} from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import * as faceapi from '@vladmandic/face-api';
import { motion, AnimatePresence } from 'motion/react';
import { GitGroupLogo } from './GitGroupLogo';

// ---------------------------------------------------------------------------
// Git Group — Home of Technology. On-device computer-vision security console.
//
// Everything below runs client-side: object detection (COCO-SSD), face
// landmark tracking (MediaPipe) and face recognition (face-api.js) all
// execute in the browser. No video frame is ever uploaded anywhere. The only
// network calls this app makes at runtime are (a) one-time model downloads,
// and (b) optional short text-only calls to Gemini to turn a structured
// alert into a one-line human-readable note for the incident log.
// ---------------------------------------------------------------------------

const FACEAPI_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';

const UNATTENDED_CLASSES = new Set(['backpack', 'handbag', 'suitcase']);

const LOITER_THRESHOLD_MS = 15000;
const LOITER_COOLDOWN_MS = 30000;
const UNATTENDED_THRESHOLD_MS = 12000;
const UNATTENDED_COOLDOWN_MS = 45000;
const ZONE_COOLDOWN_MS = 15000;
const WATCHLIST_COOLDOWN_MS = 60000;
const WATCHLIST_MATCH_DISTANCE = 0.55; // face-api.js euclidean distance; lower = stricter
const MAX_ALERTS = 60;
const WATCHLIST_STORAGE_KEY = 'sentinel_watchlist_v1';

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

interface SmoothedBox {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  score: number;
  opacity: number;
  labelX: number;
  labelY: number;
  firstSeen: number;
  aloneSince: number | null;
}

interface Zone {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface WatchlistPerson {
  id: string;
  name: string;
  reason: string;
  descriptor: number[];
  photo: string;
  addedAt: number;
}

type AlertType = 'ZONE_BREACH' | 'LOITERING' | 'UNATTENDED_OBJECT' | 'WATCHLIST_MATCH' | 'SYSTEM';
type Severity = 'info' | 'warning' | 'critical';

interface SecurityAlert {
  id: string;
  type: AlertType;
  severity: Severity;
  message: string;
  narrative?: string;
  snapshot?: string;
  timestamp: number;
}

const ALERT_ICON: Record<AlertType, React.ElementType> = {
  ZONE_BREACH: Crosshair,
  LOITERING: Clock,
  UNATTENDED_OBJECT: PackageX,
  WATCHLIST_MATCH: UserX,
  SYSTEM: Shield,
};

const SEVERITY_COLOR: Record<Severity, string> = {
  info: 'text-white/70 border-white/20',
  warning: 'text-amber-400 border-amber-500/50',
  critical: 'text-red-400 border-red-500/60',
};

// Best-effort, text-only call to Gemini to turn a structured alert into a
// short plain-English incident note. Never sends image/video data. Falls
// back silently (the raw `message` is already shown) if no key is set or
// the call fails, so this is a nice-to-have, not a dependency.
const generateIncidentNarrative = async (type: AlertType, message: string): Promise<string | undefined> => {
  try {
    const apiKey = process.env.GEMINI_API_KEY || process.env.API_KEY;
    if (!apiKey) return undefined;
    const ai = new GoogleGenAI({ apiKey });
    const prompt = `You are a security operations assistant. Rewrite this automated alert as a single, plain-English sentence (max 25 words) for a security incident log. Do not speculate about intent, identity, or danger beyond what is stated. Alert type: ${type}. Details: ${message}`;
    const response = await ai.models.generateContent({
      model: 'gemini-flash-lite-latest',
      contents: prompt,
    });
    return response.text?.trim();
  } catch (e) {
    console.warn('Incident narrative generation failed (non-critical):', e);
    return undefined;
  }
};

export default function App() {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [status, setStatus] = useState('Loading Detection Models...');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const [isWatchlistOpen, setIsWatchlistOpen] = useState(false);
  const [isDefiningZone, setIsDefiningZone] = useState(false);
  const [zone, setZone] = useState<Zone | null>(null);
  const [alerts, setAlerts] = useState<SecurityAlert[]>([]);
  const [watchlist, setWatchlist] = useState<WatchlistPerson[]>([]);
  const [peopleCount, setPeopleCount] = useState(0);
  const [objectsSeen, setObjectsSeen] = useState<string[]>([]);
  const [enrollName, setEnrollName] = useState('');
  const [enrollReason, setEnrollReason] = useState('');
  const [enrollBusy, setEnrollBusy] = useState(false);
  const [enrollError, setEnrollError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const faceCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const objectModelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const faceApiModelsLoadedRef = useRef(false);

  const isMonitoringRef = useRef(false);
  const detectLoopRef = useRef<number | null>(null);
  const smoothedBoxesRef = useRef<Map<string, SmoothedBox>>(new Map());
  const alertCooldownRef = useRef<Map<string, number>>(new Map());
  const lastFaceScanAtRef = useRef<number>(0);
  const zoneRef = useRef<Zone | null>(null);
  const watchlistRef = useRef<WatchlistPerson[]>([]);
  const drawingZoneRef = useRef<{ x: number; y: number } | null>(null);

  // --- Load watchlist from local storage (client-side only, never leaves the device) ---
  useEffect(() => {
    try {
      const raw = localStorage.getItem(WATCHLIST_STORAGE_KEY);
      if (raw) setWatchlist(JSON.parse(raw));
    } catch (e) {
      console.warn('Could not load saved watchlist:', e);
    }
  }, []);

  useEffect(() => {
    watchlistRef.current = watchlist;
    try {
      localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(watchlist));
    } catch (e) {
      console.warn('Could not persist watchlist:', e);
    }
  }, [watchlist]);

  useEffect(() => {
    zoneRef.current = zone;
  }, [zone]);

  // --- Load models on mount ---
  useEffect(() => {
    const loadModels = async () => {
      try {
        await tf.ready();
        const cocoModel = await cocoSsd.load();
        objectModelRef.current = cocoModel;

        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
        );
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          },
          outputFaceBlendshapes: false,
          runningMode: 'VIDEO',
          numFaces: 1,
        });
        faceLandmarkerRef.current = faceLandmarker;

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(FACEAPI_MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(FACEAPI_MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(FACEAPI_MODEL_URL),
        ]);
        faceApiModelsLoadedRef.current = true;

        setIsModelLoaded(true);
        setStatus('Idle');
      } catch (err: any) {
        console.error('Failed to load models:', err);
        setStatus('Error loading models');
        setErrorMsg(err.message || 'Failed to load detection models.');
      }
    };

    loadModels();

    return () => {
      stopSession();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Snapshot capture for the incident log ---
  const captureSnapshot = useCallback((box?: { x: number; y: number; width: number; height: number }): string | undefined => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return undefined;

    let sx = 0, sy = 0, sw = video.videoWidth, sh = video.videoHeight;
    if (box) {
      const pad = 0.35;
      sx = Math.max(0, box.x - box.width * pad);
      sy = Math.max(0, box.y - box.height * pad);
      sw = Math.min(video.videoWidth - sx, box.width * (1 + pad * 2));
      sh = Math.min(video.videoHeight - sy, box.height * (1 + pad * 2));
    }
    if (sw <= 0 || sh <= 0) return undefined;

    const outW = 320;
    const outH = Math.max(1, Math.round(outW * (sh / sw)));
    const temp = document.createElement('canvas');
    temp.width = outW;
    temp.height = outH;
    const tctx = temp.getContext('2d');
    if (!tctx) return undefined;
    tctx.drawImage(video, sx, sy, sw, sh, 0, 0, outW, outH);
    return temp.toDataURL('image/jpeg', 0.7);
  }, []);

  // --- Central alert dispatcher (deduped with a per-cause cooldown) ---
  const pushAlert = useCallback((
    type: AlertType,
    severity: Severity,
    message: string,
    cooldownKey: string,
    cooldownMs: number,
    snapshot?: string
  ) => {
    const now = Date.now();
    const last = alertCooldownRef.current.get(cooldownKey) || 0;
    if (now - last < cooldownMs) return;
    alertCooldownRef.current.set(cooldownKey, now);

    const alert: SecurityAlert = {
      id: `${now}-${Math.random().toString(36).slice(2, 8)}`,
      type,
      severity,
      message,
      snapshot,
      timestamp: now,
    };
    setAlerts((prev) => [alert, ...prev].slice(0, MAX_ALERTS));

    if (type !== 'SYSTEM') {
      generateIncidentNarrative(type, message).then((narrative) => {
        if (narrative) {
          setAlerts((prev) => prev.map((a) => (a.id === alert.id ? { ...a, narrative } : a)));
        }
      });
    }
  }, []);

  const dismissAlert = (id: string) => setAlerts((prev) => prev.filter((a) => a.id !== id));
  const clearAllAlerts = () => setAlerts([]);

  // --- Watchlist face matching (throttled, separate from the object-detection loop) ---
  const runWatchlistScan = useCallback(async (video: HTMLVideoElement) => {
    if (!faceApiModelsLoadedRef.current || watchlistRef.current.length === 0) return;
    try {
      const detections = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
        .withFaceLandmarks()
        .withFaceDescriptors();

      for (const det of detections) {
        let best: { person: WatchlistPerson; distance: number } | null = null;
        for (const person of watchlistRef.current) {
          const distance = faceapi.euclideanDistance(det.descriptor, new Float32Array(person.descriptor));
          if (distance < WATCHLIST_MATCH_DISTANCE && (!best || distance < best.distance)) {
            best = { person, distance };
          }
        }
        if (best) {
          const box = det.detection.box;
          const snapshot = captureSnapshot({ x: box.x, y: box.y, width: box.width, height: box.height });
          const similarity = Math.max(1, Math.min(99, Math.round((1 - best.distance) * 100)));
          pushAlert(
            'WATCHLIST_MATCH',
            'critical',
            `Watchlist match: ${best.person.name}${best.person.reason ? ` (${best.person.reason})` : ''} — approx. ${similarity}% similarity, verify before acting.`,
            `watchlist:${best.person.id}`,
            WATCHLIST_COOLDOWN_MS,
            snapshot
          );
        }
      }
    } catch (e) {
      console.error('Watchlist scan error:', e);
    }
  }, [captureSnapshot, pushAlert]);

  // --- Decorative + informational face-mesh scan panel (no emotion inference) ---
  const drawFaceScanViz = (video: HTMLVideoElement) => {
    const faceCanvas = faceCanvasRef.current;
    if (!faceCanvas || !faceLandmarkerRef.current) return;
    const fCtx = faceCanvas.getContext('2d');
    if (!fCtx) return;

    const faceResult = faceLandmarkerRef.current.detectForVideo(video, performance.now());
    fCtx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);

    if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
      const landmarks = faceResult.faceLandmarks[0];
      const time = performance.now() / 1500;

      let minX = video.videoWidth, maxX = 0, minY = video.videoHeight, maxY = 0;
      for (const pt of landmarks) {
        const px = pt.x * video.videoWidth;
        const py = pt.y * video.videoHeight;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      }
      const faceWidth = maxX - minX;
      const faceHeight = maxY - minY;
      const centerX = minX + faceWidth / 2;
      const centerY = minY + faceHeight / 2;
      const scanY = minY + ((Math.sin(time) + 1) / 2) * faceHeight;
      const scale = Math.min(faceCanvas.width / faceWidth, faceCanvas.height / faceHeight) * 0.8;

      for (const pt of landmarks) {
        const px = pt.x * video.videoWidth;
        const py = pt.y * video.videoHeight;
        const dist = Math.abs(py - scanY) / faceHeight;
        const opacity = Math.max(0.15, 1.0 - dist * 4);
        fCtx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
        fCtx.beginPath();
        const drawX = faceCanvas.width / 2 + (px - centerX) * scale;
        const drawY = faceCanvas.height / 2 + (py - centerY) * scale;
        fCtx.arc(drawX, drawY, 1.5, 0, 2 * Math.PI);
        fCtx.fill();
      }
    }
  };

  // --- Main per-frame detection loop ---
  const runDetection = async () => {
    if (!isMonitoringRef.current || !videoRef.current || !canvasRef.current || !objectModelRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (video.readyState >= 2 && ctx) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      try {
        const predictions = await objectModelRef.current.detect(video);
        const now = Date.now();

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // --- Tracking / smoothing ---
        const newSmoothedBoxes = new Map<string, SmoothedBox>();
        const unassignedPredictions = [...predictions];

        smoothedBoxesRef.current.forEach((box, id) => {
          let closestIdx = -1;
          let minDist = Infinity;
          unassignedPredictions.forEach((pred, idx) => {
            if (pred.class === box.class) {
              const [px, py, pw, ph] = pred.bbox;
              const dist = Math.hypot(px + pw / 2 - (box.x + box.width / 2), py + ph / 2 - (box.y + box.height / 2));
              if (dist < 150 && dist < minDist) {
                minDist = dist;
                closestIdx = idx;
              }
            }
          });

          if (closestIdx !== -1) {
            const pred = unassignedPredictions[closestIdx];
            const [px, py, pw, ph] = pred.bbox;
            const lerp = 0.2;
            box.x += (px - box.x) * lerp;
            box.y += (py - box.y) * lerp;
            box.width += (pw - box.width) * lerp;
            box.height += (ph - box.height) * lerp;
            box.opacity = Math.min(1, box.opacity + 0.12);
            box.score = pred.score;
            const targetLabelX = box.x + box.width + 20;
            const targetLabelY = box.y - 20;
            box.labelX += (targetLabelX - box.labelX) * lerp;
            box.labelY += (targetLabelY - box.labelY) * lerp;
            newSmoothedBoxes.set(id, box);
            unassignedPredictions.splice(closestIdx, 1);
          } else {
            box.opacity -= 0.06;
            if (box.opacity > 0) newSmoothedBoxes.set(id, box);
          }
        });

        unassignedPredictions.forEach((pred) => {
          if (pred.score < 0.5) return;
          const id = Math.random().toString(36).substring(7);
          const [x, y, width, height] = pred.bbox;
          newSmoothedBoxes.set(id, {
            x, y, width, height, class: pred.class, score: pred.score, opacity: 0,
            labelX: x + width + 40, labelY: y - 40,
            firstSeen: now, aloneSince: null,
          });
        });

        smoothedBoxesRef.current = newSmoothedBoxes;

        // --- Draw the restricted zone, if defined ---
        const currentZone = zoneRef.current;
        if (currentZone) {
          const zw = currentZone.x2 - currentZone.x1;
          const zh = currentZone.y2 - currentZone.y1;
          ctx.save();
          ctx.strokeStyle = 'rgba(239,68,68,0.7)';
          ctx.lineWidth = 2;
          ctx.setLineDash([8, 6]);
          ctx.strokeRect(currentZone.x1, currentZone.y1, zw, zh);
          ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(239,68,68,0.08)';
          ctx.fillRect(currentZone.x1, currentZone.y1, zw, zh);
          ctx.font = '700 11px "JetBrains Mono", monospace';
          ctx.fillStyle = 'rgba(239,68,68,0.9)';
          ctx.fillText('RESTRICTED ZONE', currentZone.x1 + 6, currentZone.y1 + 16);
          ctx.restore();
        }

        // --- Per-object analysis, drawing, alerting ---
        let personCount = 0;
        const classesSeen = new Set<string>();

        smoothedBoxesRef.current.forEach((box, id) => {
          classesSeen.add(box.class);
          const centerX = box.x + box.width / 2;
          const centerY = box.y + box.height / 2;

          let inZone = false;
          if (currentZone && box.class === 'person') {
            inZone = centerX >= currentZone.x1 && centerX <= currentZone.x2 && centerY >= currentZone.y1 && centerY <= currentZone.y2;
          }

          let isLoitering = false;
          if (box.class === 'person') {
            if (box.opacity > 0.5) personCount += 1;
            if (now - box.firstSeen > LOITER_THRESHOLD_MS) isLoitering = true;
          }

          let isUnattended = false;
          if (UNATTENDED_CLASSES.has(box.class)) {
            let nearestPersonDist = Infinity;
            smoothedBoxesRef.current.forEach((other) => {
              if (other.class === 'person' && other.opacity > 0.3) {
                const d = Math.hypot((other.x + other.width / 2) - centerX, (other.y + other.height / 2) - centerY);
                if (d < nearestPersonDist) nearestPersonDist = d;
              }
            });
            if (nearestPersonDist > 220) {
              if (box.aloneSince === null) box.aloneSince = now;
              if (now - box.aloneSince > UNATTENDED_THRESHOLD_MS) isUnattended = true;
            } else {
              box.aloneSince = null;
            }
          }

          if (inZone) {
            pushAlert('ZONE_BREACH', 'warning', 'A person entered the restricted zone.', `zone:${id}`, ZONE_COOLDOWN_MS, captureSnapshot(box));
          }
          if (isLoitering) {
            pushAlert('LOITERING', 'warning', `A person has remained in view for over ${Math.round(LOITER_THRESHOLD_MS / 1000)}s.`, `loiter:${id}`, LOITER_COOLDOWN_MS, captureSnapshot(box));
          }
          if (isUnattended) {
            pushAlert('UNATTENDED_OBJECT', 'critical', `An unattended ${box.class} was left with no one nearby.`, `unattended:${id}`, UNATTENDED_COOLDOWN_MS, captureSnapshot(box));
          }

          // --- Draw HUD box ---
          const { x, y, width, height, opacity, labelX, labelY } = box;
          const flagged = inZone || isLoitering || isUnattended;
          const color = flagged ? '239,68,68' : '255,255,255';
          const statusTag = inZone ? ' · ZONE BREACH' : isUnattended ? ' · UNATTENDED' : isLoitering ? ' · LOITERING' : '';
          const text = `${box.class.toUpperCase()} (${Math.round(box.score * 100)}%)${statusTag}`;

          ctx.strokeStyle = `rgba(${color}, ${opacity * 0.9})`;
          ctx.lineWidth = flagged ? 2 : 1;

          const cornerLength = Math.min(15, width / 4, height / 4);
          ctx.beginPath();
          ctx.moveTo(x, y + cornerLength); ctx.lineTo(x, y); ctx.lineTo(x + cornerLength, y);
          ctx.moveTo(x + width - cornerLength, y); ctx.lineTo(x + width, y); ctx.lineTo(x + width, y + cornerLength);
          ctx.moveTo(x + width, y + height - cornerLength); ctx.lineTo(x + width, y + height); ctx.lineTo(x + width - cornerLength, y + height);
          ctx.moveTo(x + cornerLength, y + height); ctx.lineTo(x, y + height); ctx.lineTo(x, y + height - cornerLength);
          ctx.stroke();

          if (flagged) {
            ctx.strokeStyle = `rgba(${color}, ${opacity * 0.4})`;
            ctx.setLineDash([3, 3]);
            ctx.strokeRect(x, y, width, height);
            ctx.setLineDash([]);
          }

          ctx.beginPath();
          ctx.moveTo(x + width, y);
          ctx.lineTo(labelX, labelY + 16);
          ctx.strokeStyle = `rgba(${color}, ${opacity * 0.5})`;
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.font = '400 10px "JetBrains Mono", monospace';
          const textWidth = ctx.measureText(text).width;
          ctx.fillStyle = `rgba(${color}, ${opacity * 0.25})`;
          ctx.fillRect(labelX, labelY, textWidth + 8, 16);
          ctx.fillStyle = `rgba(${color}, ${opacity})`;
          ctx.fillText(text, labelX + 4, labelY + 11);
        });

        setPeopleCount(personCount);
        setObjectsSeen(Array.from(classesSeen).sort());

        // --- Throttled watchlist face-recognition pass ---
        if (now - lastFaceScanAtRef.current > 800) {
          lastFaceScanAtRef.current = now;
          runWatchlistScan(video);
        }

        // --- Decorative biometric scan panel ---
        drawFaceScanViz(video);
      } catch (err) {
        console.error('Detection error:', err);
      }
    }

    if (isMonitoringRef.current) {
      detectLoopRef.current = requestAnimationFrame(runDetection);
    }
  };

  const startSession = async () => {
    if (!isModelLoaded) return;

    try {
      setErrorMsg(null);
      setStatus('Starting camera...');

      let stream = streamRef.current;
      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          });
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play().catch((e) => console.error('Video play error:', e));
          }
          setIsCameraActive(true);
        } catch (camErr: any) {
          console.error('Camera error:', camErr);
          setStatus('Camera Error');
          setErrorMsg('Camera access denied. Please allow camera access in your browser settings, then refresh the page.');
          return;
        }
      }

      isMonitoringRef.current = true;
      setIsMonitoring(true);
      setStatus('Monitoring Active');
      detectLoopRef.current = requestAnimationFrame(runDetection);
      pushAlert('SYSTEM', 'info', 'Monitoring session started.', `system:start:${Date.now()}`, 1);
    } catch (err: any) {
      console.error('Setup Error:', err);
      setStatus('Failed to start');
      setErrorMsg(err.message || 'An unknown error occurred during setup.');
    }
  };

  const stopSession = (closeCamera: boolean = true) => {
    if (isMonitoringRef.current) {
      pushAlert('SYSTEM', 'info', 'Monitoring session stopped.', `system:stop:${Date.now()}`, 1);
    }
    isMonitoringRef.current = false;
    setIsMonitoring(false);

    if (detectLoopRef.current) {
      cancelAnimationFrame(detectLoopRef.current);
      detectLoopRef.current = null;
    }
    smoothedBoxesRef.current.clear();
    setPeopleCount(0);
    setObjectsSeen([]);
    setStatus('Idle');

    if (closeCamera && streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setIsCameraActive(false);
    }
  };

  // --- Zone drawing (drag directly on the video canvas) ---
  const canvasPointFromEvent = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
  };

  const handleZonePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDefiningZone) return;
    const pt = canvasPointFromEvent(e);
    if (pt) drawingZoneRef.current = pt;
  };

  const handleZonePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDefiningZone || !drawingZoneRef.current) return;
    const pt = canvasPointFromEvent(e);
    if (!pt) return;
    const start = drawingZoneRef.current;
    setZone({
      x1: Math.min(start.x, pt.x), y1: Math.min(start.y, pt.y),
      x2: Math.max(start.x, pt.x), y2: Math.max(start.y, pt.y),
    });
  };

  const handleZonePointerUp = () => {
    if (!isDefiningZone) return;
    drawingZoneRef.current = null;
    setIsDefiningZone(false);
    setInfoMsg('Restricted zone updated.');
  };

  // --- Watchlist enrollment ---
  const handleEnrollPhoto = async (file: File) => {
    setEnrollError(null);
    setEnrollBusy(true);
    try {
      if (!faceApiModelsLoadedRef.current) {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(FACEAPI_MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(FACEAPI_MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(FACEAPI_MODEL_URL),
        ]);
        faceApiModelsLoadedRef.current = true;
      }

      const imgUrl = URL.createObjectURL(file);
      const img = await faceapi.fetchImage(imgUrl);
      const detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();
      URL.revokeObjectURL(imgUrl);

      if (!detection) {
        setEnrollError('No face detected in that photo. Use a clear, front-facing, well-lit photo.');
        setEnrollBusy(false);
        return;
      }

      const size = 160;
      const pad = 0.4;
      const box = detection.detection.box;
      const temp = document.createElement('canvas');
      temp.width = size;
      temp.height = size;
      const tctx = temp.getContext('2d');
      if (tctx) {
        tctx.drawImage(
          img,
          Math.max(0, box.x - box.width * pad), Math.max(0, box.y - box.height * pad),
          box.width * (1 + pad * 2), box.height * (1 + pad * 2),
          0, 0, size, size
        );
      }

      const entry: WatchlistPerson = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        name: enrollName.trim() || 'Unnamed',
        reason: enrollReason.trim(),
        descriptor: Array.from(detection.descriptor),
        photo: temp.toDataURL('image/jpeg', 0.85),
        addedAt: Date.now(),
      };
      setWatchlist((prev) => [...prev, entry]);
      setEnrollName('');
      setEnrollReason('');
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (e: any) {
      console.error(e);
      setEnrollError(e.message || 'Failed to process that photo.');
    } finally {
      setEnrollBusy(false);
    }
  };

  const removeWatchlistEntry = (id: string) => setWatchlist((prev) => prev.filter((p) => p.id !== id));

  return (
    <div className="h-[100dvh] w-full bg-black text-white flex overflow-hidden font-mono relative">
      {/* Background Camera Feed */}
      <div className="absolute inset-0 z-0">
        {!isCameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white/50 z-10 font-mono text-sm">
            <Camera className="w-8 h-8 mb-4 opacity-50" />
            <p>SYSTEM.CAMERA_OFFLINE</p>
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover grayscale contrast-125 opacity-60 transition-opacity duration-500 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
        />
        <canvas
          ref={canvasRef}
          onPointerDown={handleZonePointerDown}
          onPointerMove={handleZonePointerMove}
          onPointerUp={handleZonePointerUp}
          className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-500 z-[15] ${isCameraActive ? 'opacity-100' : 'opacity-0'} ${isDefiningZone ? 'pointer-events-auto cursor-crosshair' : 'pointer-events-none'}`}
        />
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.8)_100%)] z-10" />
        <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] z-10" />
      </div>

      {/* Overlays */}
      <div className="relative z-20 w-full h-full pointer-events-auto p-4 sm:p-6 overflow-y-auto overflow-x-hidden pb-32 sm:pb-6">
        <div className="flex flex-col lg:flex-row justify-between gap-4 min-h-full">

          {/* Left Column */}
          <div className="contents lg:flex lg:flex-col lg:justify-between w-full lg:w-80 pointer-events-none shrink-0">
            <div className="flex flex-col gap-4 shrink-0 order-1 lg:order-none pointer-events-auto">
              <div className="flex flex-col items-start gap-4 shrink-0">
                <div className="flex items-start justify-between w-full">
                  <div>
                    <h1 className="text-2xl font-bold tracking-tighter text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.8)] flex items-center gap-2">
                      <GitGroupLogo className="w-7 h-7" /> GIT GROUP
                    </h1>
                    <p className="text-[10px] text-white/70 font-mono uppercase tracking-widest">Home of Technology · Visual Security Console</p>
                  </div>
                  <button
                    onClick={() => setIsInfoOpen(true)}
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-full transition-colors backdrop-blur-md border border-white/20 shrink-0"
                    title="App Information"
                  >
                    <Info className="w-5 h-5 text-white" />
                  </button>
                </div>
                <div className="text-xs font-mono text-white/80 flex items-center gap-2 bg-black/40 backdrop-blur px-3 py-1.5 border border-white/20">
                  <div className={`w-2 h-2 rounded-none ${status === 'Monitoring Active' ? 'bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]' : status.includes('Starting') ? 'bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.8)]' : status === 'Loading Detection Models...' ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]' : status.includes('Error') ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]' : 'bg-zinc-600'}`} />
                  {status}
                </div>
              </div>

              {/* Controls */}
              <div className="flex flex-col items-stretch gap-2 shrink-0">
                <button
                  onClick={() => (isMonitoring ? stopSession(true) : startSession())}
                  disabled={!isModelLoaded}
                  className={`flex justify-center items-center gap-3 px-6 py-4 font-mono text-sm font-bold uppercase tracking-widest transition-all duration-300 border-2 backdrop-blur-md ${
                    isMonitoring
                      ? 'bg-red-500/20 text-red-400 border-red-500 hover:bg-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.4)]'
                      : 'bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_20px_rgba(255,255,255,0.3)]'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {!isModelLoaded ? (
                    <><Loader2 className="w-5 h-5 animate-spin" /> INITIALIZING...</>
                  ) : isMonitoring ? (
                    <><Square className="w-5 h-5 fill-current" /> STOP MONITORING</>
                  ) : (
                    <><Play className="w-5 h-5 fill-current" /> START MONITORING</>
                  )}
                </button>

                <div className="flex gap-2">
                  <button
                    onClick={() => setIsDefiningZone((v) => !v)}
                    disabled={!isCameraActive}
                    className={`flex-1 flex justify-center items-center gap-2 px-3 py-2.5 text-[10px] font-bold uppercase tracking-widest border backdrop-blur-md transition-colors disabled:opacity-40 ${isDefiningZone ? 'bg-red-500/20 border-red-500 text-red-400' : 'bg-white/5 border-white/20 text-white/70 hover:bg-white/10'}`}
                  >
                    <Crosshair className="w-3.5 h-3.5" /> {isDefiningZone ? 'DRAW ON FEED...' : 'DEFINE ZONE'}
                  </button>
                  {zone && (
                    <button
                      onClick={() => setZone(null)}
                      className="px-3 py-2.5 text-[10px] font-bold uppercase tracking-widest border border-white/20 bg-white/5 hover:bg-white/10 text-white/70 backdrop-blur-md"
                      title="Clear restricted zone"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>

                <button
                  onClick={() => setIsWatchlistOpen(true)}
                  className="flex justify-center items-center gap-2 px-3 py-2.5 text-[10px] font-bold uppercase tracking-widest border border-white/20 bg-white/5 hover:bg-white/10 text-white/70 backdrop-blur-md"
                >
                  <UserX className="w-3.5 h-3.5" /> WATCHLIST ({watchlist.length})
                </button>
              </div>

              <div className="h-[38vh] lg:hidden pointer-events-none shrink-0" />
            </div>

            {/* Bottom Left: Biometric Scan + Stats */}
            <div className="flex flex-col gap-4 shrink-0 lg:mt-auto order-4 lg:order-none pointer-events-auto">
              <div className="bg-black/40 backdrop-blur-md border border-white/20 p-4 w-full shadow-[0_0_30px_rgba(0,0,0,0.8)] relative overflow-hidden flex flex-col h-48 shrink-0" title="Live face-landmark tracking (used for on-device liveness / watchlist scanning)">
                <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2 shrink-0 flex items-center gap-2">
                  <ScanFace className="w-3 h-3" /> Biometric Scan
                </h3>
                <div className="relative w-full flex-1 border border-white/10 flex items-center justify-center bg-white/5 min-h-0">
                  <canvas ref={faceCanvasRef} width={300} height={300} className={`w-full h-full object-contain transition-opacity duration-500 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`} />
                </div>
              </div>

              <div className="bg-black/40 backdrop-blur-md border border-white/20 p-5 w-full shadow-[0_0_30px_rgba(0,0,0,0.8)]">
                <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <Users className="w-3 h-3" /> Occupancy
                </h3>
                <div className="text-3xl font-light tracking-tighter mb-3 text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.5)]">
                  {peopleCount} <span className="text-xs text-white/50 uppercase">people in frame</span>
                </div>
                {objectsSeen.length === 0 ? (
                  <p className="text-[10px] text-white/40 italic">No entities detected.</p>
                ) : (
                  <ul className="flex flex-wrap gap-1.5">
                    {objectsSeen.map((obj) => (
                      <li key={obj} className="text-[9px] px-2 py-1 bg-white/5 border border-white/10 text-white/70 uppercase tracking-wider">{obj}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>

          {/* Right Column: Incident Log */}
          <div className="contents lg:flex lg:flex-col w-full lg:w-96 pointer-events-none shrink-0 mt-0">
            <div className="flex flex-col gap-4 shrink-0 w-full order-2 lg:order-none pointer-events-auto lg:h-full">
              <div className="w-full bg-black/40 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)] flex flex-col lg:flex-1 lg:min-h-0">
                <div className="flex items-center justify-between mb-3 shrink-0">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest flex items-center gap-2">
                    <Bell className="w-3 h-3" /> Incident Log ({alerts.length})
                  </h3>
                  {alerts.length > 0 && (
                    <button onClick={clearAllAlerts} className="text-[9px] uppercase text-white/40 hover:text-white/80 tracking-widest">Clear all</button>
                  )}
                </div>
                <div className="flex flex-col gap-2 overflow-y-auto lg:flex-1 min-h-0 max-h-[40vh] lg:max-h-none pr-1">
                  {alerts.length === 0 ? (
                    <p className="text-[10px] text-white/40 italic">No incidents recorded yet.</p>
                  ) : (
                    <AnimatePresence>
                      {alerts.map((a) => {
                        const Icon = ALERT_ICON[a.type];
                        return (
                          <motion.div
                            key={a.id}
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 10 }}
                            className={`flex gap-3 p-3 border bg-black/50 ${SEVERITY_COLOR[a.severity]}`}
                          >
                            {a.snapshot && (
                              <img src={a.snapshot} alt="" className="w-14 h-14 object-cover shrink-0 border border-white/20 grayscale" />
                            )}
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-1.5 mb-1">
                                <Icon className="w-3 h-3 shrink-0" />
                                <span className="text-[9px] uppercase tracking-widest font-bold">{a.type.replace('_', ' ')}</span>
                                <span className="text-[9px] text-white/40 ml-auto shrink-0">{new Date(a.timestamp).toLocaleTimeString()}</span>
                              </div>
                              <p className="text-[11px] leading-snug text-white/90">{a.narrative || a.message}</p>
                            </div>
                            <button onClick={() => dismissAlert(a.id)} className="text-white/30 hover:text-white/80 shrink-0">
                              <CheckCircle2 className="w-4 h-4" />
                            </button>
                          </motion.div>
                        );
                      })}
                    </AnimatePresence>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Error Modal */}
      <AnimatePresence>
        {errorMsg && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto">
            <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }} className="bg-zinc-900 border border-red-500/50 p-6 max-w-md w-full shadow-[0_0_40px_rgba(239,68,68,0.2)] relative">
              <div className="flex items-start gap-4 mb-6">
                <div className="p-3 bg-red-500/10 border border-red-500/30 shrink-0">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-red-500 uppercase tracking-widest">{status}</h3>
                  <p className="text-sm mt-2 text-red-400/80 leading-relaxed">{errorMsg}</p>
                </div>
              </div>
              <button onClick={() => setErrorMsg(null)} className="w-full py-3 text-xs font-mono font-bold uppercase tracking-widest transition-colors bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 text-red-400">
                Dismiss
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info Toast */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col justify-end items-center pointer-events-none z-30 w-[calc(100%-2rem)] sm:w-full max-w-md">
        <AnimatePresence>
          {infoMsg && !errorMsg && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }} className="bg-black/80 backdrop-blur-md border border-white/30 p-4 flex items-start gap-3 text-white shadow-[0_0_20px_rgba(255,255,255,0.1)] mb-4 w-full">
              <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5" />
              <div>
                <h3 className="font-bold text-sm">{status}</h3>
                <p className="text-xs mt-1 text-white/80">{infoMsg}</p>
              </div>
              <button onClick={() => setInfoMsg(null)} className="ml-auto text-white/40 hover:text-white/80"><X className="w-4 h-4" /></button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Watchlist Modal */}
      <AnimatePresence>
        {isWatchlistOpen && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto" onClick={() => setIsWatchlistOpen(false)}>
            <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }} onClick={(e) => e.stopPropagation()} className="bg-zinc-900 border border-white/20 p-6 max-w-lg w-full shadow-[0_0_40px_rgba(0,0,0,0.8)] relative max-h-[90vh] overflow-y-auto">
              <div className="flex items-start justify-between gap-4 mb-4">
                <h2 className="text-xl font-bold text-white flex items-center gap-2"><UserX className="w-5 h-5" /> Watchlist</h2>
                <button onClick={() => setIsWatchlistOpen(false)} className="p-2 shrink-0 border border-white/20 bg-black/50 hover:bg-white/10 text-white/50 hover:text-white transition-colors"><X className="w-4 h-4" /></button>
              </div>

              <p className="text-xs text-white/60 leading-relaxed mb-4">
                Add a person you don't want to let in. Upload one clear, front-facing photo — the system computes a face signature on-device and alerts you if that person appears on camera. Stored only on this device.
              </p>

              <div className="border border-white/10 bg-black/40 p-4 mb-4 space-y-3">
                <input
                  type="text"
                  placeholder="Name"
                  value={enrollName}
                  onChange={(e) => setEnrollName(e.target.value)}
                  className="w-full bg-white/5 border border-white/20 px-3 py-2 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-white/50"
                />
                <input
                  type="text"
                  placeholder="Reason for flag (e.g. trespass notice, banned customer)"
                  value={enrollReason}
                  onChange={(e) => setEnrollReason(e.target.value)}
                  className="w-full bg-white/5 border border-white/20 px-3 py-2 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-white/50"
                />
                <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleEnrollPhoto(f); }} />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={enrollBusy}
                  className="w-full flex justify-center items-center gap-2 px-4 py-2.5 text-xs font-bold uppercase tracking-widest border border-white bg-white/10 hover:bg-white/20 text-white disabled:opacity-50"
                >
                  {enrollBusy ? <><Loader2 className="w-4 h-4 animate-spin" /> PROCESSING...</> : <><Upload className="w-4 h-4" /> Upload Photo & Add</>}
                </button>
                {enrollError && <p className="text-[11px] text-red-400">{enrollError}</p>}
              </div>

              {watchlist.length === 0 ? (
                <p className="text-xs text-white/40 italic">No one on the watchlist yet.</p>
              ) : (
                <ul className="space-y-2">
                  {watchlist.map((p) => (
                    <li key={p.id} className="flex items-center gap-3 bg-white/5 border border-white/10 p-2">
                      <img src={p.photo} alt={p.name} className="w-10 h-10 object-cover shrink-0 grayscale" />
                      <div className="min-w-0 flex-1">
                        <div className="text-sm text-white truncate">{p.name}</div>
                        {p.reason && <div className="text-[10px] text-white/50 truncate">{p.reason}</div>}
                      </div>
                      <button onClick={() => removeWatchlistEntry(p.id)} className="text-white/30 hover:text-red-400 shrink-0"><Trash2 className="w-4 h-4" /></button>
                    </li>
                  ))}
                </ul>
              )}

              <p className="text-[10px] text-white/40 mt-4 pt-4 border-t border-white/10 leading-relaxed">
                Face recognition is regulated in many jurisdictions (e.g. BIPA, GDPR). Confirm your organization has the legal basis to use it, keep the match threshold conservative, and always have a person confirm a match before acting on it.
              </p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info Modal */}
      <AnimatePresence>
        {isInfoOpen && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto" onClick={() => setIsInfoOpen(false)}>
            <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }} onClick={(e) => e.stopPropagation()} className="bg-zinc-900 border border-white/20 p-6 max-w-lg w-full shadow-[0_0_40px_rgba(0,0,0,0.8)] relative max-h-[90vh] overflow-y-auto">
              <div className="flex items-start justify-between gap-4 mb-4">
                <h2 className="text-xl font-bold text-white flex items-center gap-2"><Shield className="w-5 h-5" /> About Git Group</h2>
                <button onClick={() => setIsInfoOpen(false)} className="p-2 shrink-0 border border-white/20 bg-black/50 hover:bg-white/10 text-white/50 hover:text-white transition-colors"><X className="w-4 h-4" /></button>
              </div>
              <div className="space-y-4 text-sm text-white/80 leading-relaxed">
                <p><strong>Git Group — Home of Technology</strong> turns a camera feed into a live security console, entirely on-device.</p>
                <p className="text-xs text-white/50">CEO: Frank Ssemakula</p>
                <ul className="list-disc pl-5 space-y-2 text-white/70 text-xs">
                  <li><strong>Zone breach:</strong> draw a restricted area on the feed; get alerted when someone enters it.</li>
                  <li><strong>Loitering:</strong> flags a person who lingers in frame past a time threshold.</li>
                  <li><strong>Unattended objects:</strong> flags a bag/case left with no one nearby.</li>
                  <li><strong>Watchlist:</strong> alerts if a person you've flagged (e.g. a banned individual) appears on camera.</li>
                  <li><strong>Incident log:</strong> every alert is timestamped with a snapshot and an optional AI-written one-line summary.</li>
                </ul>
                <p className="text-xs text-white/50 mt-4 pt-4 border-t border-white/10">
                  Object detection, face tracking and face matching all run locally in your browser. No video is uploaded or stored off-device. When enabled, incident summaries send only short text (alert type + description) to Gemini — never image or video data.
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

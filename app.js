// =============================================================================
// Live Face Swap — app.js
//
// Pipeline overview:
//   1. MediaPipe Face Landmarker gives us 468 normalized (x,y,z) landmarks for
//      any face image (the uploaded photo, once) and for every webcam frame.
//   2. We triangulate the uploaded face's landmarks with a Delaunay
//      triangulation (computed once, on upload). This gives a fixed list of
//      triangle index-triples, e.g. [12, 45, 190], that reference positions
//      in *both* faces (index 12 always means "the same relative facial
//      point" in the source photo and in the live face).
//   3. Every frame, for each triangle we take the 3 source points (from the
//      uploaded photo, fixed) and the 3 destination points (from the live
//      face, this frame) and compute the affine transform that maps one onto
//      the other. Canvas 2D lets us apply that transform, clip to the
//      destination triangle, and drawImage() the source photo — only the
//      pixels inside the clipped triangle actually get painted, so 468
//      points x ~800 triangles reconstructs the whole warped face.
//   4. The reassembled warped face is masked with a blurred (feathered)
//      alpha shape following the live face's outer contour, with the inner
//      mouth opening (teeth/tongue cavity) punched out of that mask — so
//      the source photo's mouth interior is never drawn, and the live
//      webcam's real teeth/tongue always show through instead. The masked
//      result is brightness-matched to the live scene, then composited over
//      the live video frame so the edges blend instead of looking like a
//      hard cutout.
//
// Nothing here touches the network except to load the MediaPipe model/wasm,
// and nothing is written to disk/localStorage/sessionStorage — the uploaded
// photo lives only in a JS variable (an offscreen canvas) for the life of
// the page.
// =============================================================================

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// The standard MediaPipe Face Mesh "face oval" landmark indices, in contour
// order. Used only to build the feathered blend mask (not for tracking).
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
  378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
  162, 21, 54, 103, 67, 109,
];

// The inner lip contour — the boundary of the mouth *opening* (teeth/tongue
// cavity), as opposed to the outer lip contour (the lips themselves). We cut
// this region out of the swap mask so the real, live mouth interior always
// shows through instead of the source photo's (mouth-shape-mismatched)
// teeth/tongue.
const INNER_LIPS = [
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
  82, 81, 80, 191,
];

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const faceUpload = document.getElementById("faceUpload");
const facePreview = document.getElementById("facePreview");
const resSelect = document.getElementById("resSelect");
const statusText = document.getElementById("statusText");
const fpsText = document.getElementById("fpsText");
const outputCanvas = document.getElementById("outputCanvas");
const webcamVideo = document.getElementById("webcamVideo");

const outCtx = outputCanvas.getContext("2d", { willReadFrequently: false });

// ---------------------------------------------------------------------------
// State (all in-memory only)
// ---------------------------------------------------------------------------
let imageLandmarker = null; // MediaPipe FaceLandmarker, IMAGE mode (uploaded photo)
let videoLandmarker = null; // MediaPipe FaceLandmarker, VIDEO mode (live webcam)
let mediapipeReady = false;

let mediaStream = null;
let rafHandle = null;

/** The uploaded face, kept only in memory. Cleared on refresh / new upload. */
const uploadedFace = {
  canvas: null, // offscreen canvas holding the uploaded photo's pixels
  width: 0,
  height: 0,
  landmarks: null, // array of {x,y} in *pixel* coords of uploadedFace.canvas
  triangles: null, // array of [i,j,k] index triples into landmarks
  avgLum: null, // average skin luminance, sampled once, for brightness matching
};

// Cached live-frame brightness gain, recomputed every LUM_SAMPLE_INTERVAL
// frames (see drawFaceSwap) — lighting doesn't change frame to frame, so
// there's no need to pay for a getImageData readback on every single frame.
const LUM_SAMPLE_INTERVAL = 6;
let cachedBrightnessGain = 1;
let frameCounter = 0;

// Offscreen scratch canvases reused every frame (avoid per-frame allocation)
const warpCanvas = document.createElement("canvas");
const warpCtx = warpCanvas.getContext("2d");
const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d");

let fpsSamples = [];
let lastFrameTime = 0;

// =============================================================================
// MediaPipe setup (lazy — only loaded when first needed)
// =============================================================================
async function ensureMediapipeReady() {
  if (mediapipeReady) return;
  statusText.textContent = "Loading face-tracking model...";
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

  // Some browsers/GPUs don't support the WebGL delegate for the wasm model;
  // fall back to CPU rather than leaving the app dead in the water.
  async function createLandmarker(runningMode) {
    try {
      return await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode,
        numFaces: 1,
      });
    } catch (err) {
      console.warn("GPU delegate failed, falling back to CPU:", err);
      return await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
        runningMode,
        numFaces: 1,
      });
    }
  }

  imageLandmarker = await createLandmarker("IMAGE");
  videoLandmarker = await createLandmarker("VIDEO");

  mediapipeReady = true;
  statusText.textContent = "Model loaded. Click \"Start Camera\" to begin.";
}

// =============================================================================
// Delaunay triangulation (Bowyer-Watson), run once per uploaded photo.
//
// We triangulate the 468 landmark points of the uploaded face in its own
// pixel space. The resulting triangle *indices* are reused every frame
// against the live face's landmark positions — the topology stays fixed,
// only the point positions move.
// =============================================================================
function delaunayTriangulate(points) {
  // Bounding super-triangle that contains all points, expanded generously.
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const dx = maxX - minX;
  const dy = maxY - minY;
  const deltaMax = Math.max(dx, dy, 1) * 10;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  const superA = { x: midX - deltaMax, y: midY - deltaMax };
  const superB = { x: midX, y: midY + deltaMax };
  const superC = { x: midX + deltaMax, y: midY - deltaMax };

  // Work with an index-augmented point list; -1/-2/-3 mark super-triangle verts.
  const pts = points.map((p, i) => ({ x: p.x, y: p.y, i }));
  const SA = { x: superA.x, y: superA.y, i: -1 };
  const SB = { x: superB.x, y: superB.y, i: -2 };
  const SC = { x: superC.x, y: superC.y, i: -3 };

  let triangles = [{ a: SA, b: SB, c: SC }];

  function circumcircleContains(tri, p) {
    const { a, b, c } = tri;
    const ax = a.x - p.x, ay = a.y - p.y;
    const bx = b.x - p.x, by = b.y - p.y;
    const cx = c.x - p.x, cy = c.y - p.y;
    const det =
      (ax * ax + ay * ay) * (bx * cy - cx * by) -
      (bx * bx + by * by) * (ax * cy - cx * ay) +
      (cx * cx + cy * cy) * (ax * by - bx * ay);
    // Orientation-independent: use signed area to know which sign means "inside".
    const area =
      (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    return area >= 0 ? det > 0 : det < 0;
  }

  function edgeKey(p, q) {
    return p.i < q.i ? `${p.i}_${q.i}` : `${q.i}_${p.i}`;
  }

  for (const p of pts) {
    const badTriangles = triangles.filter((t) => circumcircleContains(t, p));

    // Find the boundary of the polygonal hole (edges not shared by two bad triangles)
    const edgeCount = new Map();
    for (const t of badTriangles) {
      const edges = [
        [t.a, t.b],
        [t.b, t.c],
        [t.c, t.a],
      ];
      for (const [e0, e1] of edges) {
        const key = edgeKey(e0, e1);
        if (edgeCount.has(key)) {
          edgeCount.get(key).count++;
        } else {
          edgeCount.set(key, { count: 1, e0, e1 });
        }
      }
    }
    const boundary = [];
    for (const { count, e0, e1 } of edgeCount.values()) {
      if (count === 1) boundary.push([e0, e1]);
    }

    triangles = triangles.filter((t) => !badTriangles.includes(t));

    for (const [e0, e1] of boundary) {
      triangles.push({ a: e0, b: e1, c: p });
    }
  }

  // Discard any triangle still touching a super-vertex, then emit index triples.
  const result = [];
  for (const t of triangles) {
    if (t.a.i < 0 || t.b.i < 0 || t.c.i < 0) continue;
    result.push([t.a.i, t.b.i, t.c.i]);
  }
  return result;
}

// =============================================================================
// Affine transform between two triangles.
//
// Solves for the 2x3 matrix M (u = a*x + b*y + c, v = d*x + e*y + f) that
// maps the 3 source points onto the 3 destination points, then returns it in
// the parameter order expected by CanvasRenderingContext2D#transform().
// =============================================================================
function triangleAffine(src, dst) {
  const [s0, s1, s2] = src;
  const [d0, d1, d2] = dst;

  // Solve the 3x3 linear system via the closed-form barycentric-basis inverse.
  const denom =
    s0.x * (s1.y - s2.y) - s1.x * (s0.y - s2.y) + s2.x * (s0.y - s1.y);
  if (Math.abs(denom) < 1e-8) return null; // degenerate triangle, skip

  const a =
    (d0.x * (s1.y - s2.y) - d1.x * (s0.y - s2.y) + d2.x * (s0.y - s1.y)) /
    denom;
  const b =
    (s0.x * (d1.x - d2.x) - s1.x * (d0.x - d2.x) + s2.x * (d0.x - d1.x)) /
    denom;
  const c =
    (s0.x * (s1.y * d2.x - s2.y * d1.x) -
      s1.x * (s0.y * d2.x - s2.y * d0.x) +
      s2.x * (s0.y * d1.x - s1.y * d0.x)) /
    denom;

  const d =
    (d0.y * (s1.y - s2.y) - d1.y * (s0.y - s2.y) + d2.y * (s0.y - s1.y)) /
    denom;
  const e =
    (s0.x * (d1.y - d2.y) - s1.x * (d0.y - d2.y) + s2.x * (d0.y - d1.y)) /
    denom;
  const f =
    (s0.x * (s1.y * d2.y - s2.y * d1.y) -
      s1.x * (s0.y * d2.y - s2.y * d0.y) +
      s2.x * (s0.y * d1.y - s1.y * d0.y)) /
    denom;

  return [a, d, b, e, c, f]; // ctx.transform(m11, m12, m21, m22, dx, dy)
}

/** Warp `srcCanvas` triangle-by-triangle onto `ctx` using src/dst point sets. */
function warpTriangles(ctx, srcCanvas, srcPoints, dstPoints, triangles) {
  for (const [i, j, k] of triangles) {
    const s0 = srcPoints[i], s1 = srcPoints[j], s2 = srcPoints[k];
    const d0 = dstPoints[i], d1 = dstPoints[j], d2 = dstPoints[k];
    if (!d0 || !d1 || !d2) continue;

    const m = triangleAffine([s0, s1, s2], [d0, d1, d2]);
    if (!m) continue;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(d0.x, d0.y);
    ctx.lineTo(d1.x, d1.y);
    ctx.lineTo(d2.x, d2.y);
    ctx.closePath();
    ctx.clip();

    ctx.transform(m[0], m[1], m[2], m[3], m[4], m[5]);
    ctx.drawImage(srcCanvas, 0, 0);
    ctx.restore();
  }
}

// =============================================================================
// Upload handling
// =============================================================================
faceUpload.addEventListener("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  try {
    await ensureMediapipeReady();
    statusText.textContent = "Analyzing uploaded face...";

    const img = await loadImageFromFile(file); // FileReader -> Image, in memory only

    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);

    const result = imageLandmarker.detect(canvas);
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      statusText.textContent = "No face detected in that photo — try another.";
      return;
    }

    const rawLandmarks = result.faceLandmarks[0];
    const pixelLandmarks = rawLandmarks.map((p) => ({
      x: p.x * canvas.width,
      y: p.y * canvas.height,
    }));

    uploadedFace.canvas = canvas;
    uploadedFace.width = canvas.width;
    uploadedFace.height = canvas.height;
    uploadedFace.landmarks = pixelLandmarks;
    uploadedFace.triangles = delaunayTriangulate(pixelLandmarks);

    const avgColor = averageColorInPolygon(canvas, pixelLandmarks, FACE_OVAL);
    uploadedFace.avgLum = avgColor ? luminance(avgColor) : null;
    cachedBrightnessGain = 1; // reset until the next live-frame sample

    facePreview.src = canvas.toDataURL("image/png");
    facePreview.hidden = false;

    statusText.textContent = mediaStream
      ? "Face loaded — swapping live."
      : 'Face loaded. Click "Start Camera" to begin.';
  } catch (err) {
    console.error(err);
    statusText.textContent = "Could not process that image.";
  } finally {
    // Allow re-selecting the same file later; the File object itself is
    // never stored anywhere beyond this handler.
    faceUpload.value = "";
  }
});

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = reader.result; // data URL held only in memory, never persisted
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// =============================================================================
// Camera controls
// =============================================================================
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

async function startCamera() {
  startBtn.disabled = true;
  try {
    await ensureMediapipeReady();

    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    webcamVideo.srcObject = mediaStream;
    await webcamVideo.play();

    applyResolution();
    stopBtn.disabled = false;
    statusText.textContent = uploadedFace.landmarks
      ? "Swapping live."
      : "Camera running. Upload a face photo to start swapping.";

    lastFrameTime = performance.now();
    fpsSamples = [];
    rafHandle = requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error(err);
    statusText.textContent =
      "Could not access the camera (permission denied or unavailable).";
    startBtn.disabled = false;
  }
}

function stopCamera() {
  if (rafHandle) cancelAnimationFrame(rafHandle);
  rafHandle = null;

  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  webcamVideo.srcObject = null;

  outCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  fpsText.textContent = "";
  statusText.textContent = 'Camera stopped. Click "Start Camera" to resume.';
}

resSelect.addEventListener("change", applyResolution);

function applyResolution() {
  const [w, h] = resSelect.value.split("x").map(Number);
  outputCanvas.width = w;
  outputCanvas.height = h;
  warpCanvas.width = w;
  warpCanvas.height = h;
  maskCanvas.width = w;
  maskCanvas.height = h;
}

applyResolution(); // size the canvas correctly even before the camera starts

// =============================================================================
// Main render loop
// =============================================================================
function renderLoop() {
  if (!mediaStream) return; // stopped

  const now = performance.now();

  if (webcamVideo.readyState >= 2) {
    const w = outputCanvas.width;
    const h = outputCanvas.height;

    // 1. Draw the current webcam frame as the base layer.
    outCtx.save();
    outCtx.scale(-1, 1); // mirror for a natural "selfie" view
    outCtx.drawImage(webcamVideo, -w, 0, w, h);
    outCtx.restore();

    // 2. Track the live face and, if we have an uploaded face ready, warp
    //    it into position and composite it on top.
    const detection = videoLandmarker.detectForVideo(webcamVideo, now);
    const liveLandmarksRaw =
      detection.faceLandmarks && detection.faceLandmarks[0];

    if (liveLandmarksRaw && uploadedFace.landmarks) {
      // Live landmarks come from the unmirrored video; flip x to match the
      // mirrored frame we just drew, and scale into canvas pixel space.
      const liveLandmarks = liveLandmarksRaw.map((p) => ({
        x: (1 - p.x) * w,
        y: p.y * h,
      }));

      drawFaceSwap(liveLandmarks, w, h);
    }
  }

  // FPS tracking
  const dt = now - lastFrameTime;
  lastFrameTime = now;
  if (dt > 0) {
    fpsSamples.push(1000 / dt);
    if (fpsSamples.length > 20) fpsSamples.shift();
    const avg = fpsSamples.reduce((a, b) => a + b, 0) / fpsSamples.length;
    fpsText.textContent = `${avg.toFixed(0)} fps`;
  }

  rafHandle = requestAnimationFrame(renderLoop);
}

/**
 * Trace a closed polygon path through `landmarks` at the given `indices`.
 * `scale`, if given, inflates the polygon outward from its own centroid by
 * that factor first. This matters for a subsequent blurred fill: blurring a
 * shape softens its edges by *shrinking* its effective peak-alpha interior
 * (a gaussian blur can't add energy, only spread it), so a thin polygon
 * blurred at hard edges never reaches full opacity anywhere. Inflating the
 * shape before blurring keeps the blur's softening confined to the outside
 * of the original boundary, so the true interior still hits full opacity —
 * this is what guarantees the inner-mouth cutout actually reaches "fully
 * transparent" at its center instead of leaving a faint ghost of the
 * source face's mouth visible.
 */
function tracePolygon(ctx, landmarks, indices, scale) {
  const pts = indices.map((idx) => landmarks[idx]).filter(Boolean);
  let points = pts;
  if (scale && scale !== 1 && pts.length > 0) {
    let cx = 0, cy = 0;
    for (const p of pts) { cx += p.x; cy += p.y; }
    cx /= pts.length; cy /= pts.length;
    points = pts.map((p) => ({
      x: cx + (p.x - cx) * scale,
      y: cy + (p.y - cy) * scale,
    }));
  }

  ctx.beginPath();
  points.forEach((p, n) => {
    if (n === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.closePath();
}

/**
 * Average RGB color within a landmark polygon (e.g. the face oval), sampled
 * on a strided grid and restricted to points actually inside the polygon
 * (not just its bounding box) so background pixels don't skew the result.
 * Used once, at upload time, on the source photo — cheap enough to afford
 * exact polygon testing since it never runs per-frame.
 */
function averageColorInPolygon(canvas, landmarks, indices) {
  const pts = indices.map((idx) => landmarks[idx]).filter(Boolean);
  if (pts.length === 0) return null;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  minX = Math.max(0, Math.floor(minX));
  minY = Math.max(0, Math.floor(minY));
  maxX = Math.min(canvas.width, Math.ceil(maxX));
  maxY = Math.min(canvas.height, Math.ceil(maxY));
  const bw = maxX - minX, bh = maxY - minY;
  if (bw <= 0 || bh <= 0) return null;

  const ctx = canvas.getContext("2d");
  const data = ctx.getImageData(minX, minY, bw, bh).data;

  const path = new Path2D();
  pts.forEach((p, n) => {
    const x = p.x - minX, y = p.y - minY;
    if (n === 0) path.moveTo(x, y);
    else path.lineTo(x, y);
  });
  path.closePath();

  let r = 0, g = 0, b = 0, count = 0;
  const stride = 3;
  for (let y = 0; y < bh; y += stride) {
    for (let x = 0; x < bw; x += stride) {
      if (!ctx.isPointInPath(path, x, y)) continue;
      const i = (y * bw + x) * 4;
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      count++;
    }
  }
  if (count === 0) return null;
  return { r: r / count, g: g / count, b: b / count };
}

/**
 * Cheap approximate average color of the live face's bounding box, read
 * straight from the already-drawn video frame. Skips exact polygon testing
 * (unlike averageColorInPolygon) since this runs on a hot path — a rough
 * average including a bit of background/hair is precise enough for
 * brightness matching.
 */
function averageColorInBoundingBox(ctx, landmarks, indices, w, h) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const idx of indices) {
    const p = landmarks[idx];
    if (!p) continue;
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  minX = Math.max(0, Math.floor(minX));
  minY = Math.max(0, Math.floor(minY));
  maxX = Math.min(w, Math.ceil(maxX));
  maxY = Math.min(h, Math.ceil(maxY));
  const bw = maxX - minX, bh = maxY - minY;
  if (bw <= 0 || bh <= 0) return null;

  const data = ctx.getImageData(minX, minY, bw, bh).data;
  let r = 0, g = 0, b = 0, count = 0;
  const stride = 4;
  for (let i = 0; i < data.length; i += 4 * stride) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
    count++;
  }
  if (count === 0) return null;
  return { r: r / count, g: g / count, b: b / count };
}

function luminance(color) {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

/**
 * Warps the uploaded face onto the live landmark positions and blends it
 * over the already-drawn video frame in outputCanvas.
 */
function drawFaceSwap(liveLandmarks, w, h) {
  // --- Rebuild the warped face on the scratch canvas ---
  warpCtx.clearRect(0, 0, w, h);
  warpTriangles(
    warpCtx,
    uploadedFace.canvas,
    uploadedFace.landmarks,
    liveLandmarks,
    uploadedFace.triangles
  );

  // --- Build the swap mask: feathered face oval, MINUS the inner mouth ---
  //
  // The inner-lip polygon marks the mouth *opening* (teeth/tongue cavity).
  // We deliberately do not warp the source face's mouth interior onto the
  // live feed — a photo's static teeth/tongue look wrong the instant the
  // live mouth opens differently. Instead we punch a feathered hole in the
  // mask there so the real, live mouth interior always shows through, while
  // the swapped face's own lips still cover the lip skin right up to that
  // hole (so the lip line itself looks continuous, not double-outlined).
  maskCtx.clearRect(0, 0, w, h);

  maskCtx.filter = "blur(6px)";
  maskCtx.fillStyle = "#fff";
  tracePolygon(maskCtx, liveLandmarks, FACE_OVAL);
  maskCtx.fill();

  maskCtx.globalCompositeOperation = "destination-out";
  maskCtx.filter = "blur(2.5px)";
  maskCtx.fillStyle = "#fff";
  tracePolygon(maskCtx, liveLandmarks, INNER_LIPS, 1.7);
  maskCtx.fill();
  maskCtx.globalCompositeOperation = "source-over";
  maskCtx.filter = "none";

  // Apply the mask to the warped face (keep only pixels inside the feathered shape)
  warpCtx.save();
  warpCtx.globalCompositeOperation = "destination-in";
  warpCtx.drawImage(maskCanvas, 0, 0);
  warpCtx.restore();

  // --- Match the warped face's brightness to the live scene's lighting ---
  //
  // A source photo shot in different lighting than the current webcam frame
  // otherwise looks like a visibly pasted-on patch. We only correct overall
  // brightness (not full per-channel color transfer) — it's the dominant
  // mismatch and is cheap enough to recompute live.
  frameCounter++;
  if (frameCounter % LUM_SAMPLE_INTERVAL === 0 && uploadedFace.avgLum) {
    const liveColor = averageColorInBoundingBox(outCtx, liveLandmarks, FACE_OVAL, w, h);
    if (liveColor) {
      const ratio = luminance(liveColor) / uploadedFace.avgLum;
      cachedBrightnessGain = Math.min(1.6, Math.max(0.6, ratio));
    }
  }

  // Composite the masked, warped face over the live video frame.
  outCtx.save();
  outCtx.filter = `brightness(${cachedBrightnessGain})`;
  outCtx.drawImage(warpCanvas, 0, 0);
  outCtx.restore();
}

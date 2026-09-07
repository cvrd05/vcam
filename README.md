# Live Face Swap

A single-page web app that live face-swaps an uploaded photo onto your webcam
feed, entirely in the browser. No backend, no build step, no accounts.

## How it works

1. **MediaPipe Face Landmarker** (loaded from a CDN at runtime) extracts 468
   facial landmarks from the uploaded photo (once) and from every webcam
   frame.
2. The uploaded photo's landmarks are triangulated with a **Delaunay
   triangulation**, computed once on upload. This produces a fixed list of
   triangles (as landmark-index triples) shared by both faces.
3. Every frame, each triangle is warped from the uploaded photo onto the
   live face's current landmark positions using a **per-triangle affine
   transform**, applied via Canvas 2D `clip()` + `transform()` +
   `drawImage()`.
4. The reassembled warped face is masked with a **blurred (feathered) alpha
   shape** following the live face's contour, then composited over the
   video frame so the edges blend instead of looking like a cutout.

The face-warping logic lives in `app.js` and is commented in detail —
see the `delaunayTriangulate`, `triangleAffine`, `warpTriangles`, and
`drawFaceSwap` functions.

## Privacy

- The uploaded photo is read with `FileReader` straight into a JS
  `Image`/canvas object — it is never written to disk, `localStorage`,
  `sessionStorage`, or any server. Refresh the page and it's gone.
- The webcam feed never leaves your machine.
- The only network requests are to load the MediaPipe library/model files
  from their CDN (`cdn.jsdelivr.net`, `storage.googleapis.com`) the first
  time you use the page.

## Running it locally

Because the page loads the MediaPipe library as an ES module, most browsers
require it to be served over `http://` rather than opened directly as a
`file://` URL. From this directory, run any simple static server, for
example:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000> in a modern desktop browser (Chrome or
Edge recommended; camera access requires HTTPS or `localhost`).

## Using it

1. Click **Upload Face** and pick a clear, front-facing photo of a face.
2. Click **Start Camera** and grant webcam permission.
3. Your face gets replaced by the uploaded photo's face in real time.
4. Use the **Quality** dropdown to trade off sharpness vs. frame rate — pick
   "Low" on an older laptop if it feels laggy.
5. Click **Stop Camera** to release the webcam.

## Notes on performance

- Processing runs at an internal resolution (selectable via the Quality
  dropdown: 320×240 / 480×360 / 640×480) that's then displayed scaled up to
  fill the on-screen canvas via CSS — this keeps the per-frame triangle-warp
  cost bounded even on lower-end hardware.
- The face tracker will use a GPU-accelerated delegate when the browser
  supports it, and automatically falls back to CPU otherwise.

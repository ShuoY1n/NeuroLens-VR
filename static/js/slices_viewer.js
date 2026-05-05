import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const MANIFEST_URL = "/outputs/manifest.json";
const OBLIQUE_PLANE_NAME = "oblique";
const OBLIQUE_ENDPOINT = "/api/slice/oblique";
const OBLIQUE_PNG_SIZE = 192;

let manifest = null;
let plane = "axial";
let idxI = 0;
let idxJ = 0;
let idxK = 0;

/**
 * Oblique mode state. The basis (right, up, n) and halfExtent are computed
 * from yaw/pitch and shared between the 3D clip plane, the slice sheet
 * placement, and the API request URL so all three views stay in sync.
 */
const oblique = {
  yawDeg: 0,
  pitchDeg: 0,
  offsetMm: 0,
  offsetMaxMm: 100,
  halfExtentMm: 100,
  n: new THREE.Vector3(0, 0, 1),
  right: new THREE.Vector3(1, 0, 0),
  up: new THREE.Vector3(0, 1, 0),
  requestSeq: 0,
  pendingRequest: null,
  activeRequest: null,
};

let scene = null;
let camera = null;
let renderer = null;
let labelRenderer = null;
let controls = null;
let raf = 0;
/** Holds brain + slice sheet + box; rotation maps NIfTI-style axes to Y-up. */
let volumeRoot = null;
let brainRoot = null;
const clipPlane = new THREE.Plane();
const clipPlaneLocal = new THREE.Plane();
let sliceSheet = null;
const texLoader = new THREE.TextureLoader();

const el = {
  status: document.getElementById("status"),
  plane: document.getElementById("plane"),
  index: document.getElementById("index"),
  indexLabel: document.getElementById("indexLabel"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  slice: document.getElementById("slice"),
  threeHost: document.getElementById("three-host"),
  cardinalIndexRow: document.getElementById("cardinalIndexRow"),
  obliqueControls: document.getElementById("obliqueControls"),
  yaw: document.getElementById("yaw"),
  yawLabel: document.getElementById("yawLabel"),
  pitch: document.getElementById("pitch"),
  pitchLabel: document.getElementById("pitchLabel"),
  obliqueOffset: document.getElementById("obliqueOffset"),
  obliqueOffsetLabel: document.getElementById("obliqueOffsetLabel"),
  resetOblique: document.getElementById("resetOblique"),
};

function setStatus(text, isError = false) {
  el.status.textContent = text;
  el.status.classList.toggle("err", isError);
}

function sliceUrl(pattern, index, pad) {
  const s = String(index).padStart(pad, "0");
  return pattern.replace("{index}", s);
}

function volDims() {
  const d = manifest.volume.dimensions;
  return { nx: d[0], ny: d[1], nz: d[2] };
}

function spacing() {
  const s = manifest.volume.voxelSpacingMm;
  return { sx: s[0], sy: s[1], sz: s[2] };
}

function volumeCentersMm() {
  const { nx, ny, nz } = volDims();
  const { sx, sy, sz } = spacing();
  return {
    cx: ((nx - 1) * sx) / 2,
    cy: ((ny - 1) * sy) / 2,
    cz: ((nz - 1) * sz) / 2,
  };
}

function volumeDiagonalMm() {
  const { nx, ny, nz } = volDims();
  const { sx, sy, sz } = spacing();
  return Math.sqrt((nx * sx) ** 2 + (ny * sy) ** 2 + (nz * sz) ** 2);
}

/**
 * Volume-local oblique normal for (yaw, pitch) in degrees.
 * Pitch tilts the default axial normal (+k) toward -j; yaw rotates the
 * tilted normal around +k. Mirrors backend/oblique_sampling.py orientation.
 */
function obliqueNormalFromAngles(yawDeg, pitchDeg) {
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const sinPitch = Math.sin(pitch);
  const cosPitch = Math.cos(pitch);
  const sinYaw = Math.sin(yaw);
  const cosYaw = Math.cos(yaw);
  return new THREE.Vector3(
    sinPitch * sinYaw,
    -sinPitch * cosYaw,
    cosPitch
  ).normalize();
}

/**
 * Same in-plane basis as backend/oblique_sampling.py: first non-degenerate projection
 * of e_x, e_y, e_z onto the plane ⊥ n (e_x first fixes pure axial / pitch 0).
 */
function obliqueBasis(normal) {
  const n = normal.clone().normalize();
  const candidates = [
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, 0, 1),
  ];
  let right = null;
  for (const axis of candidates) {
    const t = axis.clone().sub(n.clone().multiplyScalar(axis.dot(n)));
    const len = t.length();
    if (len >= 1e-10) {
      right = t.divideScalar(len);
      break;
    }
  }
  if (!right) {
    right = new THREE.Vector3(1, 0, 0);
  }
  const up = new THREE.Vector3().crossVectors(n, right).normalize();
  return { right, up, n };
}

function recomputeObliqueBasis() {
  const n = obliqueNormalFromAngles(oblique.yawDeg, oblique.pitchDeg);
  const { right, up } = obliqueBasis(n);
  oblique.n.copy(n);
  oblique.right.copy(right);
  oblique.up.copy(up);
  oblique.halfExtentMm = volumeDiagonalMm() / 2;

  const { cx, cy, cz } = volumeCentersMm();
  const offsetMax =
    Math.abs(cx * n.x) + Math.abs(cy * n.y) + Math.abs(cz * n.z);
  oblique.offsetMaxMm = Math.max(offsetMax, 1);

  el.obliqueOffset.min = (-oblique.offsetMaxMm).toFixed(1);
  el.obliqueOffset.max = oblique.offsetMaxMm.toFixed(1);
  const clampedOffset = Math.min(
    Math.max(oblique.offsetMm, -oblique.offsetMaxMm),
    oblique.offsetMaxMm
  );
  oblique.offsetMm = clampedOffset;
  el.obliqueOffset.value = clampedOffset.toFixed(1);
  el.obliqueOffsetLabel.textContent = clampedOffset.toFixed(1);
}

function isObliqueActive() {
  return plane === OBLIQUE_PLANE_NAME;
}

function currentPlaneSpec() {
  return manifest.slices[plane];
}

function readActiveIndexFromState() {
  if (plane === "axial") return idxK;
  if (plane === "coronal") return idxJ;
  return idxI;
}

function writeActiveIndexToState(v) {
  if (plane === "axial") idxK = v;
  else if (plane === "coronal") idxJ = v;
  else idxI = v;
}

function syncSliderFromState() {
  const spec = currentPlaneSpec();
  el.index.min = "0";
  el.index.max = String(spec.count - 1);
  el.index.value = String(readActiveIndexFromState());
  el.indexLabel.textContent = el.index.value;
}

function refresh2D() {
  if (!manifest) return;
  el.slice.hidden = false;
  if (isObliqueActive()) {
    el.slice.src = obliqueSliceUrl();
    return;
  }
  const spec = currentPlaneSpec();
  const idx = readActiveIndexFromState();
  el.slice.src = sliceUrl(spec.urlPattern, idx, spec.indexPad);
}

function obliqueSliceUrl() {
  const params = new URLSearchParams({
    nx: oblique.n.x.toFixed(6),
    ny: oblique.n.y.toFixed(6),
    nz: oblique.n.z.toFixed(6),
    offset: oblique.offsetMm.toFixed(3),
    size: String(OBLIQUE_PNG_SIZE),
  });
  return `${OBLIQUE_ENDPOINT}?${params.toString()}`;
}

/**
 * Three.js keeps geometry where (n·x + plane.constant) >= 0.
 * Normals chosen so the side opposite the slice-sheet front stays visible.
 * Cardinal modes match the prior behaviour; oblique mirrors that convention
 * by clipping along -n (so the brain "behind" the cut surface is hidden).
 */
function setClipPlaneForActiveAxis() {
  const p = new THREE.Vector3();
  const n = new THREE.Vector3();

  if (isObliqueActive()) {
    n.copy(oblique.n).multiplyScalar(-1);
    p.copy(oblique.n).multiplyScalar(oblique.offsetMm);
  } else {
    const { sx, sy, sz } = spacing();
    const { cx, cy, cz } = volumeCentersMm();
    if (plane === "axial") {
      const zCut = idxK * sz - cz;
      n.set(0, 0, -1);
      p.set(0, 0, zCut);
    } else if (plane === "coronal") {
      const yCut = idxJ * sy - cy;
      n.set(0, -1, 0);
      p.set(0, yCut, 0);
    } else {
      const xCut = idxI * sx - cx;
      n.set(-1, 0, 0);
      p.set(xCut, 0, 0);
    }
  }
  clipPlaneLocal.setFromNormalAndCoplanarPoint(n, p);
  if (volumeRoot) {
    volumeRoot.updateMatrixWorld(true);
    clipPlane.copy(clipPlaneLocal).applyMatrix4(volumeRoot.matrixWorld);
  } else {
    clipPlane.copy(clipPlaneLocal);
  }
}

function applyClippingToBrain() {
  if (!brainRoot || brainRoot.userData.neuroLensClipReady) return;
  brainRoot.traverse((obj) => {
    if (!obj.isMesh) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    mats.forEach((m, i) => {
      if (!m) return;
      const mat = m.clone();
      mat.clippingPlanes = [clipPlane];
      mat.clipIntersection = false;
      mat.side = THREE.DoubleSide;
      if (Array.isArray(obj.material)) obj.material[i] = mat;
      else obj.material = mat;
    });
  });
  brainRoot.userData.neuroLensClipReady = true;
}

function layoutSliceSheet() {
  if (!sliceSheet) return;
  const { nx, ny, nz } = volDims();
  const { sx, sy, sz } = spacing();
  const { cx, cy, cz } = volumeCentersMm();
  /* Nudge sheet past clip plane so grazing views do not z-fight; a bit more than before. */
  const eps = Math.min(sx, sy, sz) * 0.14;

  sliceSheet.matrixAutoUpdate = true;
  sliceSheet.rotation.set(0, 0, 0);
  sliceSheet.quaternion.set(0, 0, 0, 1);
  sliceSheet.scale.set(1, 1, 1);

  if (isObliqueActive()) {
    layoutSliceSheetOblique(eps);
    return;
  }

  if (plane === "axial") {
    const zCut = idxK * sz - cz;
    sliceSheet.scale.set((nx - 1) * sx, (ny - 1) * sy, 1);
    sliceSheet.position.set(0, 0, zCut - eps);
  } else if (plane === "coronal") {
    const yCut = idxJ * sy - cy;
    sliceSheet.scale.set((nx - 1) * sx, (nz - 1) * sz, 1);
    sliceSheet.rotation.x = Math.PI / 2;
    sliceSheet.position.set(0, yCut - eps, 0);
  } else {
    const xCut = idxI * sx - cx;
    /* First scale axis = Z extent (k); negative scale mirrors across Z through quad center (z=0). */
    sliceSheet.scale.set(-(nz - 1) * sz, (ny - 1) * sy, 1);
    sliceSheet.rotation.y = -Math.PI / 2;
    sliceSheet.position.set(xCut - eps, 0, 0);
  }
}

/**
 * Position the slice sheet so its local +X = `right`, +Y = `up`, +Z = `n`.
 * Scaling matches the PNG's covered area (2 * halfExtentMm per side) and
 * the sheet is nudged epsilon along -n to avoid z-fighting with the clip.
 */
function layoutSliceSheetOblique(epsMm) {
  const r = oblique.right;
  const u = oblique.up;
  const n = oblique.n;
  const basis = new THREE.Matrix4().makeBasis(r, u, n);
  const q = new THREE.Quaternion().setFromRotationMatrix(basis);

  const size = 2 * oblique.halfExtentMm;
  sliceSheet.quaternion.copy(q);
  sliceSheet.scale.set(size, size, 1);
  sliceSheet.position
    .copy(n)
    .multiplyScalar(oblique.offsetMm - epsMm);
}

function loadSliceTexture(url) {
  return new Promise((resolve, reject) => {
    texLoader.load(
      url,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        resolve(tex);
      },
      undefined,
      reject
    );
  });
}

/**
 * Slice PNGs use 0 for background (see pipeline windowing). Fade those out so the quad
 * does not paint a black rectangle—only tissue shows, matching the cut brain.
 */
function buildSliceSheetMaterial(tex) {
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    side: THREE.DoubleSide,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -3,
    polygonOffsetUnits: -3,
    transparent: true,
    toneMapped: false,
  });
  mat.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <opaque_fragment>",
      `#include <opaque_fragment>
          {
            float _lum = max(gl_FragColor.r, max(gl_FragColor.g, gl_FragColor.b));
            float _a = smoothstep(0.028, 0.058, _lum);
            gl_FragColor.a *= _a;
            if (gl_FragColor.a < 0.035) discard;
          }`
    );
  };
  return mat;
}

async function updateSliceSheetTexture() {
  if (!sliceSheet || !manifest) return;
  const url = isObliqueActive()
    ? obliqueSliceUrl()
    : (() => {
        const spec = currentPlaneSpec();
        return sliceUrl(spec.urlPattern, readActiveIndexFromState(), spec.indexPad);
      })();
  const seq = ++oblique.requestSeq;
  try {
    const tex = await loadSliceTexture(url);
    // Discard if the user advanced past this request (oblique slider drag).
    if (seq !== oblique.requestSeq) {
      tex.dispose?.();
      return;
    }
    if (isObliqueActive()) {
      configureSliceTextureOblique(tex);
    } else {
      configureSliceTexture(tex, plane);
    }
    const old = sliceSheet.material.map;
    if (old) old.dispose();
    sliceSheet.material.dispose();
    sliceSheet.material = buildSliceSheetMaterial(tex);
  } catch {
    setStatus("Failed to load slice PNG.", true);
  }
}

/**
 * Oblique PNGs are emitted with row 0 = +up, col 0 = -right. Pair that
 * convention with `flipY = false` so plane local +X maps to image right
 * and plane local +Y maps to image top — see backend/oblique_sampling.py.
 */
function configureSliceTextureOblique(tex) {
  tex.center.set(0.5, 0.5);
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.flipY = false;
  tex.repeat.set(1, 1);
  tex.offset.set(0, 0);
  tex.rotation = 0;
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  tex.updateMatrix();
}

async function refreshCut() {
  if (!scene) return;
  setClipPlaneForActiveAxis();
  layoutSliceSheet();
  await updateSliceSheetTexture();
}

function tick() {
  raf = requestAnimationFrame(tick);
  controls.update();
  renderer.render(scene, camera);
  if (labelRenderer) labelRenderer.render(scene, camera);
}

function onResize() {
  if (!renderer || !camera) return;
  const host = el.threeHost;
  const w = host.clientWidth;
  const h = host.clientHeight;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(w, Math.max(h, 1));
  if (labelRenderer) labelRenderer.setSize(w, Math.max(h, 1));
}

function addVolumeAxisLabels() {
  const { nx, ny, nz } = volDims();
  const { sx, sy, sz } = spacing();
  const hx = ((nx - 1) * sx) / 2;
  const hy = ((ny - 1) * sy) / 2;
  const hz = ((nz - 1) * sz) / 2;
  const margin = Math.max(Math.min(sx, sy, sz) * 6, 8);

  const axes = [
    {
      text: "X · i",
      title: "1st voxel index (sagittal plane)",
      color: "#ff6b6b",
      pos: new THREE.Vector3(hx + margin, 0, 0),
    },
    {
      text: "Y · j",
      title: "2nd voxel index (coronal plane)",
      color: "#69db7c",
      pos: new THREE.Vector3(0, hy + margin, 0),
    },
    {
      text: "Z · k",
      title: "3rd voxel index (axial plane)",
      color: "#74c0fc",
      pos: new THREE.Vector3(0, 0, hz + margin),
    },
  ];

  for (const a of axes) {
    const div = document.createElement("div");
    div.className = "axis-label";
    div.textContent = a.text;
    div.title = a.title;
    div.style.color = a.color;
    const label = new CSS2DObject(div);
    label.position.copy(a.pos);
    volumeRoot.add(label);
  }
}

/**
 * Marching-cubes verts use the same IJK→mm mapping as the slice PNGs:
 * axis a runs from 0 toward (dims[a]-1)*spacing[a]. Clip planes use i*sx-cx, etc.
 * Do NOT center by mesh bbox — that shifts the surface away from the cut planes.
 */
function alignBrainToVolumeGrid(root) {
  const { nx, ny, nz } = volDims();
  const { sx, sy, sz } = spacing();
  root.position.set(
    -((nx - 1) * sx) / 2,
    -((ny - 1) * sy) / 2,
    -((nz - 1) * sz) / 2
  );
}

/**
 * PNG row/col vs quad UV: axial raw I×J; coronal/sagittal np.rot90 in pipeline (slice_export).
 * Sagittal: 90° in the slice (Y–Z) plane only. (UV repeat/flip reflection made the cut vanish with clamp + alpha.)
 */
function configureSliceTexture(tex, planeName) {
  tex.center.set(0.5, 0.5);
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.flipY = true;
  tex.repeat.set(1, 1);
  tex.offset.set(0, 0);
  /* Grazing-angle “sheet” artifact: mips average in transparent border; keep LOD0 only. */
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  if (planeName === "axial") {
    tex.rotation = Math.PI / 2;
  } else if (planeName === "sagittal") {
    tex.rotation = Math.PI / 2;
  } else {
    tex.rotation = 0;
  }
  tex.updateMatrix();
}

function initThree() {
  const host = el.threeHost;
  const w = host.clientWidth || 640;
  const h = host.clientHeight || 480;

  scene = new THREE.Scene();
  scene.background = null;

  // GLB uses PBR materials; they stay black without lights (unlike MeshBasicMaterial).
  const hemi = new THREE.HemisphereLight(0xc8d4e8, 0x1a1a22, 0.9);
  const sun = new THREE.DirectionalLight(0xffffff, 1.15);
  sun.position.set(1.2, 2.0, 1.5);
  const fill = new THREE.DirectionalLight(0xaaccff, 0.35);
  fill.position.set(-1.5, -0.5, -1);
  scene.add(hemi, sun, fill);

  camera = new THREE.PerspectiveCamera(50, w / Math.max(h, 1), 0.1, 100000);
  const { nx, ny, nz } = volDims();
  const { sx, sy, sz } = spacing();
  const diag = Math.sqrt((nx * sx) ** 2 + (ny * sy) ** 2 + (nz * sz) ** 2);
  camera.position.set(diag * 0.55, diag * 0.4, diag * 0.65);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, Math.max(h, 1));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.localClippingEnabled = true;
  host.appendChild(renderer.domElement);

  labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(w, Math.max(h, 1));
  labelRenderer.domElement.style.position = "absolute";
  labelRenderer.domElement.style.left = "0";
  labelRenderer.domElement.style.top = "0";
  labelRenderer.domElement.style.pointerEvents = "none";
  host.appendChild(labelRenderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.update();

  volumeRoot = new THREE.Group();
  // Voxel Z (slice axis) -> world +Y so the brain is upright; tweak euler if your data differs.
  volumeRoot.rotation.set(-Math.PI / 2, 0, Math.PI, "XYZ");
  scene.add(volumeRoot);

  const boxGeom = new THREE.BoxGeometry(
    (nx - 1) * sx,
    (ny - 1) * sy,
    (nz - 1) * sz
  );
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(boxGeom),
    new THREE.LineBasicMaterial({ color: 0x555555 })
  );
  volumeRoot.add(edges);
  addVolumeAxisLabels();

  const sheetGeom = new THREE.PlaneGeometry(1, 1);
  sliceSheet = new THREE.Mesh(
    sheetGeom,
    new THREE.MeshBasicMaterial({ visible: true, color: 0x888888, side: THREE.DoubleSide })
  );
  sliceSheet.renderOrder = 10;
  volumeRoot.add(sliceSheet);

  window.addEventListener("resize", onResize);
  cancelAnimationFrame(raf);
  tick();
}

function loadBrainGlb(url) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => resolve(gltf.scene),
      undefined,
      reject
    );
  });
}

async function attachBrain() {
  const url = manifest.mesh && manifest.mesh.wholeBrainUrl;
  if (!url) {
    setStatus("Manifest missing mesh.wholeBrainUrl — run pipeline.", true);
    return;
  }
  try {
    if (brainRoot) {
      volumeRoot.remove(brainRoot);
      brainRoot.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
          const ms = Array.isArray(o.material) ? o.material : [o.material];
          ms.forEach((m) => {
            if (m.map) m.map.dispose();
            m.dispose();
          });
        }
      });
      brainRoot = null;
    }
    brainRoot = await loadBrainGlb(url);
    alignBrainToVolumeGrid(brainRoot);
    volumeRoot.add(brainRoot);
    setClipPlaneForActiveAxis();
    applyClippingToBrain();
    setStatus("Drag to orbit · slider moves the cut");
  } catch (e) {
    setStatus("Could not load brain GLB.", true);
  }
}

function applyPlane() {
  plane = el.plane.value;
  const oblique_ = isObliqueActive();
  el.cardinalIndexRow.hidden = oblique_;
  el.prev.hidden = oblique_;
  el.next.hidden = oblique_;
  el.obliqueControls.hidden = !oblique_;

  if (!oblique_ && obliquePngRafId) {
    cancelAnimationFrame(obliquePngRafId);
    obliquePngRafId = 0;
  }

  if (oblique_) {
    recomputeObliqueBasis();
  } else {
    syncSliderFromState();
  }
  refresh2D();
  refreshCut();
}

function updateObliqueGeometryOnly() {
  if (!scene || !isObliqueActive()) return;
  setClipPlaneForActiveAxis();
  layoutSliceSheet();
}

function onIndexInput() {
  if (isObliqueActive()) return;
  const spec = currentPlaneSpec();
  let v = Number(el.index.value);
  v = Math.min(Math.max(0, v), spec.count - 1);
  el.index.value = String(v);
  writeActiveIndexToState(v);
  el.indexLabel.textContent = String(v);
  refresh2D();
  refreshCut();
}

function onObliqueAngleInput() {
  oblique.yawDeg = Number(el.yaw.value);
  oblique.pitchDeg = Number(el.pitch.value);
  el.yawLabel.textContent = oblique.yawDeg.toFixed(0);
  el.pitchLabel.textContent = oblique.pitchDeg.toFixed(0);
  recomputeObliqueBasis();
  updateObliqueGeometryOnly();
  scheduleObliquePngRefresh();
}

function onObliqueOffsetInput() {
  oblique.offsetMm = Number(el.obliqueOffset.value);
  el.obliqueOffsetLabel.textContent = oblique.offsetMm.toFixed(1);
  // Offset changes do not invalidate the basis, only the plane position.
  updateObliqueGeometryOnly();
  scheduleObliquePngRefresh();
}

function resetOblique() {
  oblique.yawDeg = 0;
  oblique.pitchDeg = 0;
  oblique.offsetMm = 0;
  el.yaw.value = "0";
  el.pitch.value = "0";
  el.yawLabel.textContent = "0";
  el.pitchLabel.textContent = "0";
  recomputeObliqueBasis();
  updateObliqueGeometryOnly();
  scheduleObliquePngRefresh(true);
}

/** Coalesce oblique PNG updates to ~one per animation frame while dragging. */
let obliquePngRafId = 0;
function flushObliquePngRefresh() {
  if (!isObliqueActive()) return;
  refresh2D();
  updateSliceSheetTexture();
}
function scheduleObliquePngRefresh(immediate = false) {
  if (!isObliqueActive()) return;
  if (immediate) {
    if (obliquePngRafId) {
      cancelAnimationFrame(obliquePngRafId);
      obliquePngRafId = 0;
    }
    flushObliquePngRefresh();
    return;
  }
  if (obliquePngRafId) return;
  obliquePngRafId = requestAnimationFrame(() => {
    obliquePngRafId = 0;
    flushObliquePngRefresh();
  });
}

el.plane.addEventListener("change", applyPlane);
el.index.addEventListener("input", onIndexInput);
el.yaw.addEventListener("input", onObliqueAngleInput);
el.pitch.addEventListener("input", onObliqueAngleInput);
el.obliqueOffset.addEventListener("input", onObliqueOffsetInput);
el.resetOblique.addEventListener("click", resetOblique);
el.prev.addEventListener("click", () => {
  if (isObliqueActive()) return;
  const spec = currentPlaneSpec();
  let v = readActiveIndexFromState() - 1;
  if (v < 0) v = spec.count - 1;
  writeActiveIndexToState(v);
  syncSliderFromState();
  refresh2D();
  refreshCut();
});
el.next.addEventListener("click", () => {
  if (isObliqueActive()) return;
  const spec = currentPlaneSpec();
  let v = readActiveIndexFromState() + 1;
  if (v >= spec.count) v = 0;
  writeActiveIndexToState(v);
  syncSliderFromState();
  refresh2D();
  refreshCut();
});

el.slice.addEventListener("error", () => {
  setStatus(
    isObliqueActive()
      ? "Oblique slice request failed — is /api/slice/oblique reachable?"
      : "Failed to load slice PNG — run pipeline and refresh.",
    true
  );
});
el.slice.addEventListener("load", () => {
  if (!manifest) return;
  const datasetLabel = manifest.datasetId ?? "dataset";
  if (isObliqueActive()) {
    setStatus(
      `${datasetLabel} · oblique · yaw ${oblique.yawDeg.toFixed(0)}° ` +
        `pitch ${oblique.pitchDeg.toFixed(0)}° offset ${oblique.offsetMm.toFixed(1)} mm`
    );
  } else {
    const spec = currentPlaneSpec();
    const n = readActiveIndexFromState() + 1;
    setStatus(`${datasetLabel} · ${plane} slice ${n}/${spec.count} · orbit 3D`);
  }
});

async function init() {
  try {
    const res = await fetch(MANIFEST_URL);
    if (!res.ok) throw new Error(res.status + " " + res.statusText);
    manifest = await res.json();
  } catch {
    setStatus("No manifest — run pipeline/process_subject.py first.", true);
    return;
  }

  const planes = Object.keys(manifest.slices || {});
  if (planes.length === 0 || !manifest.volume) {
    setStatus("Manifest missing volume or slices.", true);
    return;
  }

  idxI = manifest.slices.sagittal.defaultIndex;
  idxJ = manifest.slices.coronal.defaultIndex;
  idxK = manifest.slices.axial.defaultIndex;

  const planeOptions = [...planes, OBLIQUE_PLANE_NAME];
  el.plane.innerHTML = planeOptions
    .map((p) => `<option value="${p}">${p}</option>`)
    .join("");
  el.plane.value = planes.includes("axial") ? "axial" : planes[0];
  plane = el.plane.value;

  initThree();
  onResize();
  syncSliderFromState();
  refresh2D();
  setClipPlaneForActiveAxis();
  await attachBrain();
  layoutSliceSheet();
  await updateSliceSheetTexture();
}

init();

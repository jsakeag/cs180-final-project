import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { buildColliders, resolveCircle, groundHeight } from './collision.js';

// ---------- config (tweak while testing) ----------
const CFG = {
  eyeHeight: 1.6,
  radius: 0.3,          // player footprint radius (m)
  walkSpeed: 3.0,       // m/s
  sprintMult: 1.8,
  accel: 12,            // how fast velocity reaches target (1/s)
  mouseSens: 0.0022,
  jumpSpeed: 4.8,       // m/s up  (apex = v^2 / 2g ≈ 0.82 m)
  gravity: 14,          // m/s^2
  fovDefault: 70, fovMin: 25, fovMax: 90,
  flySpeed: 12,         // m/s when flying (Shift = x3)
  flyClimb: 8,          // m/s up (Space) / down (Q)
  thirdDist: 3.0, thirdMin: 1.0, thirdMax: 8.0,
};

const params = new URLSearchParams(location.search);
const scenePath = params.get('scene') || 'scenes/demo-room.json';

// ---------- renderer / scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1d2228);

const camera = new THREE.PerspectiveCamera(CFG.fovDefault, innerWidth / innerHeight, 0.05, 1500);
camera.rotation.order = 'YXZ';

scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
scene.add(sun, sun.target);

const worldGroup = new THREE.Group();    // walls + floor
const objectGroup = new THREE.Group();   // placed objects
const debugGroup = new THREE.Group();    // collider outlines
const labelGroup = new THREE.Group();
scene.add(worldGroup, objectGroup, debugGroup, labelGroup);
debugGroup.visible = false;

// third-person avatar
const avatar = new THREE.Mesh(
  new THREE.CapsuleGeometry(CFG.radius, CFG.eyeHeight - 2 * CFG.radius, 4, 12),
  new THREE.MeshStandardMaterial({ color: 0x4fa3ff })
);
avatar.castShadow = true;
avatar.visible = false;
scene.add(avatar);

// ---------- settings (Esc menu; remembered per browser) ----------
const SETTINGS_KEY = 'engine-settings';
const settings = { photoTextures: true, relief: true, predicted: true, labels: true };
try { Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')); } catch { /* private mode */ }
function saveSettings() { try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* ignore */ } }

// ---------- state ----------
const player = { x: 0, y: 0, z: 0, yaw: 0, pitch: 0, vx: 0, vy: 0, vz: 0, grounded: true, hits: [] };
let sceneDef = null;
let colliders = [];
let grid = null;
let thirdPerson = false;
let thirdDist = CFG.thirdDist;
let flying = false;      // F toggles
let panelOpen = false;    // paint panel up: don't show the Esc menu
let photoCam = null;     // when set, camera is locked to a photo's recovered viewpoint
let photoIdx = -1;
const keys = new Set();

// ---------- helpers ----------
function colorFor(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return new THREE.Color().setHSL((h % 360) / 360, 0.45, 0.55);
}

function makeLabel(text) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  ctx.font = '28px Menlo, monospace';
  const w = Math.ceil(ctx.measureText(text).width) + 16;
  c.width = w; c.height = 40;
  ctx.font = '28px Menlo, monospace';
  ctx.fillStyle = 'rgba(0,0,0,.6)';
  ctx.fillRect(0, 0, w, 40);
  ctx.fillStyle = '#fff';
  ctx.fillText(text, 8, 29);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c) }));
  sprite.scale.set(w / 160, 40 / 160, 1);
  return sprite;
}

function colliderOutline(c) {
  const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(c.hx * 2, 0.02, c.hz * 2));
  const line = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0xff3366 }));
  line.position.set(c.cx, 0.02, c.cz);
  line.rotation.y = -c.angle;
  line.userData.colliderId = c.id;
  return line;
}

const gltfLoader = new GLTFLoader();
const texLoader = new THREE.TextureLoader();
function photoTexture(path) {
  const t = texLoader.load(path);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}
const models = { total: 0, loaded: 0, failed: 0 };  // shown in the HUD

// ---------- projecting the photo onto furniture ----------
// Each photo becomes a projector with the photo camera's real intrinsics. A model
// fragment takes the photo's pixel when (1) it's visible from that camera (depth pre-pass
// from the photo camera) and (2) the pixel belongs to that object's silhouette in the
// photo (label image from texture.py). From the photo viewpoint the model looks like the
// photo; elsewhere it keeps its own (photo-recolored) texture.
const PROJ_NEAR = 0.3, PROJ_FAR = 40, PROJ_BIAS = 0.0015;
let projectors = [];
function makeProjector(c) {
  const [fx, fy, cx, cy] = c.K, [w, h] = c.size;
  const pcam = new THREE.PerspectiveCamera();
  pcam.rotation.order = 'YXZ';
  pcam.position.set(c.position[0], c.position[1], c.position[2]);
  pcam.rotation.set(THREE.MathUtils.degToRad(c.pitch_deg), THREE.MathUtils.degToRad(c.yaw_deg), 0);
  pcam.updateMatrixWorld(true);
  const n = PROJ_NEAR, f = PROJ_FAR;
  const P = new THREE.Matrix4().set(
    2 * fx / w, 0, 1 - 2 * cx / w, 0,
    0, 2 * fy / h, 2 * cy / h - 1, 0,
    0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
    0, 0, -1, 0);
  pcam.projectionMatrix.copy(P);
  pcam.projectionMatrixInverse.copy(P).invert();
  const rt = new THREE.WebGLRenderTarget(1024, Math.round(1024 * h / w));
  rt.depthTexture = new THREE.DepthTexture(rt.width, rt.height);
  const photo = texLoader.load(c.photo);              // raw sRGB: written after color conversion
  photo.colorSpace = THREE.NoColorSpace;
  const labels = texLoader.load(c.labels);
  labels.colorSpace = THREE.NoColorSpace;
  labels.magFilter = labels.minFilter = THREE.NearestFilter;
  labels.generateMipmaps = false;
  return { pcam, rt, photo, labels, VP: new THREE.Matrix4().multiplyMatrices(P, pcam.matrixWorldInverse) };
}
function projectedMaterial(mat, proj, label) {
  const m = mat.clone();
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, {
      photoVP: { value: proj.VP }, photoTex: { value: proj.photo }, photoLabels: { value: proj.labels },
      photoDepth: { value: proj.rt.depthTexture }, photoLabel: { value: label },
    });
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vPhotoWorld;')
      .replace('#include <project_vertex>', '#include <project_vertex>\nvPhotoWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
varying vec3 vPhotoWorld;
uniform mat4 photoVP;
uniform sampler2D photoTex;
uniform sampler2D photoLabels;
uniform sampler2D photoDepth;
uniform float photoLabel;`)
      .replace('#include <dithering_fragment>', `{
  vec4 pc = photoVP * vec4(vPhotoWorld, 1.0);
  if (pc.w > 0.0) {
    vec3 nd = pc.xyz / pc.w;
    vec2 puv = nd.xy * 0.5 + 0.5;
    if (puv.x > 0.0 && puv.x < 1.0 && puv.y > 0.0 && puv.y < 1.0) {
      float seenDepth = texture2D(photoDepth, puv).x;
      float lbl = floor(texture2D(photoLabels, puv).r * 255.0 + 0.5);
      if (nd.z * 0.5 + 0.5 <= seenDepth + ${PROJ_BIAS} && abs(lbl - photoLabel) < 0.5)
        gl_FragColor.rgb = texture2D(photoTex, puv).rgb;
    }
  }
}
#include <dithering_fragment>`);
  };
  m.customProgramCacheKey = () => 'photo-projected';
  return m;
}
const depthOnly = new THREE.MeshBasicMaterial({ colorWrite: false });
function renderProjectionDepth() {
  if (!projectors.some(Boolean)) return;
  const hidden = [labelGroup, debugGroup, avatar].filter((g) => g.visible);
  hidden.forEach((g) => { g.visible = false; });
  const gridWas = grid?.visible;
  if (grid) grid.visible = false;
  scene.overrideMaterial = depthOnly;
  for (const p of projectors.filter(Boolean)) {
    renderer.setRenderTarget(p.rt);
    renderer.clear();
    renderer.render(scene, p.pcam);
  }
  renderer.setRenderTarget(null);
  scene.overrideMaterial = null;
  hidden.forEach((g) => { g.visible = true; });
  if (grid) grid.visible = gridWas;
}

// ---------- outdoor (campus) ----------
const OUT_MAT = {
  building: new THREE.MeshStandardMaterial({ color: 0xd9d2c3 }),
  roof: new THREE.MeshStandardMaterial({ color: 0x8c6f5a }),
  grass: new THREE.MeshStandardMaterial({ color: 0x6f9a52, side: THREE.DoubleSide }),
  plaza: new THREE.MeshStandardMaterial({ color: 0xc9c2b5, side: THREE.DoubleSide }),
  path: new THREE.MeshStandardMaterial({ color: 0xe8e1d2, side: THREE.DoubleSide }),
  steps: new THREE.MeshStandardMaterial({ color: 0xb3ab9c, side: THREE.DoubleSide }),
};
const flatShape = (poly) => new THREE.Shape(poly.map(([x, z]) => new THREE.Vector2(x, -z)));
function flatMesh(geom, mat, y) {
  const m = new THREE.Mesh(geom, mat);
  m.rotation.x = -Math.PI / 2;
  m.position.y = y;
  m.receiveShadow = true;
  return m;
}
function stripGeometry(lines) {
  // each polyline segment -> a quad of the path's width, all in one geometry
  const pos = [];
  for (const { line, width } of lines) {
    for (let i = 0; i + 1 < line.length; i++) {
      const [x0, z0] = line[i], [x1, z1] = line[i + 1];
      const L = Math.hypot(x1 - x0, z1 - z0) || 1;
      const nx = -(z1 - z0) / L * width / 2, nz = (x1 - x0) / L * width / 2;
      const a = [x0 + nx, -(z0 + nz)], b = [x0 - nx, -(z0 - nz)], c = [x1 - nx, -(z1 - nz)], d = [x1 + nx, -(z1 + nz)];
      pos.push(...a, 0, ...b, 0, ...c, 0, ...a, 0, ...c, 0, ...d, 0);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return g;
}
function buildOutdoor(sch) {
  const surf = sch.surfaces ?? {};
  for (const poly of surf.grass ?? []) worldGroup.add(flatMesh(new THREE.ShapeGeometry(flatShape(poly)), OUT_MAT.grass, 0.02));
  for (const poly of surf.plaza ?? []) worldGroup.add(flatMesh(new THREE.ShapeGeometry(flatShape(poly)), OUT_MAT.plaza, 0.015));
  if (surf.path?.length) worldGroup.add(flatMesh(stripGeometry(surf.path), OUT_MAT.path, 0.03));
  if (surf.steps?.length) worldGroup.add(flatMesh(stripGeometry(surf.steps), OUT_MAT.steps, 0.035));
  (sch.blocks ?? []).forEach((b, bi) => {
    const g = new THREE.ExtrudeGeometry(flatShape(b.polygon), { depth: b.height, bevelEnabled: false });
    const m = new THREE.Mesh(g, [OUT_MAT.roof, OUT_MAT.building]);
    m.userData.block = bi;
    m.rotation.x = -Math.PI / 2;
    m.castShadow = m.receiveShadow = true;
    worldGroup.add(m);
    if (b.name) {
      const cx = b.polygon.reduce((s, p) => s + p[0], 0) / b.polygon.length;
      const cz = b.polygon.reduce((s, p) => s + p[1], 0) / b.polygon.length;
      const label = makeLabel(b.name);
      label.position.set(cx, b.height + 2, cz);
      label.scale.multiplyScalar(4);
      labelGroup.add(label);
    }
  });
}
// ---------- Paint Berkeley: facades from photos + rules that reshape buildings ----------
const STONE = new THREE.MeshStandardMaterial({ color: 0xe9e4d8, roughness: 0.85 });
const TERRACOTTA = new THREE.MeshStandardMaterial({ color: 0xb4532f, roughness: 0.8, side: THREE.DoubleSide });
function wallFrame(block, edge, span, edgeEnd = edge) {
  // right-handed frame on a building wall: x along the wall (viewer's right when facing
  // it from outside), y up, z out of the wall. Origin = the wall's left end at the ground.
  const P = block.polygon;
  let area = 0;
  for (let i = 0; i < P.length; i++) { const [x0, z0] = P[i], [x1, z1] = P[(i + 1) % P.length]; area += x0 * z1 - x1 * z0; }
  const a = new THREE.Vector3(P[edge][0], 0, P[edge][1]);
  const b = new THREE.Vector3(P[(edgeEnd + 1) % P.length][0], 0, P[(edgeEnd + 1) % P.length][1]);   // a run of edges = one wall
  const t = b.clone().sub(a).normalize();
  const n = new THREE.Vector3(t.z, 0, -t.x).multiplyScalar(area > 0 ? 1 : -1);   // outward
  const up = new THREE.Vector3(0, 1, 0);
  const x = up.clone().cross(n);                                                   // viewer's right
  const [s0, s1] = span;
  const origin = x.dot(t) > 0 ? a.clone().addScaledVector(t, s0) : a.clone().addScaledVector(t, s1);
  const g = new THREE.Group();
  g.matrixAutoUpdate = false;
  g.matrix.makeBasis(x, up, n).setPosition(origin);
  return g;
}
function columnMesh(radius, height, order) {
  // turned column: base, tapering shaft (entasis), capital by order
  const r = radius, h = height, pts = [];
  const cap = order === 'corinthian' ? 0.12 : order === 'ionic' ? 0.07 : 0.05;
  pts.push([0, 0], [r * 1.35, 0], [r * 1.35, h * 0.03], [r * 1.15, h * 0.05], [r, h * 0.06]);
  for (let i = 1; i <= 8; i++) { const f = i / 8; pts.push([r * (1 - 0.14 * f * f), h * (0.06 + f * (0.94 - cap - 0.06))]); }
  pts.push([r * (order === 'corinthian' ? 1.25 : 1.05), h * (1 - cap * 0.4)], [r * 1.3, h * (1 - cap * 0.2)], [r * 1.3, h], [0, h]);
  const g = new THREE.Group();
  g.add(new THREE.Mesh(new THREE.LatheGeometry(pts.map(([u, v]) => new THREE.Vector2(u, v)), 24), STONE));
  const abacus = new THREE.Mesh(new THREE.BoxGeometry(r * 2.8, h * 0.025, r * 2.8), STONE);
  abacus.position.y = h - h * 0.0125;
  g.add(abacus);
  if (order === 'ionic') {      // scroll volutes on both sides
    for (const sgn of [-1, 1]) {
      const v = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.32, r * 0.32, r * 2.4, 16), STONE);
      v.rotation.x = Math.PI / 2;
      v.position.set(sgn * r * 1.15, h * (1 - 0.06), 0);
      g.add(v);
    }
  }
  g.traverse((m) => { if (m.isMesh) m.castShadow = m.receiveShadow = true; });
  return g;
}
function archSurround(x0, y0, x1, y1, depth) {
  const r = (x1 - x0) / 2, m = Math.max(0.25, 0.3 * r);
  const outer = new THREE.Shape([[x0 - m, y0], [x1 + m, y0], [x1 + m, y1 + m], [x0 - m, y1 + m]].map(([u, v]) => new THREE.Vector2(u, v)));
  const hole = new THREE.Path();
  hole.moveTo(x0, y0);
  hole.lineTo(x1, y0);
  hole.lineTo(x1, Math.max(y0, y1 - r));
  hole.absarc((x0 + x1) / 2, Math.max(y0, y1 - r), r, 0, Math.PI, false);
  hole.lineTo(x0, y0);
  outer.holes.push(hole);
  const mesh = new THREE.Mesh(new THREE.ExtrudeGeometry(outer, { depth, bevelEnabled: false }), STONE);
  mesh.castShadow = mesh.receiveShadow = true;
  return mesh;
}
function tileCap(block) {
  // footprint isn't a rectangle (wings, courtyards): a low tiled cap that follows it
  const shape = new THREE.Shape(block.polygon.map(([x, z]) => new THREE.Vector2(x, -z)));
  const m = new THREE.Mesh(new THREE.ExtrudeGeometry(shape, { depth: 1.2, bevelEnabled: false }), TERRACOTTA);
  m.rotation.x = -Math.PI / 2;
  m.position.y = block.height;
  m.castShadow = m.receiveShadow = true;
  return m;
}
function hipRoof(block) {
  // terracotta hip roof over the footprint's oriented bounding box (PCA axes), if the
  // footprint is close to that box; otherwise a cap that follows the footprint
  const P = block.polygon, n = P.length;
  const cx = P.reduce((s, p) => s + p[0], 0) / n, cz = P.reduce((s, p) => s + p[1], 0) / n;
  let sxx = 0, szz = 0, sxz = 0;
  for (const [x, z] of P) { sxx += (x - cx) ** 2; szz += (z - cz) ** 2; sxz += (x - cx) * (z - cz); }
  const ang = 0.5 * Math.atan2(2 * sxz, sxx - szz);
  const u = [Math.cos(ang), Math.sin(ang)], v = [-Math.sin(ang), Math.cos(ang)];
  let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
  for (const [x, z] of P) {
    const pu = (x - cx) * u[0] + (z - cz) * u[1], pv = (x - cx) * v[0] + (z - cz) * v[1];
    u0 = Math.min(u0, pu); u1 = Math.max(u1, pu); v0 = Math.min(v0, pv); v1 = Math.max(v1, pv);
  }
  let area = 0;
  for (let i = 0; i < n; i++) { const [x0, z0] = P[i], [x1, z1] = P[(i + 1) % n]; area += x0 * z1 - x1 * z0; }
  if (Math.abs(area / 2) < 0.75 * (u1 - u0) * (v1 - v0)) return tileCap(block);
  const o = 0.6;  // eave overhang
  u0 -= o; u1 += o; v0 -= o; v1 += o;
  const L = u1 - u0, W = v1 - v0, long = L >= W;
  const half = Math.min(L, W) / 2, rise = Math.min(6, half * Math.tan(THREE.MathUtils.degToRad(24)));
  const um = (u0 + u1) / 2, vm = (v0 + v1) / 2;
  const r0 = long ? [u0 + half, vm] : [um, v0 + half], r1 = long ? [u1 - half, vm] : [um, v1 - half];
  const W2 = (pu, pv, y) => [cx + pu * u[0] + pv * v[0], y, cz + pu * u[1] + pv * v[1]];
  const y0 = block.height, y1 = block.height + rise;
  const c = [W2(u0, v0, y0), W2(u1, v0, y0), W2(u1, v1, y0), W2(u0, v1, y0)];
  const R0 = W2(...r0, y1), R1 = W2(...r1, y1);
  const tris = long
    ? [[c[0], c[1], R1], [c[0], R1, R0], [c[2], c[3], R0], [c[2], R0, R1], [c[1], c[2], R1], [c[3], c[0], R0]]
    : [[c[1], c[2], R1], [c[1], R1, R0], [c[3], c[0], R0], [c[3], R0, R1], [c[0], c[1], R0], [c[2], c[3], R1]];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(tris.flat(2), 3));
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, TERRACOTTA);
  m.castShadow = m.receiveShadow = true;
  return m;
}
// relief: base plane this far out from the wall, so recesses (down to -RELIEF_BASE) stay visible
const RELIEF_BASE = 1.2;
const reliefCache = new Map();
function loadRelief(info) {
  if (!reliefCache.has(info.image)) reliefCache.set(info.image, new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, img.width, img.height).data;
      const [lo, hi] = info.range, z = new Float32Array(img.width * img.height);
      for (let i = 0; i < z.length; i++) z[i] = lo + (d[i * 4] / 255) * (hi - lo);
      resolve({ w: img.width, h: img.height, z });
    };
    img.onerror = () => resolve(null);
    img.src = info.image;
  }));
  return reliefCache.get(info.image);
}
function reliefGeometry(relief, L, H, reps) {
  // grid over the wall (x 0..L, y 0..H); each vertex pushed out by the relief sample
  const nx = Math.min(1200, relief.w * reps), ny = Math.min(240, relief.h);
  const g = new THREE.PlaneGeometry(L, H, nx - 1, ny - 1);
  g.translate(L / 2, H / 2, 0);
  const pos = g.attributes.position, uv = g.attributes.uv;
  for (let i = 0; i < pos.count; i++) {
    const u = uv.getX(i) * reps, v = uv.getY(i);
    const fx = (u - Math.floor(u === reps ? reps - 1e-6 : u)) * (relief.w - 1), fy = (1 - v) * (relief.h - 1);
    const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(x0 + 1, relief.w - 1), y1 = Math.min(y0 + 1, relief.h - 1);
    const ax = fx - x0, ay = fy - y0, Z = relief.z, W = relief.w;
    const zz = (1 - ax) * (1 - ay) * Z[y0 * W + x0] + ax * (1 - ay) * Z[y0 * W + x1] + (1 - ax) * ay * Z[y1 * W + x0] + ax * ay * Z[y1 * W + x1];
    pos.setZ(i, Math.max(zz, -RELIEF_BASE + 0.06));
  }
  g.computeVertexNormals();
  return g;
}
async function buildPaintings(def) {
  const blocks = def.schematic?.blocks ?? [];
  const roofed = new Set();
  for (const app of def.paintings ?? []) {
    const p = await fetch(`paintings/${app.painting}/painting.json`).then((r) => r.json()).catch(() => null);
    if (!p || sceneDef !== def) continue;
    const block = blocks[app.block];
    if (!block) continue;
    const frame = wallFrame(block, app.edge, app.span, app.edge_end ?? app.edge);
    const L = app.span[1] - app.span[0], H = block.height;
    const [Ws, Hs] = p.size_m;
    const reps = Math.max(1, Math.round(L / Ws)), sx = L / (reps * Ws), sy = H / Hs;
    // the facade itself, repeated to the wall's length, stretched to its height; with its
    // relief map, a displaced surface (the photo's depth: columns out, windows in)
    const relief = settings.photoTextures && settings.relief && p.relief ? await loadRelief(p.relief) : null;
    if (sceneDef !== def) return;
    if (settings.photoTextures) {
      const tex = photoTexture(p.texture);
      tex.wrapS = THREE.RepeatWrapping;
      tex.repeat.set(reps, 1);
      const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9 });
      let face;
      if (relief) {
        face = new THREE.Mesh(reliefGeometry(relief, L, H, reps), mat);
        face.position.set(0, 0, RELIEF_BASE);
        face.castShadow = true;
      } else {
        face = new THREE.Mesh(new THREE.PlaneGeometry(L, H), mat);
        face.position.set(L / 2, H / 2, 0.06);
      }
      face.receiveShadow = true;
      frame.add(face);
    }
    // rules: the building's shape follows what the facade showed
    for (const r of p.rules ?? []) {
      if (relief && (r.type === 'colonnade' || r.type === 'arcade')) continue;   // the photo's own relief shows them
      if (r.type === 'colonnade') {
        const rad = Math.min(1.2, r.radius * Math.min(sx, sy)), colH = (r.top - r.base) * sy, out = rad + 0.5;
        for (let i = 0; i < reps; i++) {     // each repeat of the facade gets its own portico
          let xmin = Infinity, xmax = -Infinity;
          for (const x of r.x) {
            const X = (x + i * Ws) * sx;
            const c = columnMesh(rad, colH, r.order);
            c.position.set(X, r.base * sy, out);
            frame.add(c);
            xmin = Math.min(xmin, X); xmax = Math.max(xmax, X);
          }
          const ent = new THREE.Mesh(new THREE.BoxGeometry(xmax - xmin + rad * 3, colH * 0.12, out + rad * 1.4), STONE);
          ent.position.set((xmin + xmax) / 2, r.base * sy + colH + colH * 0.06, (out + rad * 1.4) / 2);
          ent.castShadow = ent.receiveShadow = true;
          frame.add(ent);
          if (r.pediment) {
            const w = xmax - xmin + rad * 3, tri = new THREE.Shape([[-w / 2, 0], [w / 2, 0], [0, w * 0.18]].map(([u, v]) => new THREE.Vector2(u, v)));
            const ped = new THREE.Mesh(new THREE.ExtrudeGeometry(tri, { depth: out + rad * 1.4, bevelEnabled: false }), STONE);
            ped.position.set((xmin + xmax) / 2, r.base * sy + colH * 1.12, 0);
            frame.add(ped);
          }
        }
      } else if (r.type === 'arcade') {
        for (let i = 0; i < reps; i++) for (const [x0, y0, x1, y1] of r.arches) {
          frame.add(archSurround((x0 + i * Ws) * sx, y0 * sy, (x1 + i * Ws) * sx, y1 * sy, 0.4));
        }
      } else if (r.type === 'roof' && !roofed.has(app.block)) {
        roofed.add(app.block);
        worldGroup.add(hipRoof(block));
      }
    }
    frame.updateMatrixWorld(true);
    worldGroup.add(frame);
  }
}

// ---------- paint mode (B): aim at a building wall, click, paint it ----------
let paintMode = false, hoverWall = null, hoverMesh = null, paintTarget = null;
const paintRay = new THREE.Raycaster();
const HILITE = new THREE.MeshBasicMaterial({ color: 0xff7a2f, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthTest: false });
function wallRun(block, k, tolDeg = 12) {
  // consecutive footprint edges facing the same way = one wall (same rule as paint.py)
  const P = block.polygon, n = P.length;
  let area = 0;
  for (let i = 0; i < n; i++) { const [x0, z0] = P[i], [x1, z1] = P[(i + 1) % n]; area += x0 * z1 - x1 * z0; }
  const normal = (i) => {
    const a = P[((i % n) + n) % n], b = P[(((i + 1) % n) + n) % n];
    const dx = b[0] - a[0], dz = b[1] - a[1], L = Math.hypot(dx, dz) || 1, s = area > 0 ? 1 : -1;
    return [s * dz / L, -s * dx / L];
  };
  const nk = normal(k), cos = Math.cos(THREE.MathUtils.degToRad(tolDeg));
  const same = (i) => { const m = normal(i); return m[0] * nk[0] + m[1] * nk[1] > cos; };
  let first = k, last = k;
  while (same(first - 1) && k - first < n - 1) first--;
  while (same(last + 1) && last - first < n - 1) last++;
  first = ((first % n) + n) % n; last = ((last % n) + n) % n;
  const a = P[first], b = P[(last + 1) % n];
  return { first, last, L: Math.hypot(b[0] - a[0], b[1] - a[1]), normal: nk };
}
function updatePaintHover() {
  if (!paintMode || !sceneDef?.outdoor) return;
  paintRay.setFromCamera(new THREE.Vector2(0, 0), camera);
  paintRay.far = 250;
  const blocks = worldGroup.children.filter((m) => m.userData.block !== undefined);
  const hit = paintRay.intersectObjects(blocks, false)[0];
  let wall = null;
  if (hit) {
    const n = hit.face.normal.clone().transformDirection(hit.object.matrixWorld);
    if (Math.abs(n.y) < 0.4) {
      const bi = hit.object.userData.block, block = sceneDef.schematic.blocks[bi], P = block.polygon;
      let best = null;
      for (let k = 0; k < P.length; k++) {
        const a = P[k], c = P[(k + 1) % P.length], dx = c[0] - a[0], dz = c[1] - a[1], L2 = dx * dx + dz * dz || 1;
        const t = Math.max(0, Math.min(1, ((hit.point.x - a[0]) * dx + (hit.point.z - a[1]) * dz) / L2));
        const d = Math.hypot(hit.point.x - a[0] - t * dx, hit.point.z - a[1] - t * dz);
        if (!best || d < best.d) best = { d, k };
      }
      const run = wallRun(block, best.k);
      wall = { bi, block, ...run, near: [hit.point.x + n.x * 1.0, hit.point.z + n.z * 1.0] };
    }
  }
  const key = wall ? `${wall.bi}:${wall.first}` : null;
  if (key !== (hoverWall ? `${hoverWall.bi}:${hoverWall.first}` : null)) {
    if (hoverMesh) { worldGroup.remove(hoverMesh); hoverMesh = null; }
    if (wall) {
      hoverMesh = wallFrame(wall.block, wall.first, [0, wall.L], wall.last);
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(wall.L, wall.block.height), HILITE);
      plane.position.set(wall.L / 2, wall.block.height / 2, 0.15);
      plane.renderOrder = 10;
      hoverMesh.add(plane);
      hoverMesh.updateMatrixWorld(true);
      worldGroup.add(hoverMesh);
    }
  }
  hoverWall = wall;
  if (wall) hoverWall.near = wall.near;   // keep the latest aim point
}
function setPaintMode(on) {
  paintMode = on && !!sceneDef?.outdoor;
  if (!paintMode && hoverMesh) { worldGroup.remove(hoverMesh); hoverMesh = null; hoverWall = null; }
  document.getElementById('crosshair').classList.toggle('paint', paintMode);
}

// paint panel
const panel = document.getElementById('paint-panel');
const pp = { job: null, photo: null, img: new Image(), corners: [], polling: null };
const CORNER_NAMES = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
const ppCv = document.getElementById('pp-photo');
pp.img.onload = drawPanelPhoto;
async function openPaintPanel(wall) {
  paintTarget = wall;
  panelOpen = true;
  document.exitPointerLock();
  overlay.classList.add('hidden');
  panel.classList.remove('hidden');
  document.getElementById('pp-title').textContent = `Paint ${wall.block.name ?? 'this building'} · ${wall.L.toFixed(0)} m wall`;
  document.getElementById('pp-status').textContent = '';
  const api = (p, o) => fetch(p, o).then(async (r) => { if (!r.ok) throw new Error(await r.text()); return r.json(); });
  try {
    const [paintings, samples] = await Promise.all([api('/api/paintings'), api('/api/samples')]);
    const g = document.getElementById('pp-gallery');
    g.innerHTML = paintings.map((p) => `<img src="${p.texture}" data-id="${p.id}" title="${p.id} · ${p.building} · ${p.rules.join(', ') || 'no rules'}" alt="${p.id}">`).join('')
      || '<span class="dim">no facades painted yet</span>';
    for (const im of g.querySelectorAll('img')) im.onclick = () => applyPainting(im.dataset.id);
    const sg = document.getElementById('pp-samples');
    sg.innerHTML = samples.map((s) => `<img src="${s.url}" data-name="${s.name}" title="${s.title} (${s.license})${s.building ? ' · ' + s.building : ''}" alt="${s.title}">`).join('');
    for (const im of sg.querySelectorAll('img')) im.onclick = () => useSample(samples.find((s) => s.name === im.dataset.name));
  } catch (err) {
    document.getElementById('pp-status').textContent = 'Painting needs the scene builder server with painting support: '
      + 'start (or restart) .venv/bin/python builder/server.py and open the game from it.';
  }
}
function closePaintPanel() {
  panel.classList.add('hidden');
  panelOpen = false;
  renderer.domElement.requestPointerLock();
}
async function reloadScene() {
  const def = await fetch(`${scenePath}?t=${Date.now()}`).then((r) => r.json());
  sceneDef = def;
  rebuildKeepingPlayer();
  if (paintMode) setPaintMode(true);
}
async function applyPainting(id) {
  const st = document.getElementById('pp-status');
  st.textContent = `applying ${id}…`;
  const r = await fetch('/api/paint/apply', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ painting: id, near: paintTarget.near }) });
  if (!r.ok) { st.textContent = (await r.text()).slice(0, 200); return; }
  await reloadScene();
  closePaintPanel();
}
function showPanelPhoto(job, photo, corners = []) {
  Object.assign(pp, { job, photo, corners: corners.map((c) => [...c]) });
  ppCv.classList.remove('hidden');
  pp.img.src = photo.url;
}
async function useSample(s) {
  const r = await fetch('/api/samples/use', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: s.name }) }).then((x) => x.json());
  // suggested corners are in the original photo's pixels; the job copy may be smaller
  const k = r.images[0].width / s.width;
  showPanelPhoto(r.job, r.images[0], (s.corners ?? []).map(([x, y]) => [x * k, y * k]));
}
document.getElementById('pp-file').onchange = async (e) => {
  const fd = new FormData();
  fd.append('images', e.target.files[0]);
  document.getElementById('pp-status').textContent = 'uploading…';
  const r = await fetch('/api/upload', { method: 'POST', body: fd }).then((x) => x.json());
  showPanelPhoto(r.job, r.images[0]);
};
function drawPanelPhoto() {
  if (!pp.photo) return;
  const w = ppCv.clientWidth || 520, k = devicePixelRatio, s = w / pp.photo.width;
  ppCv.width = w * k; ppCv.height = pp.photo.height * s * k;
  const c = ppCv.getContext('2d');
  c.scale(k, k);
  c.drawImage(pp.img, 0, 0, w, pp.photo.height * s);
  if (pp.corners.length) {
    c.strokeStyle = '#ff7a2f'; c.lineWidth = 2; c.fillStyle = 'rgba(255,122,47,.18)';
    c.beginPath(); pp.corners.forEach(([x, y], i) => (i ? c.lineTo : c.moveTo).call(c, x * s, y * s));
    if (pp.corners.length === 4) c.closePath();
    c.fill(); c.stroke();
    pp.corners.forEach(([x, y], i) => { c.fillStyle = '#ff7a2f'; c.beginPath(); c.arc(x * s, y * s, 6, 0, 7); c.fill();
      c.fillStyle = '#fff'; c.font = 'bold 10px system-ui'; c.fillText(i + 1, x * s - 3, y * s + 3.5); });
  }
  document.getElementById('pp-help').textContent = pp.corners.length < 4
    ? `click the facade's ${CORNER_NAMES[pp.corners.length]} corner (rough is fine)` : 'facade outlined ✓ (reset to redo)';
  document.getElementById('pp-run').disabled = pp.corners.length !== 4;
}
ppCv.onclick = (e) => {
  if (pp.corners.length >= 4) return;
  const r = ppCv.getBoundingClientRect(), s = pp.photo.width / ppCv.clientWidth;
  pp.corners.push([(e.clientX - r.left) * s, (e.clientY - r.top) * s]);
  drawPanelPhoto();
};
document.getElementById('pp-reset').onclick = () => { pp.corners = []; drawPanelPhoto(); };
document.getElementById('pp-close').onclick = closePaintPanel;
document.getElementById('pp-run').onclick = async () => {
  const st = document.getElementById('pp-status');
  st.textContent = 'reading the facade (first run loads the models, ~30 s)…';
  document.getElementById('pp-run').disabled = true;
  const { job } = await fetch('/api/paint', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job: pp.job, name: pp.photo.name, corners: pp.corners, near: paintTarget.near,
                           label: document.getElementById('pp-label').value }) }).then((r) => r.json());
  clearInterval(pp.polling);
  pp.polling = setInterval(async () => {
    const j = await fetch(`/api/job/${job}`).then((r) => r.json());
    const last = j.log.filter((l) => /rules|elements|aspect/.test(l)).slice(-2).join(' · ');
    st.textContent = `${j.stage} · ${j.elapsed}s ${last ? '· ' + last.trim() : ''}`;
    if (!j.done) return;
    clearInterval(pp.polling);
    if (j.error) { st.textContent = j.error; document.getElementById('pp-run').disabled = false; return; }
    await reloadScene();
    closePaintPanel();
  }, 1500);
};

// simple stand-ins for street furniture without a model
const PROCEDURAL = {
  tree(o) {
    const g = new THREE.Group();
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.25, 3, 8), new THREE.MeshStandardMaterial({ color: 0x6b4a32 }));
    trunk.position.y = 1.5;
    const crown = new THREE.Mesh(new THREE.IcosahedronGeometry(2.6, 1), new THREE.MeshStandardMaterial({ color: 0x4f7a3a, flatShading: true }));
    crown.position.y = 5;
    for (const m of [trunk, crown]) { m.castShadow = m.receiveShadow = true; g.add(m); }
    return g;
  },
  lamp_post(o) {
    const g = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, o.size[1], 8), new THREE.MeshStandardMaterial({ color: 0x2e3238 }));
    pole.position.y = o.size[1] / 2;
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.22, 12, 8), new THREE.MeshStandardMaterial({ color: 0xfff2c8, emissive: 0x443a20 }));
    lamp.position.y = o.size[1];
    for (const m of [pole, lamp]) { m.castShadow = true; g.add(m); }
    return g;
  },
};

// Fit a loaded model uniformly into the object's bbox, bottom on the floor.
function fitModel(root, size) {
  const box = new THREE.Box3().setFromObject(root);
  const s = box.getSize(new THREE.Vector3());
  const k = Math.min(size[0] / s.x, size[1] / s.y, size[2] / s.z);
  root.scale.setScalar(k);
  const b2 = new THREE.Box3().setFromObject(root);
  const c2 = b2.getCenter(new THREE.Vector3());
  root.position.set(-c2.x, -b2.min.y, -c2.z);
}

// ---------- scene building ----------
function buildScene(def) {
  for (const g of [worldGroup, objectGroup, debugGroup, labelGroup]) g.clear();
  models.total = models.loaded = models.failed = 0;
  if (grid) scene.remove(grid);

  const sch = def.schematic ?? {};
  const wallH = sch.wall_height ?? 2.7;

  // floor
  const fmin = sch.floor?.min ?? [-10, -10];
  const fmax = sch.floor?.max ?? [10, 10];
  const fw = fmax[0] - fmin[0], fd = fmax[1] - fmin[1];
  // floor: the room polygon when given (rooms needn't be axis aligned), else the min/max box
  let floorGeom;
  const polys = sch.floor?.polygons ?? (sch.floor?.polygon ? [sch.floor.polygon] : null);
  if (polys) {   // room + attached spaces (hallways): one shape each
    floorGeom = new THREE.ShapeGeometry(polys.map((poly) =>
      new THREE.Shape(poly.map(([x, z]) => new THREE.Vector2(x - (fmin[0] + fw / 2), -(z - (fmin[1] + fd / 2)))))));
  } else {
    floorGeom = new THREE.PlaneGeometry(fw, fd);
  }
  const pt = settings.photoTextures ? def.photo_textures : null;
  for (const p of projectors) p?.rt.dispose();
  projectors = pt ? (def.cameras ?? []).map((c) => (c.K && c.photo && c.labels ? makeProjector(c) : null)) : [];
  let floorMat = new THREE.MeshStandardMaterial({ color: 0x8a8070 });
  if (pt?.floor) {
    // the photo's floor, rectified top-down in meters: uv from world x / z
    const [x0, z0] = pt.floor.min, [x1, z1] = pt.floor.max;
    const cx = fmin[0] + fw / 2, cz = fmin[1] + fd / 2;
    const pos = floorGeom.attributes.position, uv = [];
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i) + cx, z = cz - pos.getY(i);
      uv.push((x - x0) / (x1 - x0), 1 - (z - z0) / (z1 - z0));
    }
    floorGeom.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    floorMat = new THREE.MeshBasicMaterial({ map: photoTexture(pt.floor.image) });   // unlit: the photo has its own light
  }
  if (def.outdoor) {
    floorMat = new THREE.MeshStandardMaterial({ color: 0x9a968c });
    scene.background = new THREE.Color(0xa9c8e6);
    scene.fog = new THREE.Fog(0xa9c8e6, 200, 1000);
  } else {
    scene.background = new THREE.Color(0x1d2228);
    scene.fog = null;
  }
  const floor = new THREE.Mesh(floorGeom, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(fmin[0] + fw / 2, 0, fmin[1] + fd / 2);
  floor.receiveShadow = true;
  worldGroup.add(floor);

  grid = new THREE.GridHelper(Math.max(fw, fd), Math.ceil(Math.max(fw, fd)), 0x333333, 0x555555);
  grid.position.set(floor.position.x, 0.005, floor.position.z);
  grid.visible = false;
  scene.add(grid);

  // light over the middle of the floor
  sun.position.set(floor.position.x + 3, 10, floor.position.z + 4);
  sun.target.position.copy(floor.position);
  const ext = Math.max(fw, fd);
  const sx = def.outdoor ? 60 : ext;
  Object.assign(sun.shadow.camera, { left: -sx, right: sx, top: sx, bottom: -sx, far: def.outdoor ? 300 : 500 });
  sun.shadow.camera.updateProjectionMatrix();

  colliders = buildColliders(def);

  // walls (render from the same colliders so visuals == physics)
  const wallMat = new THREE.MeshStandardMaterial({ color: 0xd8d2c4 });
  for (const c of colliders.filter((c) => c.id.startsWith('wall:'))) {
    const geo = new THREE.BoxGeometry(c.hx * 2, wallH, c.hz * 2);
    let mat = wallMat;
    const img = pt?.walls?.[c.id.slice(5)];
    if (img) {
      // big faces (+z, -z) show the wall's photo texture, u running from -> to on both
      const uv = geo.attributes.uv;
      for (let i = 20; i < 24; i++) uv.setX(i, 1 - uv.getX(i));
      const photo = new THREE.MeshBasicMaterial({ map: photoTexture(img) });
      mat = [wallMat, wallMat, wallMat, wallMat, photo, photo];
    }
    const m = new THREE.Mesh(geo, mat);
    m.position.set(c.cx, wallH / 2, c.cz);
    m.rotation.y = -c.angle;
    m.castShadow = m.receiveShadow = true;
    worldGroup.add(m);
  }

  if (def.outdoor) { buildOutdoor(sch); buildPaintings(def); }

  // objects
  for (const o of def.objects ?? []) {
    const holder = new THREE.Group();
    holder.position.set(o.position[0], o.position[1] ?? 0, o.position[2]);
    holder.rotation.y = THREE.MathUtils.degToRad(o.rotation_y ?? 0);
    holder.userData = o;
    objectGroup.add(holder);

    const placeholder = PROCEDURAL[o.class] && !o.model ? PROCEDURAL[o.class](o) : new THREE.Mesh(
      new THREE.BoxGeometry(...o.size),
      new THREE.MeshStandardMaterial({ color: colorFor(o.class) })
    );
    if (placeholder.isMesh) placeholder.position.y = o.size[1] / 2;
    placeholder.castShadow = placeholder.receiveShadow = true;
    holder.add(placeholder);

    const modelPath = (settings.photoTextures && o.model_photo) || o.model;
    if (modelPath) {
      models.total++;
      gltfLoader.load(modelPath, (gltf) => {
        fitModel(gltf.scene, o.size);
        const proj = o.photo_proj && projectors[o.photo_proj.camera];
        gltf.scene.traverse((n) => {
          if (!n.isMesh) return;
          n.castShadow = n.receiveShadow = true;
          if (proj) n.material = Array.isArray(n.material)
            ? n.material.map((m) => projectedMaterial(m, proj, o.photo_proj.label))
            : projectedMaterial(n.material, proj, o.photo_proj.label);
        });
        holder.remove(placeholder);
        holder.add(gltf.scene);
        models.loaded++;
        if (models.loaded + models.failed === models.total) renderProjectionDepth();
      }, undefined, (err) => {
        models.failed++;
        console.warn(`model failed for ${o.id}:`, err);
        if (models.loaded + models.failed === models.total) renderProjectionDepth();
      });
    }

    const conf = o.source?.confidence;
    const predicted = o.source?.placed_by === 'predicted';
    const label = makeLabel(predicted ? `${o.class} (predicted)` : conf != null ? `${o.class} ${conf.toFixed(2)}` : o.class);
    holder.userData.predicted = label.userData.predicted = predicted;
    label.position.set(o.position[0], (o.position[1] ?? 0) + o.size[1] + 0.25, o.position[2]);
    labelGroup.add(label);
  }

  for (const c of colliders) debugGroup.add(colliderOutline(c));
  labelGroup.visible = settings.labels;
  applyPredicted();
  respawn();
}

function respawn() {
  let sp = sceneDef.spawn ?? { position: [0, 0, 0], yaw_deg: 0 };
  const at = params.get('at')?.split(',').map(Number);   // ?at=x,z,yaw(rad): start somewhere specific
  if (at?.length === 3 && at.every(Number.isFinite)) sp = { position: [at[0], 0, at[1]], yaw_deg: THREE.MathUtils.radToDeg(at[2]) };
  player.x = sp.position[0];
  player.y = sp.position[1] ?? 0;
  player.z = sp.position[2];
  player.yaw = THREE.MathUtils.degToRad(sp.yaw_deg ?? 0);
  player.pitch = 0;
  player.vx = player.vy = player.vz = 0;
  player.grounded = true;
}

// ---------- input ----------
const overlay = document.getElementById('overlay');
overlay.addEventListener('click', (e) => { if (e.target === overlay) renderer.domElement.requestPointerLock(); });
renderer.domElement.addEventListener('click', () => {
  if (paintMode && hoverWall && document.pointerLockElement === renderer.domElement) openPaintPanel(hoverWall);
  else if (!panelOpen) renderer.domElement.requestPointerLock();
});
document.addEventListener('pointerlockchange', () => {
  overlay.classList.toggle('hidden', document.pointerLockElement === renderer.domElement || panelOpen);
  if (!document.pointerLockElement) keys.clear();
});
document.addEventListener('mousemove', (e) => {
  if (document.pointerLockElement !== renderer.domElement) return;
  player.yaw -= e.movementX * CFG.mouseSens;
  player.pitch -= e.movementY * CFG.mouseSens;
  player.pitch = Math.max(-1.5, Math.min(1.5, player.pitch));
});
addEventListener('wheel', (e) => {
  if (thirdPerson) {
    thirdDist = THREE.MathUtils.clamp(thirdDist + e.deltaY * 0.005, CFG.thirdMin, CFG.thirdMax);
  } else {
    camera.fov = THREE.MathUtils.clamp(camera.fov + e.deltaY * 0.05, CFG.fovMin, CFG.fovMax);
    camera.updateProjectionMatrix();
  }
}, { passive: true });
addEventListener('keydown', (e) => {
  if (e.code === 'Space') e.preventDefault();
  keys.add(e.code);
  if (e.code === 'KeyC') debugGroup.visible = !debugGroup.visible;
  if (e.code === 'KeyL') toggleLabels();
  if (e.code === 'KeyG' && grid) grid.visible = !grid.visible;
  if (e.code === 'KeyR') respawn();
  if (e.code === 'KeyF') { flying = !flying; player.vy = 0; }
  if (e.code === 'KeyB') setPaintMode(!paintMode);
  if (e.code === 'KeyO') togglePredicted();
  if (e.code === 'KeyP') cyclePhotoCam();
  if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space', 'KeyQ'].includes(e.code) && photoCam) exitPhotoCam();
  if (e.code === 'KeyV') { thirdPerson = !thirdPerson; avatar.visible = thirdPerson; }
});
addEventListener('keyup', (e) => keys.delete(e.code));
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------- movement ----------
function updatePlayer(dt) {
  let f = 0, s = 0;
  if (keys.has('KeyW')) f += 1;
  if (keys.has('KeyS')) f -= 1;
  if (keys.has('KeyD')) s += 1;
  if (keys.has('KeyA')) s -= 1;
  const len = Math.hypot(f, s) || 1;
  f /= len; s /= len;

  // camera forward is -z rotated by yaw
  const sinY = Math.sin(player.yaw), cosY = Math.cos(player.yaw);
  const sprint = keys.has('ShiftLeft') || keys.has('ShiftRight');
  const speed = flying ? CFG.flySpeed * (sprint ? 3 : 1) : CFG.walkSpeed * (sprint ? CFG.sprintMult : 1);
  const tx = (-sinY * f + cosY * s) * speed;
  const tz = (-cosY * f - sinY * s) * speed;

  const a = 1 - Math.exp(-CFG.accel * dt);
  player.vx += (tx - player.vx) * a;
  player.vz += (tz - player.vz) * a;

  // substep so fast movement can't tunnel through thin walls
  const dist = Math.hypot(player.vx, player.vz) * dt;
  const steps = Math.max(1, Math.ceil(dist / (CFG.radius * 0.5)));
  const hits = new Set();
  const x0 = player.x, z0 = player.z;
  for (let i = 0; i < steps; i++) {
    const r = resolveCircle(player.x + (player.vx * dt) / steps, player.z + (player.vz * dt) / steps, CFG.radius, colliders, player.y);
    r.hits.forEach((h) => hits.add(h));
    player.x = r.x; player.z = r.z;
  }
  // on contact, keep only the velocity that actually happened (slide along walls, no stored push)
  if (hits.size && dt > 0) {
    player.vx = (player.x - x0) / dt;
    player.vz = (player.z - z0) / dt;
  }
  player.hits = [...hits];

  if (flying) {   // no gravity: Space climbs, Q descends; walls still block, roofs don't
    const climb = ((keys.has('Space') ? 1 : 0) - (keys.has('KeyQ') ? 1 : 0)) * CFG.flyClimb * (sprint ? 3 : 1);
    player.vy += (climb - player.vy) * a;
    player.y = Math.max(groundHeight(player.x, player.z, CFG.radius, colliders, player.y), player.y + player.vy * dt);
    player.grounded = false;
    return;
  }
  // vertical: jump, gravity, land on the floor or on top of objects
  if (keys.has('Space') && player.grounded) {
    player.vy = CFG.jumpSpeed;
    player.grounded = false;
  }
  // ground is looked up from the pre-fall height so fast falls can't skip a surface
  const ground = groundHeight(player.x, player.z, CFG.radius, colliders, player.y);
  player.vy -= CFG.gravity * dt;
  player.y += player.vy * dt;
  if (player.y <= ground) {
    player.y = ground;
    player.vy = 0;
    player.grounded = true;
  } else {
    player.grounded = false;
  }
}

function cyclePhotoCam() {
  const cams = sceneDef?.cameras ?? [];
  if (!cams.length) return;
  photoIdx = (photoIdx + 1) % cams.length;
  photoCam = cams[photoIdx];
  // stand where the photographer stood, looking the same way
  player.x = photoCam.position[0];
  player.z = photoCam.position[2];
  player.yaw = THREE.MathUtils.degToRad(photoCam.yaw_deg);
  player.pitch = THREE.MathUtils.degToRad(photoCam.pitch_deg);
  player.vx = player.vz = 0;
  camera.fov = photoCam.fov_v_deg;
  camera.updateProjectionMatrix();
}
function exitPhotoCam() {
  photoCam = null;
  camera.fov = CFG.fovDefault;
  camera.updateProjectionMatrix();
}

const raycaster = new THREE.Raycaster();
function updateCamera() {
  if (photoCam) {
    camera.position.set(photoCam.position[0], photoCam.position[1], photoCam.position[2]);
    camera.rotation.set(player.pitch, player.yaw, 0);
    avatar.visible = false;
    return;
  }
  avatar.visible = thirdPerson;
  const head = new THREE.Vector3(player.x, player.y + CFG.eyeHeight, player.z);
  camera.rotation.set(player.pitch, player.yaw, 0);
  if (!thirdPerson) {
    camera.position.copy(head);
  } else {
    avatar.position.set(player.x, player.y + CFG.eyeHeight / 2, player.z);
    avatar.rotation.y = player.yaw;
    const back = new THREE.Vector3(0, 0, 1).applyEuler(camera.rotation);
    // pull the camera in if a wall/object is between it and the head
    raycaster.set(head, back);
    raycaster.far = thirdDist;
    const hit = raycaster.intersectObjects([worldGroup, objectGroup], true)[0];
    const d = hit ? Math.max(0.2, hit.distance - 0.15) : thirdDist;
    camera.position.copy(head).addScaledVector(back, d);
  }
  // highlight colliders currently touched
  for (const line of debugGroup.children) {
    line.material.color.set(player.hits.includes(line.userData.colliderId) ? 0x33ff88 : 0xff3366);
  }
}

// O: hide / show furniture the room model predicted for the parts no photo saw
function applyPredicted() {
  const show = settings.predicted;
  for (const g of [objectGroup, labelGroup]) for (const n of g.children) if (n.userData.predicted) n.visible = show;
  const hidden = new Set(sceneDef.objects.filter((o) => o.source?.placed_by === 'predicted').map((o) => `obj:${o.id}`));
  colliders = buildColliders(sceneDef).filter((c) => show || !hidden.has(c.id));
  debugGroup.clear();
  for (const c of colliders) debugGroup.add(colliderOutline(c));
}
function togglePredicted() { settings.predicted = !settings.predicted; saveSettings(); applyPredicted(); renderMenu(); }

// rebuild the scene in place (keeps where the player is)
function rebuildKeepingPlayer() {
  const keep = { ...player };
  buildScene(sceneDef);
  Object.assign(player, keep);
}
function togglePhotoTextures() { settings.photoTextures = !settings.photoTextures; saveSettings(); rebuildKeepingPlayer(); renderMenu(); }
function toggleRelief() { settings.relief = !settings.relief; saveSettings(); rebuildKeepingPlayer(); renderMenu(); }
function toggleLabels() { settings.labels = !settings.labels; labelGroup.visible = settings.labels; saveSettings(); renderMenu(); }

// ---------- Esc menu ----------
const menu = document.getElementById('menu');
const MENU_ITEMS = [
  ['photoTextures', 'Photo textures', 'floor, walls and furniture colors taken from the photos (off = plain, faster)', togglePhotoTextures],
  ['relief', '3D facades', 'painted walls get the depth (columns out, windows in) estimated from their photo', toggleRelief],
  ['predicted', 'Predicted furniture', 'pieces the room model added where no photo looked', togglePredicted],
  ['labels', 'Labels', 'class names over objects', toggleLabels],
];
function renderMenu() {
  menu.querySelector('.items').innerHTML = MENU_ITEMS.map(([k, name, hint]) =>
    `<button class="toggle" data-k="${k}"><span class="state ${settings[k] ? 'on' : ''}">${settings[k] ? 'ON' : 'OFF'}</span>` +
    `<span class="name">${name}</span><span class="hint">${hint}</span></button>`).join('');
}
menu.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  e.stopPropagation();
  if (b.id === 'resume') { renderer.domElement.requestPointerLock(); return; }
  if (b.id === 'exit') { location.href = '/'; return; }
  const item = MENU_ITEMS.find(([k]) => k === b.dataset.k);
  if (item) item[3]();
});
renderMenu();

// ---------- HUD ----------
const hud = document.getElementById('hud');
const nPredicted = () => (sceneDef?.objects ?? []).filter((o) => o.source?.placed_by === 'predicted').length;
let fpsAcc = 0, fpsFrames = 0, fps = 0;
function updateHud(dt) {
  fpsAcc += dt; fpsFrames++;
  if (fpsAcc > 0.5) { fps = fpsFrames / fpsAcc; fpsAcc = fpsFrames = 0; }
  const deg = ((THREE.MathUtils.radToDeg(player.yaw) % 360) + 360) % 360;
  hud.textContent =
    `scene  ${sceneDef?.name ?? '-'}\n` +
    `pos    ${player.x.toFixed(2)}, ${player.y.toFixed(2)}, ${player.z.toFixed(2)}  ${flying ? 'flying (F to land)' : player.grounded ? 'grounded' : 'air'}\n` +
    `yaw    ${deg.toFixed(0)}°   speed ${Math.hypot(player.vx, player.vz).toFixed(2)} m/s\n` +
    `view   ${photoCam ? `photo ${photoCam.image.split('/').pop()} (WASD to leave)` : thirdPerson ? `3rd (${thirdDist.toFixed(1)} m)` : `1st (fov ${camera.fov.toFixed(0)})`}\n` +
    `touch  ${player.hits.join(' ') || '-'}\n` +
    (nPredicted() ? `room   ${sceneDef.schematic.room?.scene_type ?? ''} · ${nPredicted()} predicted ${settings.predicted ? '(O hides)' : '(hidden, O shows)'}\n` : '') +
    (paintMode ? `paint  ${hoverWall ? (hoverWall.block.name ?? 'building') + ' · click to paint this wall' : 'aim at a building wall'} (B to stop)\n` : '') +
    (models.total ? `models ${models.loaded}/${models.total}${models.failed ? ` (${models.failed} failed)` : models.loaded < models.total ? ' loading...' : ''}\n` : '') +
    `fps    ${fps.toFixed(0)}`;
}

// ---------- main loop ----------
const clock = new THREE.Clock();
function loop() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (sceneDef) {
    if (sceneDef.outdoor) {   // a 500 m shadow map is mush: keep a sharp 120 m one around the player
      sun.position.set(player.x + 40, 90, player.z + 50);
      sun.target.position.set(player.x, 0, player.z);
    }
    updatePlayer(dt);
    updateCamera();
    updatePaintHover();
    updateHud(dt);
  }
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

fetch(scenePath)
  .then((r) => { if (!r.ok) throw new Error(`${r.status} ${scenePath}`); return r.json(); })
  .then((def) => { sceneDef = def; buildScene(def); loop(); })
  .catch((err) => { hud.textContent = `failed to load scene: ${err.message}`; });

// expose for console poking while testing
window.dbg = { CFG, player, get colliders() { return colliders; }, scene,
  setPaintMode, openPaintPanel, get hoverWall() { return hoverWall; } };

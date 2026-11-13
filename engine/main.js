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

const camera = new THREE.PerspectiveCamera(CFG.fovDefault, innerWidth / innerHeight, 0.05, 200);
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
const settings = { photoTextures: true, predicted: true, labels: true };
try { Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')); } catch { /* private mode */ }
function saveSettings() { try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* ignore */ } }

// ---------- state ----------
const player = { x: 0, y: 0, z: 0, yaw: 0, pitch: 0, vx: 0, vy: 0, vz: 0, grounded: true, hits: [] };
let sceneDef = null;
let colliders = [];
let grid = null;
let thirdPerson = false;
let thirdDist = CFG.thirdDist;
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
  Object.assign(sun.shadow.camera, { left: -ext, right: ext, top: ext, bottom: -ext });
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

  // objects
  for (const o of def.objects ?? []) {
    const holder = new THREE.Group();
    holder.position.set(o.position[0], o.position[1] ?? 0, o.position[2]);
    holder.rotation.y = THREE.MathUtils.degToRad(o.rotation_y ?? 0);
    holder.userData = o;
    objectGroup.add(holder);

    const placeholder = new THREE.Mesh(
      new THREE.BoxGeometry(...o.size),
      new THREE.MeshStandardMaterial({ color: colorFor(o.class) })
    );
    placeholder.position.y = o.size[1] / 2;
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
  const sp = sceneDef.spawn ?? { position: [0, 0, 0], yaw_deg: 0 };
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
renderer.domElement.addEventListener('click', () => renderer.domElement.requestPointerLock());
document.addEventListener('pointerlockchange', () => {
  overlay.classList.toggle('hidden', document.pointerLockElement === renderer.domElement);
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
  if (e.code === 'KeyO') togglePredicted();
  if (e.code === 'KeyP') cyclePhotoCam();
  if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space'].includes(e.code) && photoCam) exitPhotoCam();
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
  const speed = CFG.walkSpeed * (keys.has('ShiftLeft') || keys.has('ShiftRight') ? CFG.sprintMult : 1);
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
function toggleLabels() { settings.labels = !settings.labels; labelGroup.visible = settings.labels; saveSettings(); renderMenu(); }

// ---------- Esc menu ----------
const menu = document.getElementById('menu');
const MENU_ITEMS = [
  ['photoTextures', 'Photo textures', 'floor, walls and furniture colors taken from the photos (off = plain, faster)', togglePhotoTextures],
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
    `pos    ${player.x.toFixed(2)}, ${player.y.toFixed(2)}, ${player.z.toFixed(2)}  ${player.grounded ? 'grounded' : 'air'}\n` +
    `yaw    ${deg.toFixed(0)}°   speed ${Math.hypot(player.vx, player.vz).toFixed(2)} m/s\n` +
    `view   ${photoCam ? `photo ${photoCam.image.split('/').pop()} (WASD to leave)` : thirdPerson ? `3rd (${thirdDist.toFixed(1)} m)` : `1st (fov ${camera.fov.toFixed(0)})`}\n` +
    `touch  ${player.hits.join(' ') || '-'}\n` +
    (nPredicted() ? `room   ${sceneDef.schematic.room?.scene_type ?? ''} · ${nPredicted()} predicted ${settings.predicted ? '(O hides)' : '(hidden, O shows)'}\n` : '') +
    (models.total ? `models ${models.loaded}/${models.total}${models.failed ? ` (${models.failed} failed)` : models.loaded < models.total ? ' loading...' : ''}\n` : '') +
    `fps    ${fps.toFixed(0)}`;
}

// ---------- main loop ----------
const clock = new THREE.Clock();
function loop() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (sceneDef) {
    updatePlayer(dt);
    updateCamera();
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
window.dbg = { CFG, player, get colliders() { return colliders; }, scene };

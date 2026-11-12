import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { buildColliders, resolveCircle } from './collision.js';

// ---------- config (tweak while testing) ----------
const CFG = {
  eyeHeight: 1.6,
  radius: 0.3,          // player footprint radius (m)
  walkSpeed: 3.0,       // m/s
  sprintMult: 1.8,
  accel: 12,            // how fast velocity reaches target (1/s)
  mouseSens: 0.0022,
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

// ---------- state ----------
const player = { x: 0, z: 0, yaw: 0, pitch: 0, vx: 0, vz: 0, hits: [] };
let sceneDef = null;
let colliders = [];
let grid = null;
let thirdPerson = false;
let thirdDist = CFG.thirdDist;
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
  if (grid) scene.remove(grid);

  const sch = def.schematic ?? {};
  const wallH = sch.wall_height ?? 2.7;

  // floor
  const fmin = sch.floor?.min ?? [-10, -10];
  const fmax = sch.floor?.max ?? [10, 10];
  const fw = fmax[0] - fmin[0], fd = fmax[1] - fmin[1];
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(fw, fd),
    new THREE.MeshStandardMaterial({ color: 0x8a8070 })
  );
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
    const m = new THREE.Mesh(new THREE.BoxGeometry(c.hx * 2, wallH, c.hz * 2), wallMat);
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

    if (o.model) {
      gltfLoader.load(o.model, (gltf) => {
        fitModel(gltf.scene, o.size);
        gltf.scene.traverse((n) => { if (n.isMesh) n.castShadow = n.receiveShadow = true; });
        holder.remove(placeholder);
        holder.add(gltf.scene);
      }, undefined, (err) => console.warn(`model failed for ${o.id}:`, err));
    }

    const conf = o.source?.confidence;
    const label = makeLabel(conf != null ? `${o.class} ${conf.toFixed(2)}` : o.class);
    label.position.set(o.position[0], (o.position[1] ?? 0) + o.size[1] + 0.25, o.position[2]);
    labelGroup.add(label);
  }

  for (const c of colliders) debugGroup.add(colliderOutline(c));
  respawn();
}

function respawn() {
  const sp = sceneDef.spawn ?? { position: [0, 0, 0], yaw_deg: 0 };
  player.x = sp.position[0];
  player.z = sp.position[2];
  player.yaw = THREE.MathUtils.degToRad(sp.yaw_deg ?? 0);
  player.pitch = 0;
  player.vx = player.vz = 0;
}

// ---------- input ----------
const overlay = document.getElementById('overlay');
overlay.addEventListener('click', () => renderer.domElement.requestPointerLock());
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
  keys.add(e.code);
  if (e.code === 'KeyC') debugGroup.visible = !debugGroup.visible;
  if (e.code === 'KeyL') labelGroup.visible = !labelGroup.visible;
  if (e.code === 'KeyG' && grid) grid.visible = !grid.visible;
  if (e.code === 'KeyR') respawn();
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
    const r = resolveCircle(player.x + (player.vx * dt) / steps, player.z + (player.vz * dt) / steps, CFG.radius, colliders);
    r.hits.forEach((h) => hits.add(h));
    player.x = r.x; player.z = r.z;
  }
  // on contact, keep only the velocity that actually happened (slide along walls, no stored push)
  if (hits.size && dt > 0) {
    player.vx = (player.x - x0) / dt;
    player.vz = (player.z - z0) / dt;
  }
  player.hits = [...hits];
}

const raycaster = new THREE.Raycaster();
function updateCamera() {
  const head = new THREE.Vector3(player.x, CFG.eyeHeight, player.z);
  camera.rotation.set(player.pitch, player.yaw, 0);
  if (!thirdPerson) {
    camera.position.copy(head);
  } else {
    avatar.position.set(player.x, CFG.eyeHeight / 2, player.z);
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

// ---------- HUD ----------
const hud = document.getElementById('hud');
let fpsAcc = 0, fpsFrames = 0, fps = 0;
function updateHud(dt) {
  fpsAcc += dt; fpsFrames++;
  if (fpsAcc > 0.5) { fps = fpsFrames / fpsAcc; fpsAcc = fpsFrames = 0; }
  const deg = ((THREE.MathUtils.radToDeg(player.yaw) % 360) + 360) % 360;
  hud.textContent =
    `scene  ${sceneDef?.name ?? '-'}\n` +
    `pos    ${player.x.toFixed(2)}, ${player.z.toFixed(2)}\n` +
    `yaw    ${deg.toFixed(0)}°   speed ${Math.hypot(player.vx, player.vz).toFixed(2)} m/s\n` +
    `view   ${thirdPerson ? `3rd (${thirdDist.toFixed(1)} m)` : `1st (fov ${camera.fov.toFixed(0)})`}\n` +
    `touch  ${player.hits.join(' ') || '-'}\n` +
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

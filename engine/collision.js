// 2D collision on the floor plane (x, z). No three.js dependency so it can be
// unit-tested in node.
//
// Every collider is an oriented rectangle: { cx, cz, hx, hz, angle, id }
//   (cx, cz)  center on the floor
//   hx, hz    half-extents along the box's local x / z axes
//   angle     radians; local x axis points along (cos a, sin a) in world (x, z).
//             A three.js mesh with the same footprint uses rotation.y = -angle.

export function wallCollider(wall, id) {
  const [x0, z0] = wall.from;
  const [x1, z1] = wall.to;
  const t = wall.thickness ?? 0.15;
  const dx = x1 - x0, dz = z1 - z0;
  const len = Math.hypot(dx, dz);
  // Extend by half the thickness on each end so corners close up.
  return {
    id,
    cx: (x0 + x1) / 2,
    cz: (z0 + z1) / 2,
    hx: len / 2 + t / 2,
    hz: t / 2,
    angle: Math.atan2(dz, dx),
  };
}

// obj.size = [w, h, d] meters, obj.position = [x, y, z], obj.rotation_y = degrees
// (three.js yaw convention).
export function objectCollider(obj, id) {
  const [w, , d] = obj.size;
  return {
    id,
    cx: obj.position[0],
    cz: obj.position[2],
    hx: w / 2,
    hz: d / 2,
    angle: -((obj.rotation_y ?? 0) * Math.PI) / 180,
  };
}

export function buildColliders(sceneDef) {
  const out = [];
  (sceneDef.schematic?.walls ?? []).forEach((w, i) => out.push(wallCollider(w, `wall:${i}`)));
  (sceneDef.objects ?? []).forEach((o) => {
    if (o.collide === false) return;
    out.push(objectCollider(o, `obj:${o.id}`));
  });
  return out;
}

// Push a circle (player footprint) out of all colliders. A few passes settle corners.
export function resolveCircle(px, pz, r, colliders, passes = 3) {
  const hits = new Set();
  for (let p = 0; p < passes; p++) {
    let moved = false;
    for (const c of colliders) {
      const cos = Math.cos(c.angle), sin = Math.sin(c.angle);
      const dx = px - c.cx, dz = pz - c.cz;
      // world -> local
      const lx = dx * cos + dz * sin;
      const lz = -dx * sin + dz * cos;
      const qx = Math.max(-c.hx, Math.min(c.hx, lx));
      const qz = Math.max(-c.hz, Math.min(c.hz, lz));
      let ox = lx - qx, oz = lz - qz;
      const dist = Math.hypot(ox, oz);
      if (dist >= r) continue;

      let nx, nz; // new local position
      if (dist > 1e-9) {
        // outside the box but within radius: push along the normal
        const k = r / dist;
        nx = qx + ox * k;
        nz = qz + oz * k;
      } else {
        // center inside the box: exit along the axis with least penetration
        const penX = c.hx - Math.abs(lx);
        const penZ = c.hz - Math.abs(lz);
        if (penX < penZ) {
          nx = Math.sign(lx || 1) * (c.hx + r);
          nz = lz;
        } else {
          nx = lx;
          nz = Math.sign(lz || 1) * (c.hz + r);
        }
      }
      // local -> world
      px = c.cx + nx * cos - nz * sin;
      pz = c.cz + nx * sin + nz * cos;
      hits.add(c.id);
      moved = true;
    }
    if (!moved) break;
  }
  return { x: px, z: pz, hits: [...hits] };
}

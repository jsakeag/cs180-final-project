// run: node engine/tests/collision.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { wallCollider, objectCollider, buildColliders, resolveCircle } from '../collision.js';

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const r = 0.3;

// wall along x at z=0: a player at z=0.1 gets pushed to z = 0.075 + 0.3
{
  const c = [wallCollider({ from: [0, 0], to: [6, 0], thickness: 0.15 }, 'w')];
  const out = resolveCircle(3, 0.1, r, c);
  near(out.x, 3); near(out.z, 0.075 + r); assert.deepEqual(out.hits, ['w']);
}
// same wall, player far away: untouched
{
  const c = [wallCollider({ from: [0, 0], to: [6, 0] }, 'w')];
  const out = resolveCircle(3, 2, r, c);
  near(out.z, 2); assert.equal(out.hits.length, 0);
}
// vertical wall (x=6): player at x=5.8 pushed to x = 6 - 0.075 - 0.3
{
  const c = [wallCollider({ from: [6, 0], to: [6, 5], thickness: 0.15 }, 'w')];
  const out = resolveCircle(5.8, 2.5, r, c);
  near(out.x, 6 - 0.075 - r); near(out.z, 2.5);
}
// rotated object (90 deg): a 0.9 x 0.35 footprint becomes 0.35 wide in x
{
  const o = { id: 'b', position: [5, 0, 4], size: [0.9, 1.8, 0.35], rotation_y: 90 };
  const c = [objectCollider(o, 'b')];
  const out = resolveCircle(5 - 0.175 - 0.1, 4, r, c); // 0.1 m from its -x face
  near(out.x, 5 - 0.175 - r, 1e-6);
}
// center inside a box exits by the shallow side
{
  const c = [objectCollider({ position: [0, 0, 0], size: [2, 1, 0.4] }, 'o')];
  const out = resolveCircle(0.5, 0.15, r, c);
  near(out.x, 0.5); near(out.z, 0.2 + r);
}
// demo scene: doorway between rooms is passable, spawn is free
{
  const def = JSON.parse(readFileSync(new URL('../scenes/demo-room.json', import.meta.url)));
  const cs = buildColliders(def);
  const door = resolveCircle(6, 2.5, r, cs);
  near(door.x, 6); near(door.z, 2.5);
  const sp = def.spawn.position;
  assert.equal(resolveCircle(sp[0], sp[2], r, cs).hits.length, 0, 'spawn overlaps something');
}
console.log('collision tests passed');

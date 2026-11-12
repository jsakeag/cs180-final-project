# Movement prototype + scene format

Three.js viewer that loads a **scene JSON** (room schematic + placed objects) and lets you walk around it.

```bash
cd engine && python3 -m http.server 8000
# open http://localhost:8000/?scene=scenes/demo-room.json
node tests/collision.test.mjs   # collision unit tests
```

Controls: click to play · WASD · Space jump (you can land on objects) · Shift sprint · mouse look · scroll zoom (FOV in 1st person, distance in 3rd) · V 1st/3rd person · P jump to each source photo's camera · C collider outlines (green = touching) · L labels · G 1 m grid · R respawn.
Tune movement live in the console via `dbg.CFG` (speed, radius, accel, sensitivity, jumpSpeed, gravity).

## Scene JSON (the contract between detection and the engine)

Units are meters. Coordinates: x/z = floor plane, y = up. `rotation_y` is in degrees (three.js yaw).

```jsonc
{
  "name": "demo-room",
  "spawn": { "position": [x, 0, z], "yaw_deg": -90 },     // -90 faces +x
  "schematic": {                                         // hand-built beforehand
    "floor": { "min": [x0, z0], "max": [x1, z1] },
    "wall_height": 2.7,
    "walls": [ { "from": [x, z], "to": [x, z], "thickness": 0.15 } ]  // doors = gaps between segments
  },
  "objects": [                                           // written by the detection pipeline
    {
      "id": "chair_0", "class": "chair",
      "position": [x, y, z],        // footprint center; y = bottom of object (0 = on floor)
      "size": [w, h, d],            // bounding box before rotation
      "rotation_y": 15,
      "collide": true,              // optional, default true (rugs etc. = false)
      "model": "models/chair.glb",  // optional; null = colored placeholder box
      "source": { "image": "img_012.jpg", "confidence": 0.87 }
    }
  ]
}
```

A `.glb` model is uniformly scaled to fit inside `size` and placed with its bottom on the floor. Collision uses the `size` footprint, so physics stays the same whether or not a model loads.

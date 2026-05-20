/**
 * PlayCanvas minimal example: decode an Catetus GLB with the polyfill
 * and feed it into a PlayCanvas `Mesh` + `MeshInstance` rendered with the
 * point-cloud primitive. PlayCanvas has a first-party Gaussian-splat
 * renderer (`GSplatComponent` / `GSplatInstance`) but its data contract
 * is internal; using a point-cloud mesh here keeps the example
 * dependency-free and lets you swap in the splat component once you know
 * your PlayCanvas version's API.
 *
 * Run from this directory:
 *   npx esbuild main.ts --bundle --format=esm --external:playcanvas --outfile=main.js
 *   python -m http.server 8080
 */
import * as pc from 'playcanvas';
import {
  decodeSFExtensions,
  decodeV5TailBytes,
  applyV5TailToScene,
} from '@catetus/glb-polyfill';

const SCENE_URL = './scene.glb';
const SHPAL_URL = './scene.glb.shpal';
const V5TAIL_URL = './scene.glb.v5tail';
const SH_C0 = 0.28209479177387814;

function splitGlb(bytes: Uint8Array): { json: unknown; bin: Uint8Array } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB');
  const total = dv.getUint32(8, true);
  let off = 12;
  let json: unknown = null;
  let bin = new Uint8Array(0);
  while (off + 8 <= total) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    const slice = bytes.subarray(off + 8, off + 8 + len);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(slice));
    else if (type === 0x004e4942) bin = slice;
    off += 8 + len;
  }
  if (!json) throw new Error('GLB missing JSON chunk');
  return { json, bin };
}

async function fetchBytes(url: string): Promise<Uint8Array | null> {
  try { const r = await fetch(url); if (!r.ok) return null; return new Uint8Array(await r.arrayBuffer()); }
  catch { return null; }
}

async function decode() {
  const glbBytes = await fetchBytes(SCENE_URL);
  if (!glbBytes) throw new Error(`missing ${SCENE_URL}`);
  const { json, bin } = splitGlb(glbBytes);
  const palBytes = await fetchBytes(SHPAL_URL);
  const sidecars = palBytes ? { 'scene.glb.shpal': palBytes } : undefined;

  const scene = decodeSFExtensions(json, bin, sidecars);
  // Polyfill always returns LINEAR scales + LINEAR opacities.
  const scales = scene.scales;
  const opacities = scene.opacities;

  const tailBytes = await fetchBytes(V5TAIL_URL);
  if (tailBytes) {
    const tail = decodeV5TailBytes(tailBytes);
    applyV5TailToScene(
      {
        positions: scene.positions, rotations: scene.rotations,
        scales, opacities,
        dcRaw: scene.dcRaw, shRest: scene.sh_rest,
        shRestCoefs: scene.sh_rest ? scene.sh_rest.length / scene.count / 3 : 0,
      },
      tail,
    );
  }
  return { count: scene.count, positions: scene.positions, dc: scene.dcRaw, opacity: opacities, bbox: scene.bbox };
}

async function main() {
  const hud = document.getElementById('hud')!;
  const canvas = document.getElementById('app') as HTMLCanvasElement;
  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: { antialias: true },
    mouse: new pc.Mouse(canvas),
  });
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
  app.scene.ambientLight = new pc.Color(0.05, 0.05, 0.05);
  app.start();

  const splats = await decode();
  hud.textContent = `${splats.count.toLocaleString()} splats`;

  // Build a PlayCanvas Mesh of unconnected points.
  const colors = new Uint8Array(splats.count * 4);
  for (let i = 0; i < splats.count; i++) {
    colors[i * 4 + 0] = Math.round(Math.min(1, Math.max(0, SH_C0 * splats.dc[i * 3 + 0] + 0.5)) * 255);
    colors[i * 4 + 1] = Math.round(Math.min(1, Math.max(0, SH_C0 * splats.dc[i * 3 + 1] + 0.5)) * 255);
    colors[i * 4 + 2] = Math.round(Math.min(1, Math.max(0, SH_C0 * splats.dc[i * 3 + 2] + 0.5)) * 255);
    colors[i * 4 + 3] = Math.round(splats.opacity[i] * 255);
  }

  const mesh = new pc.Mesh(app.graphicsDevice);
  mesh.setPositions(splats.positions);
  mesh.setColors32(colors);
  mesh.update(pc.PRIMITIVE_POINTS);

  const mat = new pc.StandardMaterial();
  mat.useLighting = false;
  mat.diffuseVertexColor = true;
  mat.update();

  const meshInst = new pc.MeshInstance(mesh, mat);
  const ent = new pc.Entity('splats');
  ent.addComponent('render', { meshInstances: [meshInst] });
  app.root.addChild(ent);

  // Orbit camera.
  const camera = new pc.Entity('camera');
  camera.addComponent('camera', { clearColor: new pc.Color(0.07, 0.07, 0.08) });
  app.root.addChild(camera);

  const center = splats.bbox
    ? new pc.Vec3(
        (splats.bbox.min[0] + splats.bbox.max[0]) / 2,
        (splats.bbox.min[1] + splats.bbox.max[1]) / 2,
        (splats.bbox.min[2] + splats.bbox.max[2]) / 2,
      )
    : new pc.Vec3();
  const r = splats.bbox
    ? Math.hypot(
        splats.bbox.max[0] - splats.bbox.min[0],
        splats.bbox.max[1] - splats.bbox.min[1],
        splats.bbox.max[2] - splats.bbox.min[2],
      ) * 0.9
    : 3;

  let t = 0;
  app.on('update', (dt) => {
    t += dt * 0.3;
    camera.setPosition(center.x + Math.cos(t) * r, center.y + r * 0.4, center.z + Math.sin(t) * r);
    camera.lookAt(center);
  });
}
main().catch((e) => { document.getElementById('hud')!.textContent = `error: ${e.message}`; console.error(e); });

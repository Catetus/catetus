/**
 * Three.js minimal example: decode an SF GLB with the polyfill and render
 * it as a `THREE.Points` cloud. Three.js does not ship a Gaussian splat
 * renderer in core (only in userland forks), so this example demonstrates
 * the data-prep half of the integration — once you have `BufferGeometry`
 * with `position` + `color` + a `splatScale` / `splatRot` / `splatOpacity`
 * attribute, swapping `THREE.Points` for any of the community splat
 * shaders (e.g. `@mkkellogg/gaussian-splats-3d`) is a 5-line change.
 *
 * Run from this directory:
 *   npx esbuild main.ts --bundle --format=esm --external:three --outfile=main.js
 *   python -m http.server 8080
 */
import * as THREE from 'three';
import {
  decodeGlb,
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

async function load(): Promise<THREE.BufferGeometry> {
  const glbBytes = await fetchBytes(SCENE_URL);
  if (!glbBytes) throw new Error(`missing ${SCENE_URL}`);

  // Two equivalent paths:
  //  1) `decodeGlb(bytes)` — one-shot wrapper, no sidecars. Easiest when
  //      the GLB has no palette extension.
  //  2) `decodeSFExtensions(json, bin, sidecars)` — full control + sidecar map.
  // We use #2 here because this example may also load a `.shpal`. When
  // there is no sidecar, `const scene = decodeGlb(glbBytes);` is equivalent.
  const palBytes = await fetchBytes(SHPAL_URL);
  const { json, bin } = splitGlb(glbBytes);
  const sidecars = palBytes ? { 'scene.glb.shpal': palBytes } : undefined;
  const scene = palBytes
    ? decodeSFExtensions(json, bin, sidecars)
    : decodeGlb(glbBytes);

  // Polyfill returns LINEAR scales + LINEAR opacities (eagerly de-logs /
  // de-logits SF_log_quant_attrs values). No conditional needed.
  const scales = scene.scales;
  const opacities = scene.opacities;

  const tailBytes = await fetchBytes(V5TAIL_URL);
  if (tailBytes) {
    const tail = decodeV5TailBytes(tailBytes);
    applyV5TailToScene(
      {
        positions: scene.positions,
        rotations: scene.rotations,
        scales,
        opacities,
        dcRaw: scene.dcRaw,
        shRest: scene.sh_rest,
        shRestCoefs: scene.sh_rest ? scene.sh_rest.length / scene.count / 3 : 0,
      },
      tail,
    );
  }

  // Bake DC → sRGB for the Points preview.
  const colors = new Float32Array(scene.count * 3);
  for (let i = 0; i < scene.count * 3; i++) {
    colors[i] = Math.min(1, Math.max(0, SH_C0 * scene.dcRaw[i] + 0.5));
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(scene.positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  // Custom per-splat attributes a splat-renderer fork can consume directly.
  geom.setAttribute('splatScale', new THREE.BufferAttribute(scales, 3));
  geom.setAttribute('splatRotation', new THREE.BufferAttribute(scene.rotations, 4));
  geom.setAttribute('splatOpacity', new THREE.BufferAttribute(opacities, 1));
  if (scene.bbox) {
    const b = new THREE.Box3(new THREE.Vector3(...scene.bbox.min), new THREE.Vector3(...scene.bbox.max));
    geom.boundingBox = b;
    geom.boundingSphere = b.getBoundingSphere(new THREE.Sphere());
  } else {
    geom.computeBoundingSphere();
  }
  return geom;
}

async function main() {
  const hud = document.getElementById('hud')!;
  const geom = await load();
  hud.textContent = `${geom.attributes.position.count.toLocaleString()} splats`;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(innerWidth, innerHeight);
  renderer.setClearColor(0x111114);
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const mat = new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: true });
  scene.add(new THREE.Points(geom, mat));

  const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.05, 500);
  const target = geom.boundingSphere!.center.clone();
  const r = geom.boundingSphere!.radius * 1.6;

  addEventListener('resize', () => {
    renderer.setSize(innerWidth, innerHeight);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
  });

  renderer.setAnimationLoop((t) => {
    const a = t * 0.0003;
    camera.position.set(target.x + Math.cos(a)*r, target.y + r*0.4, target.z + Math.sin(a)*r);
    camera.lookAt(target);
    renderer.render(scene, camera);
  });
}
main().catch((e) => { document.getElementById('hud')!.textContent = `error: ${e.message}`; console.error(e); });

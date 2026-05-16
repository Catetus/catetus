import { PROJECT_GATHER_WGSL } from '../../packages/viewer/dist/webgpu/shaders.generated.js';
import { templateSplatsAccess } from '../../packages/viewer/dist/webgpu/buffer-pager.js';

const dilSrc = PROJECT_GATHER_WGSL.replace(/let reg = 0\.3; \/\/ SF_EWA_DILATION/g, 'let reg = 0.300000; // SF_EWA_DILATION(override=0.3)');

const splitMarker = '// cs_project_gather — full projection';
const markerIdx = dilSrc.indexOf(splitMarker);
const firstBindingIdx = dilSrc.indexOf('@group(0) @binding(0)');
const preamble = dilSrc.slice(0, firstBindingIdx);
const keygenSrc = preamble + dilSrc.slice(firstBindingIdx, markerIdx);
const gatherSrc = preamble + dilSrc.slice(markerIdx);

const NUM_PAGES = 2;
const SPP = 33554176;

const tplK = templateSplatsAccess(keygenSrc, 'k_splats', NUM_PAGES, SPP);
const tplG = templateSplatsAccess(gatherSrc, 'g_splats', NUM_PAGES, SPP);

console.log('=== KEYGEN bindings ===');
console.log(tplK.wgsl.split('\n').filter(l => l.includes('@group') || l.includes('@compute')).join('\n'));
console.log('\n=== GATHER bindings ===');
console.log(tplG.wgsl.split('\n').filter(l => l.includes('@group') || l.includes('@compute')).join('\n'));

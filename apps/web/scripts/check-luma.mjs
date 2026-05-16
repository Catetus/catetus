// Compute mean luma and pixel variance for the latest yaw-0 capture.
// Thresholds from CLAUDE.md plan: luma > 0.05 AND variance > 0.005.
import { readFileSync } from 'fs';
import { PNG } from 'pngjs';

const path = process.argv[2] || '/Users/montabano1/Desktop/.wt-hero-v2/tasks/hero-v2-proof/yaw-0.png';
const png = PNG.sync.read(readFileSync(path));
const { data, width, height } = png;
let sumY = 0, sumY2 = 0;
const n = width * height;
for (let i = 0; i < data.length; i += 4) {
  const r = data[i] / 255, g = data[i+1] / 255, b = data[i+2] / 255;
  // Rec. 709 luma
  const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  sumY += Y;
  sumY2 += Y * Y;
}
const meanY = sumY / n;
const varY = sumY2 / n - meanY * meanY;
console.log(JSON.stringify({ path, width, height, meanLuma: +meanY.toFixed(5), variance: +varY.toFixed(5) }));

// CodecGS-Lite GLB decoder demo.
//
// Pipeline:
//   1. Fetch a .glb produced by experiments/w3-codecgs-integrate/code/pack_codecgs_glb.py
//   2. Parse GLB header + JSON chunk + BIN chunk.
//   3. For each bufferView referenced from SF_codecgs_lite.channels, copy the
//      mp4 bytes and run them through mp4box.js + WebCodecs.VideoDecoder.
//   4. For one demo channel (opacity), gather the decoded VideoFrame into a
//      Uint8Array and render it to a canvas at grid resolution so the user
//      can visually confirm the decode worked.
//   5. The dequantize math (mins/maxs from the extension) is shown but not
//      executed against splats here — the focus is "does the decode work?"
//
// This file is intentionally framework-free so it loads cleanly from a static
// /public path without any bundler step.

const GLB_MAGIC = 0x46546c67; // "glTF"
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

const fileInput = document.getElementById('file');
const loadDefaultBtn = document.getElementById('loadDefault');
const extMetaEl = document.getElementById('ext-meta');
const channelsTbody = document.querySelector('#channels-table tbody');
const summaryEl = document.getElementById('summary');
const opacityCanvas = document.getElementById('opacity-canvas');
const reconMetaEl = document.getElementById('recon-meta');

fileInput.addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const buf = new Uint8Array(await f.arrayBuffer());
  await runDecode(buf, f.name);
});

loadDefaultBtn.addEventListener('click', async () => {
  loadDefaultBtn.disabled = true;
  try {
    const r = await fetch('codecgs-bonsai.glb');
    if (!r.ok) throw new Error(`fetch codecgs-bonsai.glb: ${r.status}`);
    const buf = new Uint8Array(await r.arrayBuffer());
    await runDecode(buf, 'codecgs-bonsai.glb');
  } catch (err) {
    extMetaEl.textContent = 'load failed: ' + err.message;
    extMetaEl.className = 'err';
  } finally {
    loadDefaultBtn.disabled = false;
  }
});

function parseGlb(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== GLB_MAGIC) throw new Error('bad GLB magic');
  if (dv.getUint32(4, true) !== 2) throw new Error('unsupported GLB version');
  const total = dv.getUint32(8, true);
  let offset = 12;
  let jsonBytes = null;
  let binBytes = null;
  while (offset + 8 <= total) {
    const chunkLen = dv.getUint32(offset, true);
    const chunkType = dv.getUint32(offset + 4, true);
    const dataStart = offset + 8;
    const dataEnd = dataStart + chunkLen;
    if (chunkType === CHUNK_JSON) {
      jsonBytes = bytes.subarray(dataStart, dataEnd);
    } else if (chunkType === CHUNK_BIN) {
      binBytes = bytes.subarray(dataStart, dataEnd);
    }
    offset = dataEnd;
  }
  if (!jsonBytes) throw new Error('missing JSON chunk');
  if (!binBytes) throw new Error('missing BIN chunk');
  const jsonStr = new TextDecoder().decode(jsonBytes).replace(/\s+$/, '');
  const root = JSON.parse(jsonStr);
  return { root, bin: binBytes };
}

function bufferViewBytes(root, bin, bvIndex) {
  const bv = root.bufferViews[bvIndex];
  const off = bv.byteOffset || 0;
  return bin.subarray(off, off + bv.byteLength);
}

async function decodeMp4Channel(mp4Bytes, label) {
  // Adapted from experiments/w2-codecgs/code/webcodecs_bench.html.
  return new Promise((resolve, reject) => {
    const mp4 = MP4Box.createFile();
    const frames = [];
    let codecStr = null;
    let widthH = 0, heightH = 0;
    const t0 = performance.now();
    let firstChunkPt = null;
    const decoder = new VideoDecoder({
      output: (f) => {
        if (firstChunkPt === null) firstChunkPt = performance.now();
        frames.push(f);  // keep VideoFrame open; caller closes after pixel readback
      },
      error: (e) => reject(e),
    });
    mp4.onError = (e) => reject(new Error('mp4box: ' + e));
    mp4.onReady = (info) => {
      const v = info.videoTracks[0];
      if (!v) return reject(new Error('no video track'));
      codecStr = v.codec;
      widthH = v.video.width;
      heightH = v.video.height;
      let description = null;
      const trak = mp4.getTrackById(v.id);
      const entries = trak.mdia.minf.stbl.stsd.entries;
      for (const e of entries) {
        const box = e.avcC || e.hvcC || e.vpcC || e.av1C;
        if (box) {
          const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
          box.write(stream);
          description = new Uint8Array(stream.buffer, 8);
          break;
        }
      }
      decoder.configure({
        codec: codecStr,
        codedWidth: widthH,
        codedHeight: heightH,
        description,
      });
      mp4.setExtractionOptions(v.id, null, { nbSamples: v.nb_samples });
      mp4.onSamples = (id, user, samples) => {
        for (const s of samples) {
          const chunk = new EncodedVideoChunk({
            type: s.is_sync ? 'key' : 'delta',
            timestamp: (s.cts * 1e6) / s.timescale,
            duration: (s.duration * 1e6) / s.timescale,
            data: s.data,
          });
          decoder.decode(chunk);
        }
        decoder.flush()
          .then(() => {
            resolve({
              codec: codecStr,
              width: widthH,
              height: heightH,
              frames,
              decode_ms: performance.now() - t0,
              label,
            });
          })
          .catch(reject);
      };
      mp4.start();
    };
    const buf = mp4Bytes.buffer.slice(mp4Bytes.byteOffset,
                                     mp4Bytes.byteOffset + mp4Bytes.byteLength);
    buf.fileStart = 0;
    mp4.appendBuffer(buf);
    mp4.flush();
  });
}

async function videoFrameToCanvas(frame, canvas) {
  // Resize canvas to match the frame, copy via drawImage.
  canvas.width = frame.codedWidth;
  canvas.height = frame.codedHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(frame, 0, 0);
}

function dequantizeMeta(ch) {
  // Compute a representative scale/bias pair from mins/maxs so a future viewer
  // pass can run: attr[i] = pixel[i] / 255 * (max-min) + min.
  // We only display it here.
  return {
    mins: ch.mins,
    maxs: ch.maxs,
    deqScale: ch.mins.map((mn, i) => (ch.maxs[i] - mn) / 255),
    deqBias: ch.mins.map((mn) => mn),
  };
}

async function runDecode(bytes, fname) {
  channelsTbody.innerHTML = '';
  extMetaEl.textContent = '';
  summaryEl.textContent = 'parsing GLB...';

  let glb;
  try {
    glb = parseGlb(bytes);
  } catch (e) {
    extMetaEl.textContent = 'GLB parse failed: ' + e.message;
    extMetaEl.className = 'err';
    return;
  }

  const root = glb.root;
  const bin = glb.bin;
  const ext = root.extensions && root.extensions.SF_codecgs_lite;
  if (!ext) {
    extMetaEl.textContent = 'No SF_codecgs_lite extension on this GLB.';
    extMetaEl.className = 'err';
    return;
  }

  const required = root.extensionsRequired || [];
  if (!required.includes('SF_codecgs_lite')) {
    extMetaEl.textContent = 'SF_codecgs_lite is not in extensionsRequired (writer bug?).';
    extMetaEl.className = 'err';
    return;
  }

  extMetaEl.className = '';
  extMetaEl.textContent = JSON.stringify(
    {
      version: ext.version,
      codec: ext.codec,
      codecString: ext.codecString,
      nSplats: ext.nSplats,
      nOrigSplats: ext.nOrigSplats,
      nSide: ext.nSide,
      shDegree: ext.shDegree,
      fRestDim: ext.fRestDim,
      sort: ext.sort,
      xyzLogTransform: ext.xyzLogTransform,
      channels: ext.channels.map((c) => ({
        name: c.name, sub: c.subChannels, frames: c.nFrames,
        crf: c.crf, bytes: c.bytes,
      })),
    },
    null,
    2,
  );

  const results = [];
  let opacityFrame = null;
  const tStart = performance.now();
  for (const ch of ext.channels) {
    const row = document.createElement('tr');
    row.innerHTML = `<td class="mono">${ch.name}</td>
                     <td class="mono">${ext.codecString}</td>
                     <td class="mono">${ch.bytes.toLocaleString()}</td>
                     <td class="mono">…</td>
                     <td class="mono">${ch.nFrames}</td>
                     <td class="mono">…</td>
                     <td>decoding…</td>`;
    channelsTbody.appendChild(row);
    const mp4Bytes = bufferViewBytes(root, bin, ch.bufferView);
    try {
      const r = await decodeMp4Channel(mp4Bytes, ch.name);
      results.push({ ch, r });
      const tds = row.querySelectorAll('td');
      tds[3].textContent = `${r.width}×${r.height}`;
      tds[5].textContent = r.decode_ms.toFixed(1);
      tds[6].innerHTML = `<span class="ok">ok (${r.frames.length} frames)</span>`;
      if (ch.name === 'opacity' && r.frames.length > 0) {
        opacityFrame = r.frames[0];
      } else {
        // close every frame we won't render to keep memory bounded.
        for (const f of r.frames) f.close();
      }
    } catch (e) {
      const tds = row.querySelectorAll('td');
      tds[6].innerHTML = `<span class="err">err: ${e.message}</span>`;
      results.push({ ch, error: e.message });
    }
  }
  const tEnd = performance.now();

  if (opacityFrame) {
    await videoFrameToCanvas(opacityFrame, opacityCanvas);
    const dq = dequantizeMeta(ext.channels.find((c) => c.name === 'opacity'));
    reconMetaEl.textContent =
      `opacity grid: ${opacityFrame.codedWidth}×${opacityFrame.codedHeight}; ` +
      `dequantize: opacity = (pixel/255) * (${dq.maxs[0].toFixed(3)} - ` +
      `${dq.mins[0].toFixed(3)}) + ${dq.mins[0].toFixed(3)}`;
    opacityFrame.close();
  }

  const okCount = results.filter((r) => r.r).length;
  const totalBytes = ext.channels.reduce((a, c) => a + c.bytes, 0);
  summaryEl.innerHTML =
    `<strong>${fname}</strong> — ` +
    `${okCount}/${ext.channels.length} channels decoded; ` +
    `total bundle bytes (sum of channel mp4s) = <strong>${totalBytes.toLocaleString()}</strong>; ` +
    `whole-file decode wall time = <strong>${(tEnd - tStart).toFixed(1)} ms</strong>.`;

  // Expose to Playwright.
  window.__codecgsResult = {
    fname,
    nSplats: ext.nSplats,
    nSide: ext.nSide,
    okCount,
    nChannels: ext.channels.length,
    totalBytes,
    decodeMs: tEnd - tStart,
    perChannel: results.map((r) => ({
      name: r.ch.name,
      ok: !!r.r,
      error: r.error || null,
      width: r.r ? r.r.width : null,
      height: r.r ? r.r.height : null,
      frames: r.r ? r.r.frames.length : 0,
      decodeMs: r.r ? r.r.decode_ms : null,
    })),
  };
}

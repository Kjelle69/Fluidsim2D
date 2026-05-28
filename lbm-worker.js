let params = null;
let running = true;
let isBatchRunning = false;
let lastFramePost = 0;
let simTime = 0;

let n0, nN, nS, nE, nW, nNE, nSE, nNW, nSW;
let rho, ux, uy, curl, barrier;
let tracers = [];
let pulseSources = [];
let force = { x: 0, y: 0, fx: 0, fy: 0, count: 0 };

const TRACER_COUNT = 260;
const MIN_RHO = 0.65;
const MAX_RHO = 1.35;
const MAX_LBM_SPEED = 0.18;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const idx = (x, y) => x + y * params.nx;

function flowSpeed() {
  return clamp(params.inletVelocity / 420, 0, MAX_LBM_SPEED);
}

function lbmViscosity() {
  return clamp(params.viscosity * 2800, 0.005, 0.2);
}

function createArrays() {
  const size = params.nx * params.ny;
  n0 = new Float32Array(size);
  nN = new Float32Array(size);
  nS = new Float32Array(size);
  nE = new Float32Array(size);
  nW = new Float32Array(size);
  nNE = new Float32Array(size);
  nSE = new Float32Array(size);
  nNW = new Float32Array(size);
  nSW = new Float32Array(size);
  rho = new Float32Array(size);
  ux = new Float32Array(size);
  uy = new Float32Array(size);
  curl = new Float32Array(size);
  barrier = new Uint8Array(size);
}

function setEquil(x, y, newUx, newUy, newRho = 1) {
  const i = idx(x, y);
  newUx = clamp(Number.isFinite(newUx) ? newUx : 0, -MAX_LBM_SPEED, MAX_LBM_SPEED);
  newUy = clamp(Number.isFinite(newUy) ? newUy : 0, -MAX_LBM_SPEED, MAX_LBM_SPEED);
  newRho = clamp(Number.isFinite(newRho) ? newRho : 1, MIN_RHO, MAX_RHO);

  const ux3 = 3 * newUx;
  const uy3 = 3 * newUy;
  const ux2 = newUx * newUx;
  const uy2 = newUy * newUy;
  const uxuy2 = 2 * newUx * newUy;
  const u2 = ux2 + uy2;
  const u215 = 1.5 * u2;
  const four9ths = 4 / 9;
  const one9th = 1 / 9;
  const one36th = 1 / 36;

  n0[i] = four9ths * newRho * (1 - u215);
  nE[i] = one9th * newRho * (1 + ux3 + 4.5 * ux2 - u215);
  nW[i] = one9th * newRho * (1 - ux3 + 4.5 * ux2 - u215);
  nN[i] = one9th * newRho * (1 - uy3 + 4.5 * uy2 - u215);
  nS[i] = one9th * newRho * (1 + uy3 + 4.5 * uy2 - u215);
  nNE[i] = one36th * newRho * (1 + ux3 - uy3 + 4.5 * (u2 - uxuy2) - u215);
  nSE[i] = one36th * newRho * (1 + ux3 + uy3 + 4.5 * (u2 + uxuy2) - u215);
  nNW[i] = one36th * newRho * (1 - ux3 - uy3 + 4.5 * (u2 + uxuy2) - u215);
  nSW[i] = one36th * newRho * (1 - ux3 + uy3 + 4.5 * (u2 - uxuy2) - u215);
  rho[i] = newRho;
  ux[i] = newUx;
  uy[i] = newUy;
}

function setBoundaries() {
  const u0 = flowSpeed();
  for (let y = 1; y < params.ny - 1; y++) {
    setEquil(0, y, u0, 0, 1);
    setEquil(params.nx - 1, y, u0, 0, 1);
  }

  if (params.domainBoundaryMode === 'tunnel') {
    for (let x = 0; x < params.nx; x++) {
      setEquil(x, 0, u0, 0, 1);
      setEquil(x, params.ny - 1, u0, 0, 1);
    }
  } else {
    for (let x = 0; x < params.nx; x++) {
      const topIn = idx(x, 1);
      const botIn = idx(x, params.ny - 2);
      setEquil(x, 0, ux[topIn], uy[topIn], Math.max(0.1, rho[topIn] || 1));
      setEquil(x, params.ny - 1, ux[botIn], uy[botIn], Math.max(0.1, rho[botIn] || 1));
    }
  }
}

function resetTracers() {
  tracers = Array.from({ length: TRACER_COUNT }, () => ({
    x: Math.random() * (params.nx - 4) + 2,
    y: Math.random() * (params.ny - 4) + 2,
  }));
}

function resetTracerAtInlet(t) {
  t.x = 1 + Math.random() * 3;
  t.y = 2 + Math.random() * (params.ny - 4);
}

function initialize(solid) {
  createArrays();
  pulseSources = [];
  if (solid && solid.length === barrier.length) barrier.set(solid);

  if (params.domainBoundaryMode === 'tunnel') {
    for (let x = 0; x < params.nx; x++) {
      barrier[idx(x, 0)] = 1;
      barrier[idx(x, params.ny - 1)] = 1;
    }
  }

  const u0 = flowSpeed();
  for (let y = 0; y < params.ny; y++) {
    for (let x = 0; x < params.nx; x++) {
      setEquil(x, y, barrier[idx(x, y)] ? 0 : u0, 0, 1);
    }
  }
  resetTracers();
  computeCurl();
}

function collide() {
  const omega = 1 / (3 * lbmViscosity() + 0.5);
  const four9ths = 4 / 9;
  const one9th = 1 / 9;
  const one36th = 1 / 36;

  for (let y = 1; y < params.ny - 1; y++) {
    for (let x = 1; x < params.nx - 1; x++) {
      const i = idx(x, y);
      if (barrier[i]) {
        ux[i] = 0;
        uy[i] = 0;
        continue;
      }

      const r = n0[i] + nN[i] + nS[i] + nE[i] + nW[i] + nNE[i] + nSE[i] + nNW[i] + nSW[i];
      if (r <= MIN_RHO || !Number.isFinite(r)) {
        setEquil(x, y, 0, 0, 1);
        continue;
      }

      const safeRho = clamp(r, MIN_RHO, MAX_RHO);
      const u = clamp((nE[i] + nNE[i] + nSE[i] - nW[i] - nNW[i] - nSW[i]) / r, -MAX_LBM_SPEED, MAX_LBM_SPEED);
      const v = clamp((nS[i] + nSE[i] + nSW[i] - nN[i] - nNE[i] - nNW[i]) / r, -MAX_LBM_SPEED, MAX_LBM_SPEED);
      if (r !== safeRho) {
        setEquil(x, y, u, v, safeRho);
        continue;
      }

      rho[i] = safeRho;
      ux[i] = u;
      uy[i] = v;

      const one9thRho = one9th * safeRho;
      const one36thRho = one36th * safeRho;
      const ux3 = 3 * u;
      const uy3 = 3 * v;
      const ux2 = u * u;
      const uy2 = v * v;
      const uxuy2 = 2 * u * v;
      const u2 = ux2 + uy2;
      const u215 = 1.5 * u2;

      n0[i] += omega * (four9ths * safeRho * (1 - u215) - n0[i]);
      nE[i] += omega * (one9thRho * (1 + ux3 + 4.5 * ux2 - u215) - nE[i]);
      nW[i] += omega * (one9thRho * (1 - ux3 + 4.5 * ux2 - u215) - nW[i]);
      nS[i] += omega * (one9thRho * (1 + uy3 + 4.5 * uy2 - u215) - nS[i]);
      nN[i] += omega * (one9thRho * (1 - uy3 + 4.5 * uy2 - u215) - nN[i]);
      nSE[i] += omega * (one36thRho * (1 + ux3 + uy3 + 4.5 * (u2 + uxuy2) - u215) - nSE[i]);
      nNE[i] += omega * (one36thRho * (1 + ux3 - uy3 + 4.5 * (u2 - uxuy2) - u215) - nNE[i]);
      nSW[i] += omega * (one36thRho * (1 - ux3 + uy3 + 4.5 * (u2 - uxuy2) - u215) - nSW[i]);
      nNW[i] += omega * (one36thRho * (1 - ux3 - uy3 + 4.5 * (u2 + uxuy2) - u215) - nNW[i]);
    }
  }

  for (let y = 1; y < params.ny - 1; y++) {
    const out = idx(params.nx - 1, y);
    const src = idx(params.nx - 2, y);
    nW[out] = nW[src];
    nNW[out] = nNW[src];
    nSW[out] = nSW[src];
  }
}

function stream() {
  force = { x: 0, y: 0, fx: 0, fy: 0, count: 0 };

  for (let y = params.ny - 2; y > 0; y--) {
    for (let x = 1; x < params.nx - 1; x++) {
      nS[idx(x, y)] = nS[idx(x, y - 1)];
      nSW[idx(x, y)] = nSW[idx(x + 1, y - 1)];
    }
  }
  for (let y = params.ny - 2; y > 0; y--) {
    for (let x = params.nx - 2; x > 0; x--) {
      nE[idx(x, y)] = nE[idx(x - 1, y)];
      nSE[idx(x, y)] = nSE[idx(x - 1, y - 1)];
    }
  }
  for (let y = 1; y < params.ny - 1; y++) {
    for (let x = params.nx - 2; x > 0; x--) {
      nN[idx(x, y)] = nN[idx(x, y + 1)];
      nNE[idx(x, y)] = nNE[idx(x - 1, y + 1)];
    }
  }
  for (let y = 1; y < params.ny - 1; y++) {
    for (let x = 1; x < params.nx - 1; x++) {
      nW[idx(x, y)] = nW[idx(x + 1, y)];
      nNW[idx(x, y)] = nNW[idx(x + 1, y + 1)];
    }
  }

  for (let y = 1; y < params.ny - 1; y++) {
    for (let x = 1; x < params.nx - 1; x++) {
      const i = idx(x, y);
      if (!barrier[i]) continue;

      nE[idx(x + 1, y)] = nW[i];
      nW[idx(x - 1, y)] = nE[i];
      nS[idx(x, y + 1)] = nN[i];
      nN[idx(x, y - 1)] = nS[i];
      nSE[idx(x + 1, y + 1)] = nNW[i];
      nSW[idx(x - 1, y + 1)] = nNE[i];
      nNE[idx(x + 1, y - 1)] = nSW[i];
      nNW[idx(x - 1, y - 1)] = nSE[i];

      force.count++;
      force.x += x;
      force.y += y;
      force.fx += nE[i] + nNE[i] + nSE[i] - nW[i] - nNW[i] - nSW[i];
      force.fy += nS[i] + nSE[i] + nSW[i] - nN[i] - nNE[i] - nNW[i];
    }
  }

  if (force.count > 0) {
    force.x /= force.count;
    force.y /= force.count;
  }
}

function computeCurl() {
  curl.fill(0);
  for (let y = 1; y < params.ny - 1; y++) {
    for (let x = 1; x < params.nx - 1; x++) {
      curl[idx(x, y)] = uy[idx(x + 1, y)] - uy[idx(x - 1, y)] - ux[idx(x, y + 1)] + ux[idx(x, y - 1)];
    }
  }
}

function moveTracers() {
  for (const t of tracers) {
    const x = clamp(Math.round(t.x), 0, params.nx - 1);
    const y = clamp(Math.round(t.y), 0, params.ny - 1);
    const i = idx(x, y);
    if (barrier[i]) {
      resetTracerAtInlet(t);
      continue;
    }
    t.x += ux[i];
    t.y += uy[i];
    if (t.x < 1 || t.x > params.nx - 2 || t.y < 1 || t.y > params.ny - 2) resetTracerAtInlet(t);
  }
}

function pushFluid(point, pushU, pushV) {
  const cx = Math.round(point.x);
  const cy = Math.round(point.y);
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const x = cx + dx;
      const y = cy + dy;
      if (x <= 2 || x >= params.nx - 3 || y <= 2 || y >= params.ny - 3 || barrier[idx(x, y)]) continue;
      setEquil(x, y, clamp(pushU / 80, -0.1, 0.1), clamp(pushV / 80, -0.1, 0.1), Math.max(0.1, rho[idx(x, y)] || 1));
    }
  }
}

function pressurePulse(point, radius, strength) {
  pulseSources = [];

  const r = Math.max(2, radius);
  const r2 = r * r;
  const minX = clamp(Math.floor(point.x - r), 1, params.nx - 2);
  const maxX = clamp(Math.ceil(point.x + r), 1, params.nx - 2);
  const minY = clamp(Math.floor(point.y - r), 1, params.ny - 2);
  const maxY = clamp(Math.ceil(point.y + r), 1, params.ny - 2);

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const i = idx(x, y);
      if (barrier[i]) continue;
      const dx = x - point.x;
      const dy = y - point.y;
      const d2 = dx * dx + dy * dy;
      if (d2 > r2) continue;
      const falloff = 0.5 + 0.5 * Math.cos(Math.PI * Math.sqrt(d2) / r);
      setEquil(x, y, ux[i] || 0, uy[i] || 0, (rho[i] || 1) + strength * falloff);
    }
  }
}

function addPressureSource(source) {
  const dx = Number(source.dirX ?? 1);
  const dy = Number(source.dirY ?? 0);
  const len = Math.hypot(dx, dy) || 1;
  const duration = Math.max(1, Math.round(Number(source.duration ?? 1)));
  pulseSources = [{
    x: clamp(Number(source.x ?? params.nx * 0.18), 1, params.nx - 2),
    y: clamp(Number(source.y ?? params.ny * 0.5), 1, params.ny - 2),
    radius: clamp(Number(source.radius ?? 10), 2, Math.max(params.nx, params.ny)),
    strength: clamp(Number(source.strength ?? 0.25), -4, 4),
    duration,
    remaining: duration,
    mode: source.mode === 'directed' ? 'directed' : 'omni',
    dirX: dx / len,
    dirY: dy / len,
  }];
}

function applyPressureSources() {
  if (pulseSources.length === 0) return;

  const next = [];
  for (const source of pulseSources) {
    const r = source.radius;
    const r2 = r * r;
    const age = source.duration - source.remaining;
    const timeWeight = source.duration <= 1
      ? 1
      : 0.5 - 0.5 * Math.cos(2 * Math.PI * age / Math.max(1, source.duration - 1));
    const minX = clamp(Math.floor(source.x - r), 1, params.nx - 2);
    const maxX = clamp(Math.ceil(source.x + r), 1, params.nx - 2);
    const minY = clamp(Math.floor(source.y - r), 1, params.ny - 2);
    const maxY = clamp(Math.ceil(source.y + r), 1, params.ny - 2);

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const i = idx(x, y);
        if (barrier[i]) continue;

        const dx = x - source.x;
        const dy = y - source.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > r2) continue;

        const dist = Math.max(Math.sqrt(d2), 0.001);
        let radial = 0.5 + 0.5 * Math.cos(Math.PI * dist / r);
        let lobe = 1;
        let nudgeX = dx / dist;
        let nudgeY = dy / dist;

        if (source.mode === 'directed') {
          const axial = dx * source.dirX + dy * source.dirY;
          const lateral = Math.abs(dx * source.dirY - dy * source.dirX);
          if (axial < 0) continue;

          const progress = source.duration <= 1
            ? 1
            : age / Math.max(1, source.duration - 1);
          const frontAxial = r * (0.2 + 0.75 * progress);
          const axialWidth = Math.max(1.2, r * 0.14);
          const halfWidth = Math.max(2, r * 0.42);
          const axialDelta = Math.abs(axial - frontAxial);
          if (axialDelta > axialWidth) continue;
          if (lateral > halfWidth) continue;

          const axialFalloff = 0.5 + 0.5 * Math.cos(Math.PI * axialDelta / axialWidth);
          const lateralFalloff = 0.5 + 0.5 * Math.cos(Math.PI * lateral / halfWidth);
          radial = 1;
          lobe = axialFalloff * lateralFalloff;
          nudgeX = source.dirX;
          nudgeY = source.dirY;
        }

        const weight = radial * timeWeight * lobe;
        if (weight <= 0) continue;

        const durationScale = Math.max(1, source.duration * 0.5);
        const directed = source.mode === 'directed';
        const densityKick = source.strength * weight * (directed ? 0.35 : 1) / durationScale;
        const velocityKick = source.strength * (directed ? 0.052 : 0.018) * weight / durationScale;
        setEquil(
          x,
          y,
          (ux[i] || 0) + nudgeX * velocityKick,
          (uy[i] || 0) + nudgeY * velocityKick,
          (rho[i] || 1) + densityKick
        );
      }
    }

    source.remaining -= 1;
    if (source.remaining > 0) next.push(source);
  }

  pulseSources = next;
}

function step() {
  setBoundaries();
  applyPressureSources();
  collide();
  stream();
  moveTracers();
  computeCurl();
  simTime += 1;
}

function colorJet(t) {
  const c = clamp(t, 0, 1);
  if (c < 0.125) return [0, 0, 255 * (c + 0.125) / 0.25];
  if (c < 0.375) return [0, 255 * (c - 0.125) / 0.25, 255];
  if (c < 0.625) {
    const r = 255 * (c - 0.375) / 0.25;
    return [r, 255, 255 - r];
  }
  if (c < 0.875) return [255, 255 * (0.875 - c) / 0.25, 0];
  return [255 * (1.125 - c) / 0.25, 0, 0];
}

function buildPixelFrame() {
  const pixels = new Uint8ClampedArray(params.nx * params.ny * 4);
  const viewMode = params.viewMode ?? 'vorticity';
  const pressureContrast = 70;
  const vorticityContrast = 17.5;

  for (let i = 0; i < barrier.length; i++) {
    const p = i * 4;
    if (barrier[i]) {
      pixels[p] = 0;
      pixels[p + 1] = 0;
      pixels[p + 2] = 0;
      pixels[p + 3] = 255;
      continue;
    }

    let value;
    if (viewMode === 'velocity') {
      value = Math.hypot(ux[i], uy[i]) * 4.0;
    } else if (viewMode === 'pressure') {
      value = (rho[i] - 1) * pressureContrast + 0.5;
    } else {
      value = curl[i] * vorticityContrast + 0.5;
    }
    const rgb = colorJet(value);
    pixels[p] = rgb[0];
    pixels[p + 1] = rgb[1];
    pixels[p + 2] = rgb[2];
    pixels[p + 3] = 255;
  }

  return pixels;
}

function buildVectorSamples() {
  const stride = Math.max(8, Math.round(params.nx / 20));
  const samples = [];
  for (let y = 2; y < params.ny - 2; y += stride) {
    for (let x = 2; x < params.nx - 2; x += stride) {
      const i = idx(x, y);
      if (barrier[i]) continue;
      samples.push(x, y, ux[i] * 30, uy[i] * 30);
    }
  }
  return new Float32Array(samples);
}

function buildFlowlineSegments() {
  if (!params.showFlowlines) return new Float32Array(0);
  const pxPerFlowline = Math.max(8, Math.round(params.nx / 34));
  const segments = [];
  for (let y = pxPerFlowline * 0.5; y < params.ny; y += pxPerFlowline) {
    for (let x = pxPerFlowline * 0.5; x < params.nx; x += pxPerFlowline) {
      const i = idx(Math.round(x), Math.round(y));
      const speed = Math.hypot(ux[i], uy[i]);
      if (barrier[i] || speed < 0.0001) continue;
      const scale = 0.55 * pxPerFlowline / speed;
      segments.push(x - ux[i] * scale, y - uy[i] * scale, x + ux[i] * scale, y + uy[i] * scale);
    }
  }
  return new Float32Array(segments);
}

function postFrame() {
  const pixels = buildPixelFrame();
  const vectorSamples = buildVectorSamples();
  const flowlineSegments = buildFlowlineSegments();
  const pulseData = new Float32Array(pulseSources.length * 7);
  for (let i = 0; i < pulseSources.length; i++) {
    const p = i * 7;
    const source = pulseSources[i];
    pulseData[p + 0] = source.x;
    pulseData[p + 1] = source.y;
    pulseData[p + 2] = source.radius;
    pulseData[p + 3] = source.remaining;
    pulseData[p + 4] = source.duration;
    pulseData[p + 5] = source.mode === 'directed' ? source.dirX : 0;
    pulseData[p + 6] = source.mode === 'directed' ? source.dirY : 0;
  }
  const tracerData = new Float32Array(tracers.length * 3);
  for (let i = 0; i < tracers.length; i++) {
    const p = i * 3;
    tracerData[p] = tracers[i].x;
    tracerData[p + 1] = tracers[i].y;
    tracerData[p + 2] = 0;
  }
  postMessage({
    type: 'frame',
    simTime,
    pixels,
    vectorSamples,
    flowlineSegments,
    tracers: tracerData,
    pulseSources: pulseData,
    force: params.showForce ? force : null,
  }, [pixels.buffer, vectorSamples.buffer, flowlineSegments.buffer, tracerData.buffer, pulseData.buffer]);
}

function tick() {
  if (params && running && !isBatchRunning) {
    const loops = Math.max(1, Math.min(40, params.substepsPerFrame));
    const start = performance.now();
    for (let i = 0; i < loops; i++) {
      step();
      if (i > 1 && performance.now() - start > params.maxFrameMs) break;
    }
    const now = performance.now();
    if (now - lastFramePost > 33) {
      postFrame();
      lastFramePost = now;
    }
  }
  setTimeout(tick, 0);
}

async function runBatch(totalSteps, chunkSize, label) {
  if (isBatchRunning) return;
  isBatchRunning = true;
  const wasRunning = running;
  running = false;
  let done = 0;
  while (done < totalSteps) {
    const n = Math.min(chunkSize, totalSteps - done);
    for (let i = 0; i < n; i++) step();
    done += n;
    postMessage({ type: 'batchProgress', done, totalSteps, label });
    postFrame();
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  isBatchRunning = false;
  running = wasRunning;
  postMessage({ type: 'batchDone', totalSteps, label, running });
}

function configure(nextParams) {
  params = { ...params, ...nextParams };
}

onmessage = event => {
  const msg = event.data;
  if (msg.type === 'init') {
    params = { ...msg.params };
    running = Boolean(msg.running);
    simTime = msg.simTime ?? 0;
    initialize(msg.solid ? new Uint8Array(msg.solid) : null);
    postFrame();
    postMessage({ type: 'ready' });
    return;
  }
  if (!params) return;
  if (msg.type === 'configure') configure(msg.params);
  if (msg.type === 'running') running = Boolean(msg.running);
  if (msg.type === 'reset' || msg.type === 'setSolid') {
    if (msg.params) configure(msg.params);
    initialize(msg.solid ? new Uint8Array(msg.solid) : null);
    postFrame();
  }
  if (msg.type === 'resetTracers') {
    resetTracers();
    postFrame();
  }
  if (msg.type === 'stir') {
    pushFluid(msg.point, msg.pushU, msg.pushV);
    postFrame();
  }
  if (msg.type === 'pressurePulse') {
    pressurePulse(msg.point, msg.radius, msg.strength ?? 0.25);
    postFrame();
  }
  if (msg.type === 'addPressureSource') {
    addPressureSource(msg.source ?? {});
    postFrame();
  }
  if (msg.type === 'batch') runBatch(msg.totalSteps, msg.chunkSize ?? 50, msg.label ?? 'Kör');
};

tick();

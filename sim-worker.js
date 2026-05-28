const PHYSICS = {
  domainWidthM: 8.0,
  domainHeightM: 6.0,
  rho: 1.225,
  characteristicLengthM: 1.0,
};

let params = null;
let simTime = 0;
let running = true;
let isBatchRunning = false;
let lastFramePost = 0;
let fields = null;
let pulseSources = [];

let derived = {
  dx: 0,
  dy: 0,
  invDx: 0,
  invDy: 0,
};

let tracers = [];
const TRACER_COUNT = 600;

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

function idx(i, j) {
  return i + j * params.nx;
}

function updateDerived() {
  derived.dx = PHYSICS.domainWidthM / params.nx;
  derived.dy = PHYSICS.domainHeightM / params.ny;
  derived.invDx = 1 / derived.dx;
  derived.invDy = 1 / derived.dy;
}

function createFields() {
  const size = params.nx * params.ny;
  return {
    u: new Float32Array(size),
    v: new Float32Array(size),
    uPrev: new Float32Array(size),
    vPrev: new Float32Array(size),
    pressure: new Float32Array(size),
    pressurePrev: new Float32Array(size),
    divergence: new Float32Array(size),
    solid: new Uint8Array(size),
    omega: new Float32Array(size),
    omegaMagGradX: new Float32Array(size),
    omegaMagGradY: new Float32Array(size),
    scratchU: new Float32Array(size),
    scratchV: new Float32Array(size),
  };
}

function inletProfile(j) {
  if (params.profileMode === 'flat') return 1.0;
  const y = j / (params.ny - 1);
  return 4 * y * (1 - y);
}

function markDomainWalls() {
  const { solid, u, v } = fields;
  if (params.domainBoundaryMode !== 'tunnel') return;

  for (let i = 0; i < params.nx; i++) {
    const kb = idx(i, 0);
    const kt = idx(i, params.ny - 1);
    solid[kb] = 1;
    solid[kt] = 1;
    u[kb] = 0; v[kb] = 0;
    u[kt] = 0; v[kt] = 0;
  }
}

function initializeFlowField() {
  const { u, v, uPrev, vPrev, solid } = fields;
  for (let j = 1; j < params.ny - 1; j++) {
    const prof = inletProfile(j);
    const targetU = params.inletVelocity * prof;
    const row = j * params.nx;
    for (let i = 0; i < params.nx; i++) {
      const k = row + i;
      if (solid[k]) continue;
      u[k] = targetU;
      v[k] = 0;
      uPrev[k] = targetU;
      vPrev[k] = 0;
    }
  }
}

function clearFluidKeepGeometry() {
  fields.u.fill(0);
  fields.v.fill(0);
  fields.uPrev.fill(0);
  fields.vPrev.fill(0);
  fields.pressure.fill(0);
  fields.pressurePrev.fill(0);
  fields.divergence.fill(0);
  fields.omega.fill(0);
  fields.omegaMagGradX.fill(0);
  fields.omegaMagGradY.fill(0);
  initializeFlowField();
  applyBoundary();
  computeVorticity();
}

function resetFields(solid) {
  fields = createFields();
  pulseSources = [];
  if (solid && solid.length === fields.solid.length) {
    fields.solid.set(solid);
  }
  markDomainWalls();
  initializeFlowField();
  applyBoundary();
  computeVorticity();
  resetTracers();
}

function sampleBilinear(arr, x, y) {
  const nx = params.nx;
  const ny = params.ny;
  const x0 = clamp(Math.floor(x), 0, nx - 1);
  const y0 = clamp(Math.floor(y), 0, ny - 1);
  const x1 = x0 < nx - 1 ? x0 + 1 : x0;
  const y1 = y0 < ny - 1 ? y0 + 1 : y0;
  const tx = clamp(x - x0, 0, 1);
  const ty = clamp(y - y0, 0, 1);

  const i00 = x0 + y0 * nx;
  const i10 = x1 + y0 * nx;
  const i01 = x0 + y1 * nx;
  const i11 = x1 + y1 * nx;

  const a = arr[i00] + (arr[i10] - arr[i00]) * tx;
  const b = arr[i01] + (arr[i11] - arr[i01]) * tx;
  return a + (b - a) * ty;
}

function velocityColor(speed, maxSpeed) {
  const t = clamp(speed / maxSpeed, 0, 1);
  const cold = [30, 90, 255];
  const mid = [80, 220, 180];
  const hot = [255, 80, 40];

  if (t < 0.5) {
    const s = t / 0.5;
    return [
      cold[0] * (1 - s) + mid[0] * s,
      cold[1] * (1 - s) + mid[1] * s,
      cold[2] * (1 - s) + mid[2] * s,
    ];
  }

  const s = (t - 0.5) / 0.5;
  return [
    mid[0] * (1 - s) + hot[0] * s,
    mid[1] * (1 - s) + hot[1] * s,
    mid[2] * (1 - s) + hot[2] * s,
  ];
}

function pressureColor(p, maxAbsPressure) {
  const t = clamp(p / maxAbsPressure, -1, 1);
  if (t >= 0) {
    const s = t;
    return [255, 255 * (1 - s), 255 * (1 - s)];
  }
  const s = -t;
  return [255 * (1 - s), 255 * (1 - s), 255];
}

function vorticityColor(omega, maxAbsOmega) {
  const t = clamp(omega / maxAbsOmega, -1, 1);
  if (t >= 0) {
    const s = t;
    return [255, 255 * (1 - 0.85 * s), 255 * (1 - s)];
  }
  const s = -t;
  return [255 * (1 - s), 255 * (1 - 0.85 * s), 255];
}

function getRenderRanges() {
  const { u, v, pressure, solid, omega } = fields;
  let maxSpeed = 0.001;
  let maxAbsPressure = 0.001;
  let maxAbsOmega = 0.001;
  const viewMode = params.viewMode ?? 'velocity';
  const needsSpeed = viewMode === 'velocity' || viewMode === 'split';
  const needsPressure = viewMode === 'pressure' || viewMode === 'split';
  const needsOmega = viewMode === 'vorticity';

  for (let k = 0; k < solid.length; k++) {
    if (solid[k]) continue;

    if (needsSpeed) {
      const speed = Math.hypot(u[k], v[k]);
      if (speed > maxSpeed) maxSpeed = speed;
    }

    if (needsPressure) {
      const ap = Math.abs(pressure[k]);
      if (ap > maxAbsPressure) maxAbsPressure = ap;
    }

    if (needsOmega) {
      const ao = Math.abs(omega[k]);
      if (ao > maxAbsOmega) maxAbsOmega = ao;
    }
  }

  return { maxSpeed, maxAbsPressure, maxAbsOmega };
}

function resetTracers() {
  tracers = Array.from({ length: TRACER_COUNT }, () => ({
    x: Math.random() * (params.nx - 4) + 2,
    y: Math.random() * (params.ny - 4) + 2,
    age: Math.random(),
  }));
}

function resetTracerAtInlet(tracer) {
  tracer.x = 1 + Math.random() * Math.max(2, params.inflowColumns + 2);
  tracer.y = 2 + Math.random() * (params.ny - 4);
  tracer.age = 0;
}

function advectTracers() {
  if (!tracers.length) return;
  const { u, v, solid } = fields;
  const scaleX = params.dt * derived.invDx;
  const scaleY = params.dt * derived.invDy;

  for (const tracer of tracers) {
    const i = clamp(Math.round(tracer.x), 0, params.nx - 1);
    const j = clamp(Math.round(tracer.y), 0, params.ny - 1);
    if (solid[idx(i, j)] || tracer.x >= params.nx - 2 || tracer.y <= 1 || tracer.y >= params.ny - 2) {
      resetTracerAtInlet(tracer);
      continue;
    }

    const ux = sampleBilinear(u, tracer.x, tracer.y);
    const vy = sampleBilinear(v, tracer.x, tracer.y);
    tracer.x += ux * scaleX;
    tracer.y += vy * scaleY;
    tracer.age += params.dt;

    if (tracer.x < 1 || tracer.x > params.nx - 2 || tracer.y < 1 || tracer.y > params.ny - 2 || tracer.age > 9) {
      resetTracerAtInlet(tracer);
    }
  }
}

function applyBoundary() {
  const { u, v, solid } = fields;
  const nx = params.nx;
  const ny = params.ny;
  const inflowCols = Math.max(1, Math.min(params.inflowColumns, nx - 2));
  const perturbAmp = params.enableInflowPerturbation ? params.perturbAmount * params.inletVelocity : 0;
  const perturbFreq = 10.0;
  const perturbWave = 0.15;

  for (let j = 1; j < ny - 1; j++) {
    const prof = inletProfile(j);
    const targetU = params.inletVelocity * prof;
    const perturb = perturbAmp * Math.sin(perturbFreq * simTime + j * perturbWave);

    for (let i = 0; i < inflowCols; i++) {
      const k = i + j * nx;
      if (solid[k]) continue;
      u[k] = targetU;
      v[k] = perturb;
    }

    const left = j * nx;
    if (!solid[left]) {
      u[left] = targetU;
      v[left] = perturb;
    }

    const right = (nx - 1) + j * nx;
    const rightIn = (nx - 2) + j * nx;
    if (!solid[right]) {
      u[right] = u[rightIn];
      v[right] = v[rightIn];
    }
  }

  for (let i = 0; i < nx; i++) {
    const b = i;
    const bIn = i + nx;
    const t = i + (ny - 1) * nx;
    const tIn = i + (ny - 2) * nx;
    u[b] = u[bIn];
    u[t] = u[tIn];

    if (params.domainBoundaryMode === 'open') {
      v[b] = v[bIn];
      v[t] = v[tIn];
    } else {
      v[b] = 0;
      v[t] = 0;
    }
  }

  for (let j = 1; j < ny - 1; j++) {
    const row = j * nx;
    const rowB = (j - 1) * nx;
    const rowT = (j + 1) * nx;

    for (let i = 1; i < nx - 1; i++) {
      const k = row + i;
      if (solid[k]) {
        u[k] = 0;
        v[k] = 0;
        continue;
      }

      const leftSolid = solid[row + i - 1] === 1;
      const rightSolid = solid[row + i + 1] === 1;
      const upperSolid = solid[rowB + i] === 1;
      const lowerSolid = solid[rowT + i] === 1;

      if (leftSolid && u[k] < 0) u[k] = 0;
      if (rightSolid && u[k] > 0) u[k] = 0;
      if (upperSolid && v[k] < 0) v[k] = 0;
      if (lowerSolid && v[k] > 0) v[k] = 0;
    }
  }
}

function advectVelocity() {
  const { u, v, uPrev, vPrev, solid } = fields;
  const nx = params.nx;
  const ny = params.ny;
  const scaleX = params.dt * derived.invDx;
  const scaleY = params.dt * derived.invDy;
  uPrev.set(u);
  vPrev.set(v);

  for (let j = 1; j < ny - 1; j++) {
    const row = j * nx;
    for (let i = 1; i < nx - 1; i++) {
      const k = row + i;
      if (solid[k]) {
        u[k] = 0;
        v[k] = 0;
        continue;
      }
      u[k] = sampleBilinear(uPrev, i - scaleX * uPrev[k], j - scaleY * vPrev[k]);
      v[k] = sampleBilinear(vPrev, i - scaleX * uPrev[k], j - scaleY * vPrev[k]);
    }
  }
}

function diffuseVelocity() {
  const { u, v, uPrev, vPrev, scratchU, scratchV, solid } = fields;
  const nx = params.nx;
  const ny = params.ny;
  const h2 = Math.min(derived.dx * derived.dx, derived.dy * derived.dy);
  const alpha = clamp(params.viscosity * params.dt * params.diffusionScale / Math.max(h2, 1e-12), 0, 0.45);
  if (alpha <= 1e-7) return;

  uPrev.set(u);
  vPrev.set(v);
  scratchU.set(u);
  scratchV.set(v);

  for (let iter = 0; iter < 4; iter++) {
    for (let j = 1; j < ny - 1; j++) {
      const row = j * nx;
      const rowB = (j - 1) * nx;
      const rowT = (j + 1) * nx;

      for (let i = 1; i < nx - 1; i++) {
        const k = row + i;
        if (solid[k]) {
          scratchU[k] = 0;
          scratchV[k] = 0;
          continue;
        }

        const l = row + i - 1;
        const r = row + i + 1;
        const b = rowB + i;
        const t = rowT + i;

        const uL = solid[l] ? 0 : scratchU[l];
        const uR = solid[r] ? 0 : scratchU[r];
        const uB = solid[b] ? 0 : scratchU[b];
        const uT = solid[t] ? 0 : scratchU[t];
        const vL = solid[l] ? 0 : scratchV[l];
        const vR = solid[r] ? 0 : scratchV[r];
        const vB = solid[b] ? 0 : scratchV[b];
        const vT = solid[t] ? 0 : scratchV[t];

        u[k] = (uPrev[k] + alpha * (uL + uR + uB + uT)) / (1 + 4 * alpha);
        v[k] = (vPrev[k] + alpha * (vL + vR + vB + vT)) / (1 + 4 * alpha);
      }
    }
    scratchU.set(u);
    scratchV.set(v);
  }
}

function computeVorticity() {
  const { u, v, solid, omega, omegaMagGradX, omegaMagGradY } = fields;
  const nx = params.nx;
  const ny = params.ny;
  const halfInvDx = 0.5 * derived.invDx;
  const halfInvDy = 0.5 * derived.invDy;
  omega.fill(0);
  omegaMagGradX.fill(0);
  omegaMagGradY.fill(0);

  for (let j = 1; j < ny - 1; j++) {
    const row = j * nx;
    const rowB = (j - 1) * nx;
    const rowT = (j + 1) * nx;
    for (let i = 1; i < nx - 1; i++) {
      const k = row + i;
      if (solid[k]) continue;
      omega[k] = (v[row + i + 1] - v[row + i - 1]) * halfInvDx - (u[rowT + i] - u[rowB + i]) * halfInvDy;
    }
  }

  for (let j = 2; j < ny - 2; j++) {
    const row = j * nx;
    const rowB = (j - 1) * nx;
    const rowT = (j + 1) * nx;
    for (let i = 2; i < nx - 2; i++) {
      const k = row + i;
      if (solid[k]) continue;
      omegaMagGradX[k] = (Math.abs(omega[row + i + 1]) - Math.abs(omega[row + i - 1])) * halfInvDx;
      omegaMagGradY[k] = (Math.abs(omega[rowT + i]) - Math.abs(omega[rowB + i])) * halfInvDy;
    }
  }
}

function applyVorticityConfinement() {
  if (!params.enableConfinement || params.confinementStrength <= 0) return;
  const { u, v, solid, omega, omegaMagGradX, omegaMagGradY } = fields;
  const nx = params.nx;
  const ny = params.ny;
  const eps = 1e-8;
  const strength = params.confinementStrength;

  for (let j = 2; j < ny - 2; j++) {
    const row = j * nx;
    for (let i = 2; i < nx - 2; i++) {
      const k = row + i;
      if (solid[k]) continue;
      const gx = omegaMagGradX[k];
      const gy = omegaMagGradY[k];
      const mag = Math.hypot(gx, gy);
      if (mag < eps) continue;
      const nxn = gx / mag;
      const nyn = gy / mag;
      u[k] += params.dt * strength * nyn * omega[k];
      v[k] += params.dt * -strength * nxn * omega[k];
    }
  }
}

function computeDivergence() {
  const { u, v, divergence, solid } = fields;
  const nx = params.nx;
  const ny = params.ny;
  const halfInvDx = 0.5 * derived.invDx;
  const halfInvDy = 0.5 * derived.invDy;
  divergence.fill(0);

  for (let j = 1; j < ny - 1; j++) {
    const row = j * nx;
    const rowB = (j - 1) * nx;
    const rowT = (j + 1) * nx;
    for (let i = 1; i < nx - 1; i++) {
      const k = row + i;
      if (solid[k]) continue;
      const l = row + i - 1;
      const r = row + i + 1;
      const b = rowB + i;
      const t = rowT + i;
      divergence[k] = ((solid[r] ? 0 : u[r]) - (solid[l] ? 0 : u[l])) * halfInvDx
        + ((solid[t] ? 0 : v[t]) - (solid[b] ? 0 : v[b])) * halfInvDy;
    }
  }
}

function solvePressure() {
  const { pressure, pressurePrev, divergence, solid } = fields;
  const nx = params.nx;
  const ny = params.ny;
  const dx2 = derived.dx * derived.dx;
  const dy2 = derived.dy * derived.dy;
  const denom = 2 * (dx2 + dy2);
  pressure.fill(0);
  pressurePrev.fill(0);

  for (let iter = 0; iter < params.pressureIters; iter++) {
    for (let j = 1; j < ny - 1; j++) {
      const row = j * nx;
      const rowB = (j - 1) * nx;
      const rowT = (j + 1) * nx;
      for (let i = 1; i < nx - 1; i++) {
        const k = row + i;
        if (solid[k]) {
          pressure[k] = 0;
          continue;
        }
        const l = row + i - 1;
        const r = row + i + 1;
        const b = rowB + i;
        const t = rowT + i;
        const pL = solid[l] ? pressurePrev[k] : pressurePrev[l];
        const pR = solid[r] ? pressurePrev[k] : pressurePrev[r];
        const pB = solid[b] ? pressurePrev[k] : pressurePrev[b];
        const pT = solid[t] ? pressurePrev[k] : pressurePrev[t];
        pressure[k] = ((pL + pR) * dy2 + (pB + pT) * dx2 - divergence[k] * dx2 * dy2) / denom;
      }
    }
    pressurePrev.set(pressure);
  }
}

function project() {
  const { u, v, pressure, solid } = fields;
  const nx = params.nx;
  const ny = params.ny;
  const halfInvDx = 0.5 * derived.invDx;
  const halfInvDy = 0.5 * derived.invDy;

  for (let j = 1; j < ny - 1; j++) {
    const row = j * nx;
    const rowB = (j - 1) * nx;
    const rowT = (j + 1) * nx;
    for (let i = 1; i < nx - 1; i++) {
      const k = row + i;
      if (solid[k]) {
        u[k] = 0;
        v[k] = 0;
        continue;
      }
      const l = row + i - 1;
      const r = row + i + 1;
      const b = rowB + i;
      const t = rowT + i;
      u[k] -= params.dt * ((solid[r] ? pressure[k] : pressure[r]) - (solid[l] ? pressure[k] : pressure[l])) * halfInvDx;
      v[k] -= params.dt * ((solid[t] ? pressure[k] : pressure[t]) - (solid[b] ? pressure[k] : pressure[b])) * halfInvDy;
    }
  }
}

function stirFluid(point, pushU, pushV, radius) {
  const { u, v, solid } = fields;
  const r2 = radius * radius;
  const minX = clamp(Math.floor(point.x - radius), 1, params.nx - 2);
  const maxX = clamp(Math.ceil(point.x + radius), 1, params.nx - 2);
  const minY = clamp(Math.floor(point.y - radius), 1, params.ny - 2);
  const maxY = clamp(Math.ceil(point.y + radius), 1, params.ny - 2);

  for (let j = minY; j <= maxY; j++) {
    const row = j * params.nx;
    for (let i = minX; i <= maxX; i++) {
      const k = row + i;
      if (solid[k]) continue;
      const ox = i - point.x;
      const oy = j - point.y;
      const d2 = ox * ox + oy * oy;
      if (d2 > r2) continue;
      const w = 0.5 + 0.5 * Math.cos(Math.PI * Math.sqrt(d2) / radius);
      u[k] = u[k] * (1 - 0.25 * w) + (u[k] + pushU) * (0.25 * w);
      v[k] = v[k] * (1 - 0.25 * w) + (v[k] + pushV) * (0.25 * w);
    }
  }

  applyBoundary();
  computeVorticity();
}

function pressurePulse(point, radius, strength) {
  pulseSources = [];

  const { u, v, pressure, solid } = fields;
  const r = Math.max(2, radius);
  const r2 = r * r;
  const minX = clamp(Math.floor(point.x - r), 1, params.nx - 2);
  const maxX = clamp(Math.ceil(point.x + r), 1, params.nx - 2);
  const minY = clamp(Math.floor(point.y - r), 1, params.ny - 2);
  const maxY = clamp(Math.ceil(point.y + r), 1, params.ny - 2);

  for (let j = minY; j <= maxY; j++) {
    const row = j * params.nx;
    for (let i = minX; i <= maxX; i++) {
      const k = row + i;
      if (solid[k]) continue;
      const dx = i - point.x;
      const dy = j - point.y;
      const d2 = dx * dx + dy * dy;
      if (d2 > r2) continue;
      const dist = Math.max(Math.sqrt(d2), 0.001);
      const falloff = 0.5 + 0.5 * Math.cos(Math.PI * dist / r);
      pressure[k] += strength * falloff;
      u[k] += (dx / dist) * strength * 0.04 * falloff;
      v[k] += (dy / dist) * strength * 0.04 * falloff;
    }
  }

  applyBoundary();
  computeVorticity();
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
    strength: clamp(Number(source.strength ?? 1), -15, 15),
    duration,
    remaining: duration,
    mode: source.mode === 'directed' ? 'directed' : 'omni',
    dirX: dx / len,
    dirY: dy / len,
  }];
}

function applyPressureSources() {
  if (pulseSources.length === 0) return;

  const { u, v, pressure, solid } = fields;
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

    for (let j = minY; j <= maxY; j++) {
      const row = j * params.nx;
      for (let i = minX; i <= maxX; i++) {
        const k = row + i;
        if (solid[k]) continue;

        const dx = i - source.x;
        const dy = j - source.y;
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

        const directed = source.mode === 'directed';
        pressure[k] += source.strength * weight * (directed ? 0.35 : 1);
        u[k] += nudgeX * source.strength * (directed ? 0.11 : 0.04) * weight;
        v[k] += nudgeY * source.strength * (directed ? 0.11 : 0.04) * weight;
      }
    }

    source.remaining -= 1;
    if (source.remaining > 0) next.push(source);
  }

  pulseSources = next;
  applyBoundary();
  computeVorticity();
}

function step() {
  simTime += params.dt;
  applyPressureSources();
  advectVelocity();
  applyBoundary();
  diffuseVelocity();
  applyBoundary();
  computeVorticity();
  applyVorticityConfinement();
  applyBoundary();
  computeDivergence();
  solvePressure();
  project();
  applyBoundary();
  computeVorticity();
  advectTracers();
}

function buildPixelFrame() {
  const { u, v, pressure, solid, omega } = fields;
  const pixels = new Uint8ClampedArray(params.nx * params.ny * 4);
  const { maxSpeed, maxAbsPressure, maxAbsOmega } = getRenderRanges();
  const pressureDisplayScale = Math.max(maxAbsPressure * params.pressureVisualScale, 1e-6);
  const splitX = Math.floor(params.nx * (params.splitRatio ?? 0.5));
  const viewMode = params.viewMode ?? 'velocity';

  for (let k = 0; k < solid.length; k++) {
    const p = 4 * k;

    if (solid[k]) {
      pixels[p + 0] = 30;
      pixels[p + 1] = 30;
      pixels[p + 2] = 35;
      pixels[p + 3] = 255;
      continue;
    }

    const speed = Math.hypot(u[k], v[k]);
    let rgb;

    if (viewMode === 'pressure') {
      rgb = pressureColor(pressure[k], pressureDisplayScale);
    } else if (viewMode === 'vorticity') {
      rgb = vorticityColor(omega[k], maxAbsOmega);
    } else if (viewMode === 'split') {
      const i = k % params.nx;
      rgb = i < splitX
        ? velocityColor(speed, maxSpeed)
        : pressureColor(pressure[k], pressureDisplayScale);
    } else {
      rgb = velocityColor(speed, maxSpeed);
    }

    pixels[p + 0] = rgb[0];
    pixels[p + 1] = rgb[1];
    pixels[p + 2] = rgb[2];
    pixels[p + 3] = 255;
  }

  return pixels;
}

function buildVectorSamples() {
  const viewMode = params.viewMode ?? 'velocity';
  if (viewMode !== 'velocity' && viewMode !== 'split') return new Float32Array(0);

  const stride = Math.max(8, Math.round(params.nx / 20));
  const samples = [];
  const vectorLimit = viewMode === 'split'
    ? Math.max(2, Math.floor(params.nx * (params.splitRatio ?? 0.5)) - 2)
    : params.nx - 2;

  for (let j = 2; j < params.ny - 2; j += stride) {
    for (let i = 2; i < vectorLimit; i += stride) {
      const k = idx(i, j);
      if (fields.solid[k]) continue;
      const ux = fields.u[k];
      const vy = fields.v[k];
      if (Math.hypot(ux, vy) < 0.02) continue;
      samples.push(i, j, ux, vy);
    }
  }

  return new Float32Array(samples);
}

function buildFlowlineSegments() {
  if (!params.showFlowlines) return new Float32Array(0);

  const seedRows = params.nx >= 400 ? 18 : 22;
  const seedCols = params.nx >= 400 ? 4 : 5;
  const maxSteps = params.nx >= 400 ? 58 : 70;
  const stepCells = 0.9;
  const segments = [];

  for (let sy = 0; sy < seedRows; sy++) {
    const seedY = 2 + (sy + 0.5) * (params.ny - 4) / seedRows;

    for (let sx = 0; sx < seedCols; sx++) {
      const seedX = 2 + sx * (params.nx - 4) / seedCols;
      let x = seedX;
      let y = seedY;
      let started = false;

      for (let n = 0; n < maxSteps; n++) {
        const i = clamp(Math.round(x), 0, params.nx - 1);
        const j = clamp(Math.round(y), 0, params.ny - 1);
        if (fields.solid[idx(i, j)] || x < 1 || x > params.nx - 2 || y < 1 || y > params.ny - 2) break;

        const ux = sampleBilinear(fields.u, x, y);
        const vy = sampleBilinear(fields.v, x, y);
        const speed = Math.hypot(ux, vy);
        if (speed < 0.02) break;

        const x0 = x;
        const y0 = y;
        x += (ux / speed) * stepCells;
        y += (vy / speed) * stepCells;

        if (started || n % 2 === 0) {
          segments.push(x0, y0, x, y);
          started = true;
        }
      }
    }
  }

  return new Float32Array(segments);
}

function postFrame() {
  if (!fields) return;

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
    tracerData[p + 0] = tracers[i].x;
    tracerData[p + 1] = tracers[i].y;
    tracerData[p + 2] = tracers[i].age;
  }

  postMessage({
    type: 'frame',
    simTime,
    pixels,
    vectorSamples,
    flowlineSegments,
    tracers: tracerData,
    pulseSources: pulseData,
  }, [
    pixels.buffer,
    vectorSamples.buffer,
    flowlineSegments.buffer,
    tracerData.buffer,
    pulseData.buffer,
  ]);
}

function tick() {
  if (params && fields && running && !isBatchRunning) {
    const loops = Math.max(1, params.substepsPerFrame);
    const frameStart = performance.now();
    let completed = 0;
    for (let i = 0; i < loops; i++) {
      step();
      completed++;
      if (completed > 1 && performance.now() - frameStart > params.maxFrameMs) break;
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
  if (!params || !fields || isBatchRunning) return;
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
  updateDerived();
}

onmessage = event => {
  const msg = event.data;

  if (msg.type === 'init') {
    params = { ...msg.params };
    simTime = msg.simTime ?? 0;
    running = Boolean(msg.running);
    updateDerived();
    resetFields(msg.solid ? new Uint8Array(msg.solid) : null);
    postFrame();
    postMessage({ type: 'ready' });
    return;
  }

  if (!params) return;

  if (msg.type === 'configure') {
    configure(msg.params);
    return;
  }

  if (msg.type === 'running') {
    running = Boolean(msg.running);
    return;
  }

  if (msg.type === 'reset') {
    configure(msg.params ?? {});
    simTime = msg.simTime ?? 0;
    resetFields(msg.solid ? new Uint8Array(msg.solid) : null);
    postFrame();
    return;
  }

  if (msg.type === 'setSolid') {
    if (msg.params) configure(msg.params);
    fields.solid.set(new Uint8Array(msg.solid));
    markDomainWalls();
    clearFluidKeepGeometry();
    postFrame();
    return;
  }

  if (msg.type === 'resetTracers') {
    resetTracers();
    postFrame();
    return;
  }

  if (msg.type === 'stir') {
    stirFluid(msg.point, msg.pushU, msg.pushV, msg.radius);
    postFrame();
    return;
  }

  if (msg.type === 'pressurePulse') {
    pressurePulse(msg.point, msg.radius, msg.strength ?? 1);
    postFrame();
    return;
  }

  if (msg.type === 'addPressureSource') {
    addPressureSource(msg.source ?? {});
    postFrame();
    return;
  }

  if (msg.type === 'batch') {
    runBatch(msg.totalSteps, msg.chunkSize ?? 50, msg.label ?? 'Kör');
  }
};

tick();

// Procedural, interactive 3D lungs with a tumour whose growth and spread follow cancer stages 0-IV.
// Everything is generated in code (no model files). Anatomical view: the patient faces the viewer,
// so the patient's right lung (larger, 3 lobes) is on the viewer's left.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";

// Seeded random numbers so the airway tree looks the same on every load.
function rng(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

// Maps a point of the unit sphere (or inside it) onto a lung shape. side: -1 = patient's right, +1 = left.
function deform(x, y, z, side) {
  const medial = -side; // direction of the heart / midline
  let X = x, Y = y * 1.5, Z = z;
  const broad = y < 0 ? 1 + 0.32 * -y : 1 - 0.28 * y;     // broad base, narrower apex
  X *= broad; Z *= broad;
  if (y < -0.2) {                                          // concave base resting on the diaphragm dome
    const r2 = Math.min(1, (x * x + z * z));
    const dome = -1.28 + 0.42 * (1 - r2);
    Y = Y + (Math.max(Y, dome) - Y) * smooth(-0.2, -0.85, y);
  }
  if (X * medial > 0) X *= 0.58;                           // flatter medial surface
  if (side > 0 && X < 0) {                                 // cardiac notch of the left lung
    const notch = Math.exp(-((Y + 0.45) ** 2) / 0.16) * smooth(-0.1, 0.7, z);
    X *= 1 - 0.75 * notch;
  }
  const n = 0.03 * Math.sin(3.1 * x + 1.7 * y) * Math.cos(2.3 * z - 1.1 * y) + 0.015 * Math.sin(7 * y + 3 * z);
  const w = side < 0 ? 0.86 : 0.76;                        // right lung is wider
  return new THREE.Vector3(X * w * (1 + n), Y * (1 + n * 0.5), Z * 0.74 * (1 + n));
}

function lungGeometry(side) {
  const g = new THREE.SphereGeometry(1, 120, 90);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const v = deform(p.getX(i), p.getY(i), p.getZ(i), side);
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}

function fresnelMaterial(inner, rim, base = 0.05, rimA = 0.75) {
  return new THREE.ShaderMaterial({
    uniforms: { uInner: { value: new THREE.Color(inner) }, uRim: { value: new THREE.Color(rim) },
                uBase: { value: base }, uRimA: { value: rimA }, uPulse: { value: 0 } },
    vertexShader: `
      varying vec3 vN; varying vec3 vV;
      void main() {
        vN = normalize(normalMatrix * normal);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vV = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform vec3 uInner; uniform vec3 uRim; uniform float uBase; uniform float uRimA; uniform float uPulse;
      varying vec3 vN; varying vec3 vV;
      void main() {
        float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.4);
        vec3 col = mix(uInner, uRim, f);
        gl_FragColor = vec4(col, uBase + f * uRimA + uPulse * 0.08);
      }`,
    transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
  });
}

function glowTexture(color) {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d"), grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, color); grd.addColorStop(0.35, color.replace(/[\d.]+\)$/, "0.35)")); grd.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

function tumourGeometry(seed) {
  const r = rng(seed), g = new THREE.IcosahedronGeometry(1, 5), p = g.attributes.position;
  const k = [r() * 6, r() * 6, r() * 6];
  for (let i = 0; i < p.count; i++) {
    const v = new THREE.Vector3(p.getX(i), p.getY(i), p.getZ(i));
    const lumps = 1 + 0.16 * Math.sin(5 * v.x + k[0]) * Math.sin(4 * v.y + k[1]) + 0.1 * Math.sin(9 * v.z + k[2])
      + 0.06 * Math.sin(17 * v.x + 13 * v.y);                         // spiculated, irregular surface
    v.multiplyScalar(lumps); p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}

export const STAGES = [
  { name: "Healthy lungs", size: 0 },
  { name: "Stage I", size: 0.13 },
  { name: "Stage II", size: 0.2 },
  { name: "Stage III", size: 0.25 },
  { name: "Stage IV", size: 0.28 },
];

export function createLungs(container, opts = {}) {
  const { hotspots: showHotspots = true, zoom = true, stage: initialStage = 1, onSelect = () => {} } = opts;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  } catch (e) {
    container.innerHTML = `<div style="display:grid;place-items:center;height:100%;color:var(--muted);text-align:center;padding:24px">
      3D view needs WebGL, which this browser has turned off.</div>`;
    return { setStage() {}, select() {}, dispose() {} };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.style.display = "block";
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
  camera.position.set(0, 0.25, 7.6);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.07;
  controls.enablePan = false; controls.enableZoom = zoom;
  controls.minDistance = 4.2; controls.maxDistance = 11;
  controls.autoRotate = !reduced; controls.autoRotateSpeed = 0.7;
  controls.target.set(0, 0.05, 0);

  scene.add(new THREE.AmbientLight(0xbcd4ff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(3, 4, 5); scene.add(key);
  const rimL = new THREE.PointLight(0x4cc2ff, 18, 20); rimL.position.set(-4, 1, -3); scene.add(rimL);
  const rimR = new THREE.PointLight(0xff7ab8, 14, 20); rimR.position.set(4, -1, -2); scene.add(rimR);

  const root = new THREE.Group(); scene.add(root);
  const lungs = new THREE.Group(); root.add(lungs);

  // ---- lungs ----
  const LUNG_POS = { "-1": new THREE.Vector3(-1.02, -0.18, 0), "1": new THREE.Vector3(1.0, -0.22, 0) };
  const lungMat = fresnelMaterial("#ff8fb8", "#9adfff", 0.035, 0.8);
  for (const side of [-1, 1]) {
    const m = new THREE.Mesh(lungGeometry(side), lungMat);
    m.position.copy(LUNG_POS[side]);
    lungs.add(m);
  }

  // ---- alveoli: glowing points scattered through each lung ----
  {
    const r = rng(7), pts = [], cols = [];
    const cA = new THREE.Color("#ffb3d1"), cB = new THREE.Color("#9adfff");
    for (const side of [-1, 1]) {
      for (let i = 0; i < 2200; i++) {
        let x, y, z;
        do { x = r() * 2 - 1; y = r() * 2 - 1; z = r() * 2 - 1; } while (x * x + y * y + z * z > 0.8);
        const v = deform(x, y, z, side).add(LUNG_POS[side]);
        pts.push(v.x, v.y, v.z);
        const c = cA.clone().lerp(cB, r()); cols.push(c.r, c.g, c.b);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
    const mat = new THREE.PointsMaterial({ size: 0.028, vertexColors: true, transparent: true, opacity: 0.55,
      blending: THREE.AdditiveBlending, depthWrite: false });
    lungs.add(new THREE.Points(g, mat));
  }

  // ---- airways: trachea, main bronchi, branching tree ----
  const airwayMat = new THREE.MeshStandardMaterial({ color: 0xdfe9f5, roughness: 0.45, metalness: 0.05,
    emissive: 0x2b4a6b, emissiveIntensity: 0.35, transparent: true, opacity: 0.92 });
  const airways = new THREE.Group(); root.add(airways);
  function tube(a, b, radius, bend = 0.08, r = Math.random) {
    const mid = a.clone().lerp(b, 0.5).add(new THREE.Vector3((r() - 0.5) * bend, (r() - 0.5) * bend, (r() - 0.5) * bend));
    const g = new THREE.TubeGeometry(new THREE.CatmullRomCurve3([a, mid, b]), 10, radius, 8, false);
    airways.add(new THREE.Mesh(g, airwayMat));
  }
  const inside = (v, side) => {
    const c = LUNG_POS[side], d = v.clone().sub(c);
    return (d.x / 0.72) ** 2 + ((d.y + 0.1) / 1.3) ** 2 + (d.z / 0.58) ** 2 < 1;
  };
  const top = new THREE.Vector3(0, 2.0, 0.05), carina = new THREE.Vector3(0, 0.78, 0.05);
  tube(top, carina, 0.085, 0.02);
  // cartilage rings on the trachea
  for (let i = 0; i < 9; i++) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.09, 0.012, 8, 28), airwayMat);
    ring.rotation.x = Math.PI / 2; ring.position.set(0, 1.9 - i * 0.12, 0.05); airways.add(ring);
  }
  const hilum = { "-1": new THREE.Vector3(-0.62, 0.35, 0.02), "1": new THREE.Vector3(0.66, 0.26, 0.02) };
  const rr = rng(42);
  function branch(from, dir, len, radius, depth, side) {
    if (depth === 0) return;
    let to = from.clone().add(dir.clone().multiplyScalar(len));
    for (let k = 0; k < 5 && !inside(to, side); k++) { len *= 0.7; to = from.clone().add(dir.clone().multiplyScalar(len)); }
    tube(from, to, radius, 0.05 * len, rr);
    for (let c = 0; c < 2; c++) {
      const axis = new THREE.Vector3(rr() - 0.5, rr() - 0.5, rr() - 0.5).normalize();
      const nd = dir.clone().applyAxisAngle(axis, (c === 0 ? 1 : -1) * (0.45 + rr() * 0.35)).normalize();
      branch(to, nd, len * (0.7 + rr() * 0.12), radius * 0.68, depth - 1, side);
    }
  }
  for (const side of [-1, 1]) {
    tube(carina, hilum[side], 0.06, 0.04);
    for (const d of [[0.25, 0.9, 0.05], [0.55, -0.35, 0.2], [0.2, -1, -0.15], [0.5, -0.6, -0.35]]) {
      branch(hilum[side], new THREE.Vector3(d[0] * side, d[1], d[2]).normalize(), 0.42, 0.04, 5, side);
    }
  }

  // ---- heart (context) ----
  const heart = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 36), fresnelMaterial("#ff5a6e", "#ff9aa9", 0.008, 0.16));
  heart.scale.set(0.5, 0.6, 0.45); heart.position.set(0.3, -0.7, 0.05); heart.rotation.z = -0.5;
  root.add(heart);

  // ---- lymph nodes ----
  const nodeMat = () => new THREE.MeshStandardMaterial({ color: 0x7fa6a0, roughness: 0.6, emissive: 0x000000 });
  const NODES = {
    hilar: [new THREE.Vector3(-0.5, 0.3, 0.18), new THREE.Vector3(-0.58, 0.12, 0.12)],
    mediastinal: [new THREE.Vector3(0.14, 0.95, 0.18), new THREE.Vector3(-0.14, 0.72, 0.2), new THREE.Vector3(0.18, 0.62, 0.16)],
  };
  const nodes = {};
  for (const [k, list] of Object.entries(NODES)) {
    nodes[k] = list.map((p) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.055, 20, 16), nodeMat());
      m.position.copy(p); root.add(m); return m;
    });
  }

  // ---- tumours ----
  const TUMOUR_POS = new THREE.Vector3(-1.12, 0.58, 0.28);           // right upper lobe
  const METS_POS = new THREE.Vector3(1.12, -0.52, 0.18);             // other lung (stage IV)
  const tumourMat = new THREE.MeshStandardMaterial({ color: 0xff5a5f, roughness: 0.42, metalness: 0.05,
    emissive: 0xff2d55, emissiveIntensity: 0.55 });
  const tumour = new THREE.Mesh(tumourGeometry(3), tumourMat); tumour.position.copy(TUMOUR_POS); root.add(tumour);
  const mets = new THREE.Mesh(tumourGeometry(9), tumourMat); mets.position.copy(METS_POS); root.add(mets);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(255,90,95,0.9)"), blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true }));
  halo.position.copy(TUMOUR_POS); root.add(halo);

  // stage IV: cells travelling from the tumour towards distant organs
  const spread = new THREE.Group(); root.add(spread);
  const DEST = [new THREE.Vector3(-0.3, 3.2, 0.4), new THREE.Vector3(-2.8, -1.8, 0.6), new THREE.Vector3(2.4, -2.4, 0.8)];
  const cells = [];
  const cellMat = new THREE.SpriteMaterial({ map: glowTexture("rgba(255,140,120,0.95)"), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true });
  DEST.forEach((d, i) => {
    const curve = new THREE.QuadraticBezierCurve3(TUMOUR_POS.clone(), TUMOUR_POS.clone().lerp(d, 0.5).add(new THREE.Vector3(0, 0.6, 0.6)), d);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)),
      new THREE.LineDashedMaterial({ color: 0xff8a7a, dashSize: 0.08, gapSize: 0.06, transparent: true, opacity: 0.6 }));
    line.computeLineDistances(); spread.add(line);
    for (let j = 0; j < 4; j++) {
      const s = new THREE.Sprite(cellMat); s.scale.setScalar(0.14); spread.add(s);
      cells.push({ s, curve, off: j / 4 + i * 0.13 });
    }
  });

  // ---- hotspots (HTML labels that follow 3D points) ----
  const HOTSPOTS = [
    { id: "trachea", label: "Trachea", pos: new THREE.Vector3(0, 1.6, 0.12), minStage: 0 },
    { id: "bronchi", label: "Bronchi", pos: new THREE.Vector3(-0.38, 0.6, 0.1), minStage: 0 },
    { id: "right", label: "Right lung", pos: new THREE.Vector3(-1.45, -0.55, 0.35), minStage: 0 },
    { id: "left", label: "Left lung", pos: new THREE.Vector3(1.4, -0.1, 0.3), minStage: 0 },
    { id: "alveoli", label: "Alveoli", pos: new THREE.Vector3(1.05, -0.95, 0.4), minStage: 0 },
    { id: "tumour", label: "Tumour", pos: TUMOUR_POS.clone().add(new THREE.Vector3(0, 0.1, 0.2)), minStage: 1 },
    { id: "nodes", label: "Lymph nodes", pos: new THREE.Vector3(-0.1, 0.85, 0.3), minStage: 2 },
    { id: "spread", label: "Spread", pos: new THREE.Vector3(1.12, -0.35, 0.4), minStage: 4 },
  ];
  const layer = document.createElement("div");
  layer.style.cssText = "position:absolute;inset:0;pointer-events:none;overflow:hidden";
  container.style.position = container.style.position || "relative";
  container.appendChild(layer);
  let selected = null;
  if (showHotspots) {
    HOTSPOTS.forEach((h) => {
      const b = document.createElement("button");
      b.className = "hotspot"; b.type = "button";
      b.innerHTML = `<span class="hs-dot"></span><span class="hs-label">${h.label}</span>`;
      b.addEventListener("click", (e) => { e.stopPropagation(); select(h.id); });
      layer.appendChild(b); h.el = b;
    });
  }
  function select(id) {
    selected = id;
    HOTSPOTS.forEach((h) => h.el && h.el.classList.toggle("on", h.id === id));
    onSelect(id);
  }

  // ---- stage ----
  let stage = -1, shownSize = 0;
  function setStage(n) {
    stage = Math.max(0, Math.min(4, n));
    nodes.hilar.forEach((m) => m.material.emissive.setHex(stage >= 2 ? 0xff7a45 : 0x000000));
    nodes.hilar.forEach((m) => m.material.color.setHex(stage >= 2 ? 0xffa27a : 0x7fa6a0));
    nodes.mediastinal.forEach((m) => { m.material.emissive.setHex(stage >= 3 ? 0xff7a45 : 0x000000); m.material.color.setHex(stage >= 3 ? 0xffa27a : 0x7fa6a0); });
    mets.visible = stage >= 4; spread.visible = stage >= 4;
    HOTSPOTS.forEach((h) => h.el && (h.el.style.display = stage >= h.minStage ? "" : "none"));
    if (selected && HOTSPOTS.find((h) => h.id === selected).minStage > stage) select(null);
  }
  setStage(initialStage);

  // ---- render loop ----
  let visible = true, raf = 0;
  const clock = new THREE.Clock();
  const v = new THREE.Vector3();
  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%";
    camera.aspect = w / h;
    camera.position.setLength((w < 520 ? 9.0 : 7.2) * (opts.distance || 1));   // fit narrow screens
    camera.updateProjectionMatrix();
  }
  function frame() {
    raf = requestAnimationFrame(frame);
    if (!visible) return;
    const t = clock.getElapsedTime();
    const breath = reduced ? 0 : Math.sin(t * 1.25);
    lungs.scale.set(1 + 0.018 * breath, 1 + 0.028 * breath, 1 + 0.018 * breath);
    const target = STAGES[stage].size;
    shownSize += (target - shownSize) * 0.08;
    tumour.visible = shownSize > 0.01; halo.visible = tumour.visible;
    tumour.scale.setScalar(Math.max(0.001, shownSize));
    tumour.rotation.y = t * 0.3;
    mets.scale.setScalar(0.12 + 0.01 * Math.sin(t * 3));
    const pulse = 0.5 + 0.5 * Math.sin(t * 2.6);
    halo.scale.setScalar(shownSize * (5 + 1.4 * pulse));
    halo.material.opacity = 0.55 + 0.35 * pulse;
    tumourMat.emissiveIntensity = 0.45 + 0.35 * pulse;
    if (spread.visible) cells.forEach((c) => { c.s.position.copy(c.curve.getPoint((t * 0.18 + c.off) % 1)); });
    controls.update();
    renderer.render(scene, camera);
    if (showHotspots) {
      const w = container.clientWidth, h = container.clientHeight;
      HOTSPOTS.forEach((hs) => {
        if (!hs.el || hs.el.style.display === "none") return;
        v.copy(hs.pos).applyMatrix4(root.matrixWorld);
        const depth = v.clone().sub(camera.position).length();
        v.project(camera);
        hs.el.style.transform = `translate(${(v.x * 0.5 + 0.5) * w}px, ${(-v.y * 0.5 + 0.5) * h}px)`;
        hs.el.style.opacity = String(Math.max(0.35, Math.min(1, 1.9 - (depth - 5.5) * 0.45)));
      });
    }
  }
  const ro = new ResizeObserver(resize); ro.observe(container); resize();
  const vis = new IntersectionObserver(([e]) => { visible = e.isIntersecting; }); vis.observe(container);
  frame();

  return {
    setStage,
    select,
    get stage() { return stage; },
    dispose() { cancelAnimationFrame(raf); ro.disconnect(); vis.disconnect(); renderer.dispose(); },
  };
}

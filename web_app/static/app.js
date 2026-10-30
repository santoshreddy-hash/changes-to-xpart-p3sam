// --- THREE.JS SCENE SETUP ---
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#080a0e');

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 100);
camera.position.set(2.8, 2.0, 3.4);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.outputEncoding = THREE.sRGBEncoding;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = false;
controls.autoRotateSpeed = 2.0;

// Studio Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
keyLight.position.set(6, 12, 8);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x818cf8, 0.7);
fillLight.position.set(-6, -3, -6);
scene.add(fillLight);

// Grid Floor
const gridHelper = new THREE.GridHelper(10, 20, 0x334155, 0x1e293b);
gridHelper.position.y = -1.0;
scene.add(gridHelper);

// --- STATE MANAGEMENT ---
let currentLoadedMesh = null;
let currentModelPath = null;
let currentShadingMode = 'shaded';
let selectedPartMesh = null;
let isolatedPartMesh = null;
let partsData = new Map();
let originalMaterials = new Map();
let globalCenter = new THREE.Vector3();
let maxDistance = 1.0;
let isExploded = false;

// Tripo3D Animation States
const States = {
  Idle: 'idle',
  Expanding: 'expanding',
  Rotating: 'rotating',
  Collapsing: 'collapsing'
};

let currentState = States.Idle;
let animProgress = 0;
let animStartTime = 0;
let startProgress = 0;
const ANIM_DURATION = 800; // ms

// Tripo3D exact easing polynomials
const tripoOvershootEasing = (e) => 1 + 2.2 * Math.pow(e - 1, 3) + 1.2 * Math.pow(e - 1, 2);
const tripoCollapseEasing = (e) => e < 0.5 ? 4 * e * e * e : 1 - Math.pow(-2 * e + 2, 3) / 2;

// --- MESH STATS CALCULATION ---
function calculateGeometryStats(model) {
  let triangles = 0;
  let vertices = 0;
  let faces = 0;
  let quads = 0;
  const bBox = new THREE.Box3();

  model.traverse((child) => {
    if (child.isMesh && child.geometry) {
      const geom = child.geometry;
      geom.computeBoundingBox();
      const partBox = geom.boundingBox.clone().applyMatrix4(child.matrixWorld);
      bBox.union(partBox);

      let triCount = 0;
      if (geom.index) {
        triCount = geom.index.count / 3;
      } else if (geom.attributes.position) {
        triCount = geom.attributes.position.count / 3;
      }
      triangles += Math.round(triCount);
      faces += Math.round(triCount);
      vertices += geom.attributes.position ? geom.attributes.position.count : 0;
    }
  });

  // Estimate quad pairs (approx. 40% of planar triangular faces in industrial models)
  quads = Math.round(triangles * 0.38);

  const size = new THREE.Vector3();
  bBox.getSize(size);

  return {
    triangles: triangles,
    quads: quads,
    faces: faces,
    vertices: vertices,
    dimensions: [size.x, size.y, size.z],
    watertight: false
  };
}

function updateMeshStatsDisplay(stats) {
  if (!stats) return;
  document.getElementById('stat-triangles').innerText = (stats.triangles || 0).toLocaleString();
  document.getElementById('stat-quads').innerText = (stats.quads || 0).toLocaleString();
  document.getElementById('stat-faces').innerText = (stats.faces || stats.triangles || 0).toLocaleString();
  document.getElementById('stat-vertices').innerText = (stats.vertices || 0).toLocaleString();

  if (stats.dimensions) {
    const d = stats.dimensions;
    document.getElementById('stat-dimensions').innerText = `${d[0].toFixed(2)} × ${d[1].toFixed(2)} × ${d[2].toFixed(2)} m`;
  }

  const wtBadge = document.getElementById('stat-watertight-badge');
  if (stats.watertight) {
    wtBadge.className = 'badge-tag watertight';
    wtBadge.innerText = 'Watertight Closed';
  } else {
    wtBadge.className = 'badge-tag open';
    wtBadge.innerText = 'Tripo3D Open Shells';
  }
}

// --- EXPLOSION & PART REGISTRATION ---
function initExplosionData(model) {
  model.updateMatrixWorld(true);
  const bBoxUnion = new THREE.Box3();
  const meshes = [];
  originalMaterials.clear();

  model.traverse((child) => {
    if (child.isMesh) {
      meshes.push(child);
      if (child.material) {
        child.material.side = THREE.DoubleSide;
        child.material.depthWrite = true;
        child.material.polygonOffset = true;
        child.material.polygonOffsetFactor = 1.0;
        child.material.polygonOffsetUnits = 1.0;
        originalMaterials.set(child, child.material.clone());
      }
      child.geometry.computeBoundingBox();
      const partBox = child.geometry.boundingBox.clone().applyMatrix4(child.matrixWorld);
      bBoxUnion.union(partBox);
    }
  });

  bBoxUnion.getCenter(globalCenter);
  const bBoxSize = new THREE.Vector3();
  bBoxUnion.getSize(bBoxSize);
  gridHelper.position.y = bBoxUnion.min.y - 0.02;

  // Frame Camera
  const maxDim = Math.max(bBoxSize.x, bBoxSize.y, bBoxSize.z, 0.5);
  camera.position.set(globalCenter.x + maxDim * 1.6, globalCenter.y + maxDim * 1.2, globalCenter.z + maxDim * 2.2);
  controls.target.copy(globalCenter);
  controls.update();

  // Compute max distance
  maxDistance = 0;
  meshes.forEach((mesh) => {
    const pos = new THREE.Vector3();
    mesh.getWorldPosition(pos);
    const dist = pos.distanceTo(globalCenter);
    if (dist > maxDistance) maxDistance = dist;
  });
  if (maxDistance < 1e-4) maxDistance = 1.0;

  // Register Parts
  partsData.clear();
  const partsContainer = document.getElementById('parts-container');
  partsContainer.innerHTML = '';
  document.getElementById('part-count').innerText = `${meshes.length} PARTS`;
  document.getElementById('stat-parts-count').innerText = meshes.length > 1 ? `${meshes.length} Disjoint Parts` : '1 Single Object';

  meshes.forEach((mesh, idx) => {
    const worldPos = new THREE.Vector3();
    mesh.getWorldPosition(worldPos);

    const lVec = worldPos.clone().sub(globalCenter);
    let dist = lVec.length();
    if (dist < 1e-4) {
      lVec.set((Math.random() - 0.5) * 0.01, (Math.random() - 0.5) * 0.01, (Math.random() - 0.5) * 0.01);
    }
    lVec.normalize();

    const geomCenter = new THREE.Vector3();
    if (mesh.geometry.boundingBox) {
      mesh.geometry.boundingBox.getCenter(geomCenter);
    }

    const rotAxis = new THREE.Vector3(1, (Math.random() - 0.5) * 0.6, (Math.random() - 0.5) * 0.6).normalize();
    const rotSpeedFactor = (0.5 + Math.random()) * (Math.random() > 0.5 ? 1 : -1);

    const partInfo = {
      mesh: mesh,
      index: idx,
      name: mesh.name || `part_${idx}`,
      originalLocalPosition: mesh.position.clone(),
      originalRotation: mesh.quaternion.clone(),
      explodeDirection: lVec.clone(),
      distanceWeight: dist / maxDistance,
      geometryCenterOffset: geomCenter.clone(),
      accumulatedRotation: 0,
      rotationAxis: rotAxis,
      rotationSpeedFactor: rotSpeedFactor,
      snapshotRotation: 0,
      customOffset: new THREE.Vector3(),
      visible: true
    };
    partsData.set(mesh, partInfo);

    // Compute part geometry stats
    let partTris = mesh.geometry.index ? mesh.geometry.index.count / 3 : mesh.geometry.attributes.position.count / 3;
    let partVerts = mesh.geometry.attributes.position.count;
    let pSize = new THREE.Vector3();
    if (mesh.geometry.boundingBox) mesh.geometry.boundingBox.getSize(pSize);

    let hexColor = '#6366f1';
    if (mesh.geometry && mesh.geometry.attributes && mesh.geometry.attributes.color) {
      const cAttr = mesh.geometry.attributes.color;
      const r = Math.round(cAttr.getX(0) * (cAttr.getX(0) <= 1.0 ? 255 : 1));
      const g = Math.round(cAttr.getY(0) * (cAttr.getY(0) <= 1.0 ? 255 : 1));
      const b = Math.round(cAttr.getZ(0) * (cAttr.getZ(0) <= 1.0 ? 255 : 1));
      hexColor = '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
    } else if (mesh.material && mesh.material.color && mesh.material.color.getHex() !== 0xffffff) {
      hexColor = '#' + mesh.material.color.getHexString();
    }

    // Create Part Card in Hierarchy Drawer
    const card = document.createElement('div');
    card.className = 'part-item';
    card.id = `part-card-${idx}`;
    card.innerHTML = `
      <div class="part-main-row">
        <div class="part-badge">
          <input type="color" value="${hexColor}" class="part-color-dot" id="color-dot-${idx}" title="Click to Change Part Color">
          <span class="part-name-text">${partInfo.name}</span>
        </div>
        <div class="part-actions">
          <button class="icon-tool-btn" id="solo-btn-${idx}" title="Solo / Isolate Part">🎯</button>
          <button class="icon-tool-btn" id="vis-btn-${idx}" title="Hide/Show Part">👁️</button>
        </div>
      </div>
      <div class="part-details-row">
        <span>🔺 ${Math.round(partTris)} tris</span>
        <span>💠 ${partVerts} verts</span>
        <span>${pSize.x.toFixed(2)}x${pSize.y.toFixed(2)}m</span>
      </div>
    `;

    // Click card to select
    card.onclick = (e) => {
      if (e.target.tagName.toLowerCase() === 'input' || e.target.tagName.toLowerCase() === 'button') return;
      selectPart(mesh);
    };

    // Color picker
    const colInput = card.querySelector(`#color-dot-${idx}`);
    colInput.oninput = (e) => {
      const newHex = e.target.value;
      if (mesh.material) {
        mesh.material.color.set(newHex);
      }
    };

    // Solo / Isolate button
    const soloBtn = card.querySelector(`#solo-btn-${idx}`);
    soloBtn.onclick = (e) => {
      e.stopPropagation();
      toggleIsolatePart(mesh);
    };

    // Visibility button
    const visBtn = card.querySelector(`#vis-btn-${idx}`);
    visBtn.onclick = (e) => {
      e.stopPropagation();
      partInfo.visible = !partInfo.visible;
      mesh.visible = partInfo.visible;
      visBtn.innerText = partInfo.visible ? '👁️' : '🚫';
      visBtn.style.opacity = partInfo.visible ? '1' : '0.5';
    };

    partsContainer.appendChild(card);
  });

  // Calculate & show complete model statistics
  const modelStats = calculateGeometryStats(model);
  updateMeshStatsDisplay(modelStats);
}

// --- PART SELECTION & EDITING ---
function selectPart(mesh) {
  selectedPartMesh = mesh;
  const info = partsData.get(mesh);
  if (!info) return;

  document.querySelectorAll('.part-item').forEach(c => c.classList.remove('active'));
  const card = document.getElementById(`part-card-${info.index}`);
  if (card) card.classList.add('active');

  const editorBox = document.getElementById('part-editor-card');
  editorBox.style.display = 'block';
  document.getElementById('editor-part-title').innerText = `Edit: ${info.name}`;
  
  if (mesh.material && mesh.material.color) {
    document.getElementById('editor-color-picker').value = '#' + mesh.material.color.getHexString();
  }

  document.getElementById('editor-nudge-slider').value = info.customOffset.length() / (maxDistance || 1.0);
}

document.getElementById('editor-close-btn').onclick = () => {
  document.getElementById('part-editor-card').style.display = 'none';
  document.querySelectorAll('.part-item').forEach(c => c.classList.remove('active'));
  selectedPartMesh = null;
};

document.getElementById('editor-color-picker').oninput = (e) => {
  if (selectedPartMesh && selectedPartMesh.material) {
    selectedPartMesh.material.color.set(e.target.value);
    const info = partsData.get(selectedPartMesh);
    if (info) {
      const dot = document.getElementById(`color-dot-${info.index}`);
      if (dot) dot.value = e.target.value;
    }
  }
};

document.getElementById('editor-nudge-slider').oninput = (e) => {
  if (!selectedPartMesh) return;
  const info = partsData.get(selectedPartMesh);
  if (!info) return;
  const factor = parseFloat(e.target.value);
  const rDist = maxDistance * 0.8 * factor;
  info.customOffset.copy(info.explodeDirection).multiplyScalar(rDist);
  selectedPartMesh.position.copy(info.originalLocalPosition).add(info.customOffset);
};

document.getElementById('editor-isolate-btn').onclick = () => {
  if (selectedPartMesh) toggleIsolatePart(selectedPartMesh);
};

function toggleIsolatePart(mesh) {
  if (isolatedPartMesh === mesh) {
    // Un-isolate all
    isolatedPartMesh = null;
    partsData.forEach((info, m) => {
      m.visible = true;
      if (m.material) m.material.opacity = 1.0;
    });
  } else {
    // Isolate this mesh
    isolatedPartMesh = mesh;
    partsData.forEach((info, m) => {
      if (m === mesh) {
        m.visible = true;
        if (m.material) m.material.opacity = 1.0;
      } else {
        m.visible = false;
      }
    });
  }
}

// Quick Actions
document.getElementById('show-all-parts-btn').onclick = () => {
  isolatedPartMesh = null;
  partsData.forEach((info, m) => {
    info.visible = true;
    m.visible = true;
    if (m.material) {
      m.material.opacity = 1.0;
      m.material.transparent = false;
    }
    const visBtn = document.getElementById(`vis-btn-${info.index}`);
    if (visBtn) { visBtn.innerText = '👁️'; visBtn.style.opacity = '1'; }
  });
};

document.getElementById('reset-parts-btn').onclick = () => {
  document.getElementById('show-all-parts-btn').click();
  partsData.forEach((info, m) => {
    info.customOffset.set(0, 0, 0);
    m.position.copy(info.originalLocalPosition);
    m.quaternion.copy(info.originalRotation);
    const origMat = originalMaterials.get(m);
    if (origMat && m.material) {
      m.material.color.copy(origMat.color);
      const dot = document.getElementById(`color-dot-${info.index}`);
      if (dot) dot.value = '#' + origMat.color.getHexString();
    }
  });
  document.getElementById('explode-slider').value = 0;
  currentState = States.Idle;
  isExploded = false;
  document.getElementById('explode-toggle').classList.remove('active');
  document.getElementById('explode-text').innerText = 'Explode';
};

// --- SHADING & VIEW MODES ---
function applyShadingMode(mode) {
  currentShadingMode = mode;
  document.querySelectorAll('.tool-btn-group button').forEach(b => {
    if (['mode-shaded', 'mode-wireframe', 'mode-normals', 'mode-xray'].includes(b.id)) {
      b.classList.remove('active');
    }
  });
  const btn = document.getElementById(`mode-${mode}`);
  if (btn) btn.classList.add('active');

  if (!currentLoadedMesh) return;

  currentLoadedMesh.traverse((child) => {
    if (child.isMesh) {
      const orig = originalMaterials.get(child);
      if (!orig) return;

      if (mode === 'shaded') {
        child.material = orig.clone();
        child.material.wireframe = false;
        child.material.transparent = false;
        child.material.opacity = 1.0;
      } else if (mode === 'wireframe') {
        child.material = orig.clone();
        child.material.wireframe = true;
      } else if (mode === 'normals') {
        child.material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
      } else if (mode === 'xray') {
        child.material = orig.clone();
        child.material.transparent = true;
        child.material.opacity = 0.35;
        child.material.depthWrite = false;
      }
    }
  });
}

document.getElementById('mode-shaded').onclick = () => applyShadingMode('shaded');
document.getElementById('mode-wireframe').onclick = () => applyShadingMode('wireframe');
document.getElementById('mode-normals').onclick = () => applyShadingMode('normals');
document.getElementById('mode-xray').onclick = () => applyShadingMode('xray');

// Camera Angle Presets
function setCameraView(view) {
  if (!currentLoadedMesh) return;
  const bBox = new THREE.Box3().setFromObject(currentLoadedMesh);
  const center = new THREE.Vector3();
  bBox.getCenter(center);
  const size = new THREE.Vector3();
  bBox.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z, 0.5);

  controls.target.copy(center);

  if (view === 'front') {
    camera.position.set(center.x, center.y, center.z + maxDim * 2.5);
  } else if (view === 'top') {
    camera.position.set(center.x, center.y + maxDim * 2.8, center.z + 0.001);
  } else if (view === 'side') {
    camera.position.set(center.x + maxDim * 2.5, center.y, center.z);
  } else if (view === 'iso') {
    camera.position.set(center.x + maxDim * 1.6, center.y + maxDim * 1.2, center.z + maxDim * 2.2);
  }
  controls.update();
}

document.getElementById('cam-front').onclick = () => setCameraView('front');
document.getElementById('cam-top').onclick = () => setCameraView('top');
document.getElementById('cam-side').onclick = () => setCameraView('side');
document.getElementById('cam-iso').onclick = () => setCameraView('iso');
document.getElementById('cam-reset').onclick = () => setCameraView('iso');
document.getElementById('cam-turntable').onclick = () => {
  controls.autoRotate = !controls.autoRotate;
  document.getElementById('cam-turntable').classList.toggle('active', controls.autoRotate);
};

// Studio Background Themes
function setStudioTheme(color, dotId) {
  scene.background = new THREE.Color(color);
  document.querySelectorAll('.theme-dot').forEach(d => d.classList.remove('active'));
  document.getElementById(dotId).classList.add('active');
}
document.getElementById('theme-dark').onclick = () => setStudioTheme('#080a0e', 'theme-dark');
document.getElementById('theme-slate').onclick = () => setStudioTheme('#1e293b', 'theme-slate');
document.getElementById('theme-light').onclick = () => setStudioTheme('#e2e8f0', 'theme-light');

document.getElementById('grid-toggle').onclick = () => {
  gridHelper.visible = !gridHelper.visible;
  document.getElementById('grid-toggle').classList.toggle('active', gridHelper.visible);
};

// --- TRIPO3D EXPLOSION ENGINE ---
const rotQuat = new THREE.Quaternion();

function applyExplosion(tValue, deltaSec) {
  const allowTumbling = document.getElementById('tumbling-toggle').checked;
  const isCollapsing = (currentState === States.Collapsing);
  const collapseEase = isCollapsing ? tripoCollapseEasing(animProgress) : 0;

  partsData.forEach((info, mesh) => {
    if (!mesh.visible) return;

    if (isCollapsing) {
      info.accumulatedRotation = info.snapshotRotation * (1 - collapseEase);
      rotQuat.setFromAxisAngle(info.rotationAxis, info.accumulatedRotation);
      rotQuat.premultiply(info.originalRotation);

      const rOrig = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(info.originalRotation);
      const rCur = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(rotQuat);
      const pivotComp = rOrig.sub(rCur);

      const rDist = maxDistance * (0.35 + 0.65 * Math.min(1, info.distanceWeight));
      const disp = info.explodeDirection.clone().multiplyScalar((1 - collapseEase) * rDist);

      mesh.position.copy(info.originalLocalPosition).add(pivotComp).add(disp).add(info.customOffset);
      mesh.quaternion.copy(rotQuat);
      return;
    }

    const tLocal = Math.max(0, Math.min(1, (tValue - (1 - info.distanceWeight) * 0.10) / 0.60));
    const easedProgress = tripoOvershootEasing(tLocal);

    if (tLocal >= 1.0 && allowTumbling) {
      info.accumulatedRotation += 0.02 * info.rotationSpeedFactor * (deltaSec * 60);
      rotQuat.setFromAxisAngle(info.rotationAxis, info.accumulatedRotation);
      rotQuat.premultiply(info.originalRotation);

      const rOrig = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(info.originalRotation);
      const rCur = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(rotQuat);
      const pivotComp = rOrig.sub(rCur);

      const rDist = maxDistance * (0.35 + 0.65 * Math.min(1, info.distanceWeight));
      const disp = info.explodeDirection.clone().multiplyScalar(rDist);

      mesh.position.copy(info.originalLocalPosition).add(pivotComp).add(disp).add(info.customOffset);
      mesh.quaternion.copy(rotQuat);
    } else {
      const rDist = maxDistance * (0.35 + 0.65 * Math.min(1, info.distanceWeight));
      const disp = info.explodeDirection.clone().multiplyScalar(easedProgress * rDist);
      mesh.position.copy(info.originalLocalPosition).add(disp).add(info.customOffset);
      mesh.quaternion.copy(info.originalRotation);
    }
  });
}

function updateAnimation(deltaSec) {
  if (currentState === States.Idle) return;

  const now = performance.now();
  const elapsed = now - animStartTime;
  const step = Math.min(elapsed / ANIM_DURATION, 1.0);

  if (currentState === States.Expanding) {
    animProgress = startProgress + (1.0 - startProgress) * step;
    document.getElementById('explode-slider').value = animProgress;
    applyExplosion(animProgress, deltaSec);
    if (step >= 1.0) {
      currentState = States.Rotating;
    }
  } else if (currentState === States.Rotating) {
    applyExplosion(1.0, deltaSec);
  } else if (currentState === States.Collapsing) {
    animProgress = startProgress * (1.0 - step);
    document.getElementById('explode-slider').value = animProgress;
    applyExplosion(animProgress, deltaSec);
    if (step >= 1.0) {
      currentState = States.Idle;
      partsData.forEach((info, mesh) => {
        mesh.position.copy(info.originalLocalPosition).add(info.customOffset);
        mesh.quaternion.copy(info.originalRotation);
        info.accumulatedRotation = 0;
      });
    }
  }
}

// Explode Controls Binding
const explodeBtn = document.getElementById('explode-toggle');
const explodeSlider = document.getElementById('explode-slider');
const explodeText = document.getElementById('explode-text');

explodeBtn.onclick = () => {
  if (partsData.size === 0) return;
  isExploded = !isExploded;
  explodeBtn.classList.toggle('active', isExploded);
  explodeText.innerText = isExploded ? 'Collapse' : 'Explode';
  animStartTime = performance.now();
  startProgress = parseFloat(explodeSlider.value);

  if (isExploded) {
    currentState = States.Expanding;
  } else {
    partsData.forEach((info) => {
      info.snapshotRotation = info.accumulatedRotation;
    });
    currentState = States.Collapsing;
  }
};

explodeSlider.oninput = (e) => {
  const val = parseFloat(e.target.value);
  currentState = States.Idle;
  isExploded = (val > 0.5);
  explodeBtn.classList.toggle('active', isExploded);
  explodeText.innerText = isExploded ? 'Collapse' : 'Explode';
  applyExplosion(val, 0.016);
};

// --- MODEL LOADING ---
const gltfLoader = new THREE.GLTFLoader();
const objLoader = new THREE.OBJLoader();
const plyLoader = new THREE.PLYLoader();

function loadModelFromUrl(url, title = null, precomputedStats = null) {
  currentModelPath = url;
  if (title) {
    document.getElementById('model-title-text').innerText = title;
  }

  const emptyPrompt = document.getElementById('empty-canvas-prompt');
  if (emptyPrompt) emptyPrompt.style.display = 'none';

  const ext = url.substring(url.lastIndexOf('.')).toLowerCase();

  const handleLoadedModel = (obj) => {
    if (currentLoadedMesh) {
      scene.remove(currentLoadedMesh);
    }
    currentLoadedMesh = obj;
    scene.add(currentLoadedMesh);
    initExplosionData(currentLoadedMesh);

    if (precomputedStats) {
      updateMeshStatsDisplay(precomputedStats);
    }

    applyShadingMode(currentShadingMode);

    // Reset controls
    isExploded = false;
    currentState = States.Idle;
    explodeBtn.classList.remove('active');
    explodeText.innerText = 'Explode';
    explodeSlider.value = 0;
  };

  if (ext === '.obj') {
    objLoader.load(url, handleLoadedModel);
  } else if (ext === '.ply') {
    plyLoader.load(url, (geom) => {
      geom.computeVertexNormals();
      const mat = new THREE.MeshStandardMaterial({ color: 0x818cf8, roughness: 0.4 });
      const mesh = new THREE.Mesh(geom, mat);
      const group = new THREE.Group();
      group.add(mesh);
      handleLoadedModel(group);
    });
  } else {
    gltfLoader.load(url, (gltf) => {
      handleLoadedModel(gltf.scene);
    }, undefined, (err) => {
      console.error('Error loading GLB:', err);
    });
  }
}

// --- FILE UPLOAD & DRAG AND DROP ---
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const sidebarDropzone = document.getElementById('sidebar-dropzone');
const dragOverlay = document.getElementById('drag-overlay');
const promptUploadBtn = document.getElementById('prompt-upload-btn');

if (uploadBtn) uploadBtn.onclick = () => fileInput.click();
if (sidebarDropzone) sidebarDropzone.onclick = () => fileInput.click();
if (promptUploadBtn) promptUploadBtn.onclick = () => fileInput.click();

async function handleFileUpload(file) {
  if (!file) return;
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  if (!['.glb', '.obj', '.ply'].includes(ext)) {
    alert(`Unsupported file format "${ext}". Please upload .glb, .obj, or .ply.`);
    return;
  }

  const banner = document.getElementById('progress-banner');
  const stageText = document.getElementById('progress-stage-text');
  const percentText = document.getElementById('progress-percent-text');
  const barFill = document.getElementById('progress-bar-fill');

  banner.style.display = 'flex';
  stageText.innerText = `Uploading & parsing ${file.name}...`;
  percentText.innerText = '40%';
  barFill.style.width = '40%';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.status === 'success') {
      stageText.innerText = 'Inspecting Geometry Topology...';
      percentText.innerText = '100%';
      barFill.style.width = '100%';
      setTimeout(() => { banner.style.display = 'none'; }, 600);

      loadModelFromUrl(data.path, file.name, data.stats);
    } else {
      banner.style.display = 'none';
      alert(data.error || 'Upload failed');
    }
  } catch (err) {
    banner.style.display = 'none';
    alert('Upload error: ' + err);
  }
}

fileInput.onchange = (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileUpload(e.target.files[0]);
  }
};

window.addEventListener('dragover', (e) => {
  e.preventDefault();
  dragOverlay.classList.add('active');
});

window.addEventListener('dragleave', (e) => {
  e.preventDefault();
  if (e.clientX <= 0 || e.clientY <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight) {
    dragOverlay.classList.remove('active');
  }
});

window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragOverlay.classList.remove('active');
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFileUpload(e.dataTransfer.files[0]);
  }
});

// --- RUN PIPELINE WITH LIVE TIMING STOPWATCH ---
let stopwatchInterval = null;
let pipelineStartTime = 0;

function startLiveStopwatch() {
  pipelineStartTime = performance.now();
  const badge = document.getElementById('live-stopwatch');
  badge.style.color = '#10b981';
  badge.style.boxShadow = '0 0 12px rgba(16, 185, 129, 0.4)';

  if (stopwatchInterval) clearInterval(stopwatchInterval);
  stopwatchInterval = setInterval(() => {
    const elapsedSec = (performance.now() - pipelineStartTime) / 1000;
    const mins = Math.floor(elapsedSec / 60);
    const secs = (elapsedSec % 60).toFixed(1);
    badge.innerText = `⏱️ ${mins.toString().padStart(2, '0')}:${secs.padStart(4, '0')}s`;
  }, 100);
}

function stopLiveStopwatch(finalFormatted) {
  if (stopwatchInterval) clearInterval(stopwatchInterval);
  const badge = document.getElementById('live-stopwatch');
  badge.style.color = '#fff';
  badge.style.boxShadow = 'none';
  if (finalFormatted) {
    badge.innerText = `⚡ ${finalFormatted}`;
  }
}

document.getElementById('run-pipeline-btn').onclick = async () => {
  if (!currentModelPath) {
    alert('Please load or upload a 3D model first!');
    return;
  }

  const engineMode = document.getElementById('engine-mode-select').value;
  const mcAlgo = document.getElementById('mc-algo-select').value;
  const resolution = parseInt(document.getElementById('res-select').value);

  const banner = document.getElementById('progress-banner');
  const stageText = document.getElementById('progress-stage-text');
  const percentText = document.getElementById('progress-percent-text');
  const barFill = document.getElementById('progress-bar-fill');
  banner.style.display = 'flex';

  // Reset timing fields
  document.getElementById('time-step1').innerText = '-- s';
  document.getElementById('time-step2').innerText = '-- s';
  document.getElementById('time-step3').innerText = '-- s';
  document.getElementById('time-total').innerText = '-- s';
  startLiveStopwatch();

  try {
    const res = await fetch('/api/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mesh_path: currentModelPath,
        mode: engineMode,
        mc_algo: mcAlgo,
        steps: 50,
        resolution: resolution,
        explosion_scale: 0.35
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const jobId = data.job_id;

    // Poll status
    const pollInterval = setInterval(async () => {
      try {
        const sRes = await fetch(`/api/status/${jobId}`);
        const sData = await sRes.json();

        stageText.innerText = sData.stage || 'Processing...';
        percentText.innerText = `${sData.progress || 0}%`;
        barFill.style.width = `${sData.progress || 0}%`;

        // Update step times if present
        if (sData.timings) {
          if (sData.timings.step1 > 0) document.getElementById('time-step1').innerText = `${sData.timings.step1}s`;
          if (sData.timings.step2 > 0) document.getElementById('time-step2').innerText = `${sData.timings.step2}s`;
        }

        if (sData.status === 'completed') {
          clearInterval(pollInterval);
          setTimeout(() => { banner.style.display = 'none'; }, 600);

          if (sData.result && sData.result.timing) {
            const t = sData.result.timing;
            document.getElementById('time-step1').innerText = `${t.step1}s`;
            document.getElementById('time-step2').innerText = `${t.step2}s`;
            document.getElementById('time-step3').innerText = `${t.step3}s`;
            document.getElementById('time-total').innerText = `${t.formatted}`;
            stopLiveStopwatch(t.formatted);
          } else {
            stopLiveStopwatch();
          }

          // Load generated multi-node assembly
          loadModelFromUrl(
            sData.result.assembly_glb,
            `${document.getElementById('model-title-text').innerText} (Segmented)`,
            sData.result.mesh_details
          );

          // Configure download buttons
          document.getElementById('download-assembly-btn').onclick = () => window.open(sData.result.assembly_glb, '_blank');
          document.getElementById('download-exploded-btn').onclick = () => window.open(sData.result.exploded_glb, '_blank');
        } else if (sData.status === 'error') {
          clearInterval(pollInterval);
          stopLiveStopwatch();
          banner.style.display = 'none';
          alert('Pipeline error: ' + sData.error);
        }
      } catch (err) {
        clearInterval(pollInterval);
        stopLiveStopwatch();
        banner.style.display = 'none';
        console.error('Polling error:', err);
      }
    }, 800);

  } catch (err) {
    stopLiveStopwatch();
    banner.style.display = 'none';
    alert('Pipeline launch error: ' + err.message);
  }
};

// Window Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Render Loop
let clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  controls.update();
  updateAnimation(delta);
  renderer.render(scene, camera);
}
animate();

// Initial load: auto-load furn16_3k if available
loadModelFromUrl('/outputs/furn16_3k_tripo_exact/train_cfg_-1.000000_dit_boxgpt_furn16_3k_assembly.glb', 'furn16_3k Furniture (Tripo3D Exact)');

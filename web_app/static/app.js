// ==============================================================================
// Tripo3D Matching Web Studio - Production 3D Engine & Blender Editing Suite
// Supports Universal Scale Auto-Framing, PBR Paint Preservation, Studio Lighting,
// Blender Transformations, and 60 FPS Video Recording.
// ==============================================================================

// --- THREE.JS SCENE SETUP ---
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#080a0e');

// Universal Camera (Dynamically Adjusted Near/Far)
let camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 2000);
camera.position.set(2.8, 2.0, 3.4);

let isOrtho = false;
let orthoCamera = null;

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: false,
  powerPreference: 'high-performance',
  preserveDrawingBuffer: true // Required for high-res snapshots and video recording
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;
renderer.outputEncoding = THREE.sRGBEncoding;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = false;
controls.autoRotateSpeed = 2.0;

// --- STUDIO LIGHTING RIG (Blender Standard) ---
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const hemiLight = new THREE.HemisphereLight(0xf8fafc, 0x1e293b, 0.8);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
keyLight.position.set(6, 12, 8);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x818cf8, 0.7);
fillLight.position.set(-6, -3, -6);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xa78bfa, 0.5);
rimLight.position.set(0, 8, -8);
scene.add(rimLight);

let baseLightIntensities = {
  ambient: 0.6,
  hemi: 0.8,
  key: 1.4,
  fill: 0.7,
  rim: 0.5
};

// --- DYNAMIC GROUND GRID ---
let gridHelper = new THREE.GridHelper(10, 20, 0x475569, 0x1e293b);
gridHelper.position.y = 0;
scene.add(gridHelper);

// --- STATE MANAGEMENT ---
let currentLoadedMesh = null;
let currentModelPath = null;
let currentShadingMode = 'shaded';
let selectedPartMesh = null;
let isolatedPartMesh = null;
let partsData = new Map();
let originalMaterials = new Map();
let paletteMaterials = new Map();
let clayMaterials = new Map();
let globalCenter = new THREE.Vector3();
let modelBoundingBox = new THREE.Box3();
let maxDistance = 1.0;
let isExploded = false;
let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];
let recordingTimer = null;
let recordStartTime = 0;

// Tripo3D Vibrant Color Palette
const TRIPO_PALETTE = [
  '#1d39a6', '#14dde4', '#f5a623', '#e4405f', '#20a530', '#9b51e0',
  '#f2784b', '#409eff', '#ff6384', '#4bc0c0', '#ffcd56', '#9966ff',
  '#36a2eb', '#ff9f40', '#2ecc71', '#e74c3c', '#3498db', '#f1c40f',
  '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6', '#34495e', '#d35400'
];

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

// --- BULLETPROOF MATERIAL CLONER ---
function cloneMaterialSafe(mat) {
  if (!mat) return new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.5 });
  if (Array.isArray(mat)) {
    return mat.map(m => cloneMaterialSafe(m));
  }
  const cloned = mat.clone();
  cloned.side = THREE.DoubleSide;
  cloned.depthWrite = true;
  if (cloned.opacity < 0.1 && !cloned.map) {
    cloned.opacity = 1.0;
    cloned.transparent = false;
  }
  return cloned;
}

// --- MESH STATS CALCULATION ---
function calculateGeometryStats(model) {
  let triangles = 0;
  let vertices = 0;
  let faces = 0;
  let quads = 0;
  const bBox = new THREE.Box3().setFromObject(model);

  model.traverse((child) => {
    if (child.isMesh && child.geometry) {
      const geom = child.geometry;
      let triCount = 0;
      if (geom.index) {
        triCount = geom.index.count / 3;
      } else if (geom.attributes && geom.attributes.position) {
        triCount = geom.attributes.position.count / 3;
      }
      triangles += Math.round(triCount);
      faces += Math.round(triCount);
      vertices += (geom.attributes && geom.attributes.position) ? geom.attributes.position.count : 0;
    }
  });

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

// --- UNIVERSAL AUTO-FRAMING & AUTO-CENTERING FOR ALL TYPES & DIMENSIONS ---
function autoFrameModel(model) {
  model.updateMatrixWorld(true);
  let box = new THREE.Box3().setFromObject(model);

  if (box.isEmpty() || !isFinite(box.min.x)) {
    console.warn('Bounding box was empty or non-finite, calculating from mesh geometries...');
    box = new THREE.Box3();
    model.traverse((child) => {
      if (child.geometry) {
        if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
        if (child.geometry.boundingBox) {
          const geomBox = child.geometry.boundingBox.clone();
          geomBox.applyMatrix4(child.matrixWorld);
          box.union(geomBox);
        }
      }
    });
  }

  if (box.isEmpty() || !isFinite(box.min.x)) {
    box.set(new THREE.Vector3(-1, 0, -1), new THREE.Vector3(1, 2, 1));
  }

  modelBoundingBox.copy(box);
  const size = new THREE.Vector3();
  box.getSize(size);
  const center = new THREE.Vector3();
  box.getCenter(center);

  const maxDim = Math.max(size.x, size.y, size.z, 0.001);

  // Idempotent shift: move horizontal center to (0, 0) and bottom base to Y = 0
  model.position.x -= center.x;
  model.position.y -= box.min.y;
  model.position.z -= center.z;
  model.updateMatrixWorld(true);

  // Recompute world-space box after centering
  const centeredBox = new THREE.Box3().setFromObject(model);
  centeredBox.getCenter(globalCenter);

  // Dynamic Camera Near / Far to prevent clipping on ANY scale from 0.0001 to 10,000,000 units
  camera.near = Math.max(maxDim * 0.0005, 0.0001);
  camera.far = Math.max(maxDim * 200, 5000);
  camera.updateProjectionMatrix();

  // Optimal 45-degree isometric framing distance
  const fovRad = (camera.fov * Math.PI) / 180;
  const cameraDist = (maxDim / 2) / Math.tan(fovRad / 2) * 1.8;

  const targetY = size.y * 0.5;
  camera.position.set(cameraDist * 0.8, targetY + cameraDist * 0.5, cameraDist * 1.1);
  controls.target.set(0, targetY, 0);
  controls.minDistance = maxDim * 0.005;
  controls.maxDistance = maxDim * 50;
  controls.update();

  // Dynamic Studio Lights repositioned to illuminate the model brilliantly
  keyLight.position.set(maxDim * 1.5, size.y + maxDim * 2.0, maxDim * 2.0);
  fillLight.position.set(-maxDim * 1.5, targetY - maxDim * 0.5, -maxDim * 1.5);
  rimLight.position.set(0, size.y + maxDim * 1.8, -maxDim * 2.2);

  // Dynamic adaptive grid beneath model
  scene.remove(gridHelper);
  const gridSize = Math.max(maxDim * 2.5, 2);
  gridHelper = new THREE.GridHelper(gridSize, 20, 0x475569, 0x1e293b);
  gridHelper.position.set(0, 0, 0);
  scene.add(gridHelper);
}

// --- EXPLOSION & PART REGISTRATION ---
function initExplosionData(model) {
  autoFrameModel(model);

  const meshes = [];
  originalMaterials.clear();
  paletteMaterials.clear();
  clayMaterials.clear();

  // Collect all valid visual objects (safely traversing multi-node hierarchies)
  model.traverse((child) => {
    // Disable any imported lights in GLB so they don't break our studio lighting
    if (child.isLight && child !== ambientLight && child !== hemiLight && child !== keyLight && child !== fillLight && child !== rimLight) {
      child.intensity = 0;
      child.visible = false;
    }

    // Force visibility and prevent camera frustum clipping
    child.visible = true;
    if (child.isMesh || child.isPoints || child.isLine) {
      child.frustumCulled = false;
    }

    if (child.isMesh && child.geometry) {
      // Ensure normals exist so lighting calculations never result in black silhouettes
      if (!child.geometry.attributes.normal) {
        child.geometry.computeVertexNormals();
      }

      meshes.push(child);

      // Safe material processing
      if (child.material) {
        const processMat = (m) => {
          m.side = THREE.DoubleSide;
          m.depthWrite = true;
          // Fix zero or NaN opacity
          if (isNaN(m.opacity) || m.opacity < 0.05) {
            m.opacity = 1.0;
            m.transparent = false;
          }
          // Avoid black hole materials when untextured
          if (m.color && m.color.r === 0 && m.color.g === 0 && m.color.b === 0 && !m.map) {
            m.color.setHex(0x333333);
          }
        };

        if (Array.isArray(child.material)) {
          child.material.forEach(processMat);
          originalMaterials.set(child, child.material.map(m => cloneMaterialSafe(m)));
        } else {
          processMat(child.material);
          originalMaterials.set(child, cloneMaterialSafe(child.material));
        }
      }

      // Pre-create Studio Clay material
      clayMaterials.set(child, new THREE.MeshStandardMaterial({
        color: 0xd4d4d8,
        roughness: 0.45,
        metalness: 0.05,
        emissive: 0x222225,
        emissiveIntensity: 0.1,
        side: THREE.DoubleSide
      }));
    }
  });

  // Calculate max explosion distance from center
  maxDistance = 0;
  meshes.forEach((mesh) => {
    const pos = new THREE.Vector3();
    mesh.getWorldPosition(pos);
    const dist = pos.distanceTo(globalCenter);
    if (dist > maxDistance) maxDistance = dist;
  });
  if (maxDistance < 1e-4) maxDistance = 1.0;

  // Register individual parts
  partsData.clear();
  meshes.forEach((mesh, idx) => {
    const worldPos = new THREE.Vector3();
    mesh.getWorldPosition(worldPos);

    const lVec = worldPos.clone().sub(globalCenter);
    let dist = lVec.length();
    if (dist < 1e-4) {
      lVec.set((Math.random() - 0.5) * 0.02, (Math.random() - 0.5) * 0.02, (Math.random() - 0.5) * 0.02);
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
      name: mesh.name || `tripo_part_${idx}`,
      originalLocalPosition: mesh.position.clone(),
      originalRotation: mesh.quaternion.clone(),
      originalScale: mesh.scale.clone(),
      explodeDirection: lVec.clone(),
      distanceWeight: dist / maxDistance,
      geometryCenterOffset: geomCenter.clone(),
      accumulatedRotation: 0,
      rotationAxis: rotAxis,
      rotationSpeedFactor: rotSpeedFactor,
      snapshotRotation: 0,
      customOffset: new THREE.Vector3(),
      customRotation: new THREE.Euler(),
      customScale: 1.0,
      visible: true
    };
    partsData.set(mesh, partInfo);

    // AI Palette material
    const palHex = TRIPO_PALETTE[idx % TRIPO_PALETTE.length];
    paletteMaterials.set(mesh, new THREE.MeshStandardMaterial({
      color: palHex,
      roughness: 0.35,
      metalness: 0.05,
      emissive: palHex,
      emissiveIntensity: 0.12,
      side: THREE.DoubleSide
    }));
  });

  rebuildPartsHierarchyUI();
}

// --- DYNAMIC PARTS HIERARCHY & DROPDOWN UI REBUILDER ---
function rebuildPartsHierarchyUI() {
  const partsContainer = document.getElementById('parts-container');
  if (!partsContainer) return;
  partsContainer.innerHTML = '';

  const partCountEl = document.getElementById('part-count');
  if (partCountEl) partCountEl.innerText = `${partsData.size} PARTS`;
  const badgeCountEl = document.getElementById('parts-badge-count');
  if (badgeCountEl) badgeCountEl.innerText = partsData.size;
  const statParts = document.getElementById('stat-parts-count');
  if (statParts) statParts.innerText = partsData.size > 1 ? `${partsData.size} Disjoint Parts` : '1 Single Object';

  // Populate Quick Part Picker Dropdown
  const quickSelect = document.getElementById('quick-part-select');
  if (quickSelect) {
    quickSelect.innerHTML = '<option value="">-- Choose Part to Inspect & Edit --</option>';
    let i = 0;
    partsData.forEach((partInfo, m) => {
      const opt = document.createElement('option');
      opt.value = partInfo.index;
      opt.textContent = `Part #${i + 1}: ${partInfo.name}`;
      quickSelect.appendChild(opt);
      i++;
    });
    quickSelect.onchange = (e) => {
      const val = e.target.value;
      if (val !== '') {
        const targetIdx = parseInt(val);
        for (const [m, info] of partsData.entries()) {
          if (info.index === targetIdx) {
            selectPart(m);
            const card = document.getElementById(`part-card-${targetIdx}`);
            if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            break;
          }
        }
      }
    };
  }

  // Populate Part Cards in Hierarchy Drawer
  partsData.forEach((partInfo, mesh) => {
    const idx = partInfo.index;
    let hexColor = TRIPO_PALETTE[idx % TRIPO_PALETTE.length];
    if (mesh.material) {
      const m = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      if (m && m.color) hexColor = '#' + m.color.getHexString();
    }

    let partTris = mesh.geometry.index ? Math.round(mesh.geometry.index.count / 3) : Math.round(mesh.geometry.attributes.position.count / 3);
    let partVerts = mesh.geometry.attributes.position ? mesh.geometry.attributes.position.count : 0;
    let pSize = new THREE.Vector3();
    if (mesh.geometry.boundingBox) mesh.geometry.boundingBox.getSize(pSize);

    const card = document.createElement('div');
    card.className = 'part-item';
    card.id = `part-card-${idx}`;
    card.innerHTML = `
      <div class="part-main-row">
        <div class="part-badge">
          <input type="color" value="${hexColor}" class="part-color-dot" id="color-dot-${idx}" title="Click to Change Part Color">
          <span class="part-name-text">${partInfo.name}</span>
        </div>
        <div style="display: flex; gap: 4px;">
          <button class="icon-tool-btn" id="solo-btn-${idx}" title="Solo / Isolate Part">🎯</button>
          <button class="icon-tool-btn" id="vis-btn-${idx}" title="Toggle Visibility">👁️</button>
        </div>
      </div>
      <div class="part-sub-row">
        <span>▲ ${partTris.toLocaleString()} tris</span>
        <span>◆ ${partVerts.toLocaleString()} verts</span>
        <span>${pSize.x.toFixed(2)}x${pSize.y.toFixed(2)}m</span>
      </div>
    `;

    card.onclick = (e) => {
      if (e.target.tagName.toLowerCase() === 'input' || e.target.tagName.toLowerCase() === 'button') return;
      selectPart(mesh);
    };

    const colInput = card.querySelector(`#color-dot-${idx}`);
    colInput.oninput = (e) => {
      const newHex = e.target.value;
      if (mesh.material) {
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach(m => { if (m.color) m.color.set(newHex); });
        } else if (mesh.material.color) {
          mesh.material.color.set(newHex);
        }
      }
    };

    const soloBtn = card.querySelector(`#solo-btn-${idx}`);
    soloBtn.onclick = (e) => {
      e.stopPropagation();
      toggleIsolatePart(mesh);
    };

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

  if (currentLoadedMesh) {
    const modelStats = calculateGeometryStats(currentLoadedMesh);
    updateMeshStatsDisplay(modelStats);
  }
}

// --- PART SELECTION & BLENDER TRANSFORM EDITING ---
function selectPart(mesh) {
  selectedPartMesh = mesh;
  const info = partsData.get(mesh);
  if (!info) return;

  const quickSelect = document.getElementById('quick-part-select');
  if (quickSelect) quickSelect.value = info.index;

  document.querySelectorAll('.part-item').forEach(c => c.classList.remove('active'));
  const card = document.getElementById(`part-card-${info.index}`);
  if (card) card.classList.add('active');

  const editorBox = document.getElementById('part-editor-card');
  editorBox.style.display = 'block';
  document.getElementById('editor-part-title').innerText = info.name;

  let mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  if (mat && mat.color) {
    document.getElementById('editor-color-picker').value = '#' + mat.color.getHexString();
  }
  if (mat) {
    if (mat.roughness !== undefined) {
      document.getElementById('editor-roughness').value = mat.roughness;
      document.getElementById('val-roughness').innerText = mat.roughness.toFixed(2);
    }
    if (mat.metalness !== undefined) {
      document.getElementById('editor-metalness').value = mat.metalness;
      document.getElementById('val-metalness').innerText = mat.metalness.toFixed(2);
    }
  }

  // Update slider positions
  document.getElementById('editor-nudge-x').value = info.customOffset.x;
  document.getElementById('val-nudge-x').innerText = info.customOffset.x.toFixed(2);
  document.getElementById('editor-nudge-y').value = info.customOffset.y;
  document.getElementById('val-nudge-y').innerText = info.customOffset.y.toFixed(2);
  document.getElementById('editor-nudge-z').value = info.customOffset.z;
  document.getElementById('val-nudge-z').innerText = info.customOffset.z.toFixed(2);

  document.getElementById('editor-rot-y').value = Math.round((info.customRotation.y * 180) / Math.PI);
  document.getElementById('val-rot-y').innerText = `${Math.round((info.customRotation.y * 180) / Math.PI)}°`;

  document.getElementById('editor-scale').value = info.customScale;
  document.getElementById('val-scale').innerText = `${info.customScale.toFixed(2)}x`;
}

document.getElementById('editor-close-btn').onclick = () => {
  document.getElementById('part-editor-card').style.display = 'none';
  document.querySelectorAll('.part-item').forEach(c => c.classList.remove('active'));
  selectedPartMesh = null;
};

// Blender Nudge X, Y, Z
['x', 'y', 'z'].forEach(axis => {
  const slider = document.getElementById(`editor-nudge-${axis}`);
  if (slider) {
    slider.oninput = (e) => {
      if (!selectedPartMesh) return;
      const info = partsData.get(selectedPartMesh);
      if (!info) return;
      const val = parseFloat(e.target.value);
      info.customOffset[axis] = val;
      document.getElementById(`val-nudge-${axis}`).innerText = val.toFixed(2);
      applyPartTransforms(info);
    };
  }
});

// Blender Rotation & Scale
document.getElementById('editor-rot-y').oninput = (e) => {
  if (!selectedPartMesh) return;
  const info = partsData.get(selectedPartMesh);
  if (!info) return;
  const deg = parseFloat(e.target.value);
  info.customRotation.y = (deg * Math.PI) / 180;
  document.getElementById('val-rot-y').innerText = `${deg}°`;
  applyPartTransforms(info);
};

document.getElementById('editor-scale').oninput = (e) => {
  if (!selectedPartMesh) return;
  const info = partsData.get(selectedPartMesh);
  if (!info) return;
  const val = parseFloat(e.target.value);
  info.customScale = val;
  document.getElementById('val-scale').innerText = `${val.toFixed(2)}x`;
  applyPartTransforms(info);
};

// Material Roughness & Metalness
document.getElementById('editor-roughness').oninput = (e) => {
  if (!selectedPartMesh) return;
  const val = parseFloat(e.target.value);
  document.getElementById('val-roughness').innerText = val.toFixed(2);
  const applyRough = (m) => { if (m && m.roughness !== undefined) m.roughness = val; };
  if (Array.isArray(selectedPartMesh.material)) selectedPartMesh.material.forEach(applyRough);
  else applyRough(selectedPartMesh.material);
};

document.getElementById('editor-metalness').oninput = (e) => {
  if (!selectedPartMesh) return;
  const val = parseFloat(e.target.value);
  document.getElementById('val-metalness').innerText = val.toFixed(2);
  const applyMetal = (m) => { if (m && m.metalness !== undefined) m.metalness = val; };
  if (Array.isArray(selectedPartMesh.material)) selectedPartMesh.material.forEach(applyMetal);
  else applyMetal(selectedPartMesh.material);
};

document.getElementById('editor-color-picker').oninput = (e) => {
  if (!selectedPartMesh) return;
  const newHex = e.target.value;
  const applyColor = (m) => { if (m && m.color) m.color.set(newHex); };
  if (Array.isArray(selectedPartMesh.material)) selectedPartMesh.material.forEach(applyColor);
  else applyColor(selectedPartMesh.material);

  const info = partsData.get(selectedPartMesh);
  if (info) {
    const dot = document.getElementById(`color-dot-${info.index}`);
    if (dot) dot.value = newHex;
  }
};

document.getElementById('editor-isolate-btn').onclick = () => {
  if (selectedPartMesh) toggleIsolatePart(selectedPartMesh);
};

document.getElementById('editor-hide-btn').onclick = () => {
  if (!selectedPartMesh) return;
  const info = partsData.get(selectedPartMesh);
  if (!info) return;
  info.visible = !info.visible;
  selectedPartMesh.visible = info.visible;
  const visBtn = document.getElementById(`vis-btn-${info.index}`);
  if (visBtn) {
    visBtn.innerText = info.visible ? '👁️' : '🚫';
    visBtn.style.opacity = info.visible ? '1' : '0.5';
  }
};

document.getElementById('editor-reset-btn').onclick = () => {
  if (!selectedPartMesh) return;
  const info = partsData.get(selectedPartMesh);
  if (!info) return;
  info.customOffset.set(0, 0, 0);
  info.customRotation.set(0, 0, 0);
  info.customScale = 1.0;
  selectPart(selectedPartMesh);
  applyPartTransforms(info);
};

function applyPartTransforms(info) {
  const m = info.mesh;
  m.scale.copy(info.originalScale).multiplyScalar(info.customScale);
  m.position.copy(info.originalLocalPosition).add(info.customOffset);
  const euler = new THREE.Euler().setFromQuaternion(info.originalRotation);
  euler.y += info.customRotation.y;
  m.quaternion.setFromEuler(euler);
}

function toggleIsolatePart(mesh) {
  if (isolatedPartMesh === mesh) {
    isolatedPartMesh = null;
    partsData.forEach((info, m) => { m.visible = info.visible; });
  } else {
    isolatedPartMesh = mesh;
    partsData.forEach((info, m) => { m.visible = (m === mesh); });
  }
}

document.getElementById('show-all-parts-btn').onclick = () => {
  isolatedPartMesh = null;
  partsData.forEach((info, m) => {
    info.visible = true;
    m.visible = true;
    const btn = document.getElementById(`vis-btn-${info.index}`);
    if (btn) {
      btn.innerText = '👁️';
      btn.style.opacity = '1';
    }
  });
};

document.getElementById('reset-parts-btn').onclick = () => {
  isolatedPartMesh = null;
  partsData.forEach((info, m) => {
    info.visible = true;
    m.visible = true;
    info.customOffset.set(0, 0, 0);
    info.customRotation.set(0, 0, 0);
    info.customScale = 1.0;
    applyPartTransforms(info);
  });
  document.getElementById('explode-slider').value = 0;
  currentState = States.Idle;
  isExploded = false;
  document.getElementById('explode-toggle').classList.remove('active');
  document.getElementById('explode-text').innerText = 'Explode';
};

// --- SHADING MODES (PBR TEXTURED, AI PALETTE, CLAY, WIREFRAME, NORMALS, XRAY) ---
function applyShadingMode(mode) {
  currentShadingMode = mode;
  ['mode-shaded', 'mode-palette', 'mode-clay', 'mode-wireframe', 'mode-normals', 'mode-xray'].forEach(id => {
    const b = document.getElementById(id);
    if (b) b.classList.remove('active');
  });
  const btn = document.getElementById(`mode-${mode}`);
  if (btn) btn.classList.add('active');

  if (!currentLoadedMesh) return;

  currentLoadedMesh.traverse((child) => {
    if (child.isMesh) {
      if (mode === 'shaded') {
        const orig = originalMaterials.get(child);
        if (orig) child.material = cloneMaterialSafe(orig);
      } else if (mode === 'palette') {
        const pal = paletteMaterials.get(child);
        if (pal) child.material = pal.clone();
      } else if (mode === 'clay') {
        const clay = clayMaterials.get(child);
        if (clay) child.material = clay.clone();
      } else if (mode === 'wireframe') {
        const orig = originalMaterials.get(child);
        child.material = orig ? cloneMaterialSafe(orig) : new THREE.MeshStandardMaterial({ color: 0x818cf8 });
        if (Array.isArray(child.material)) child.material.forEach(m => m.wireframe = true);
        else child.material.wireframe = true;
      } else if (mode === 'normals') {
        child.material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
      } else if (mode === 'xray') {
        const orig = originalMaterials.get(child);
        child.material = orig ? cloneMaterialSafe(orig) : new THREE.MeshStandardMaterial();
        const setXray = (m) => {
          m.transparent = true;
          m.opacity = 0.35;
          m.depthWrite = false;
        };
        if (Array.isArray(child.material)) child.material.forEach(setXray);
        else setXray(child.material);
      }
    }
  });
}

document.getElementById('mode-shaded').onclick = () => applyShadingMode('shaded');
document.getElementById('mode-palette').onclick = () => applyShadingMode('palette');
document.getElementById('mode-clay').onclick = () => applyShadingMode('clay');
document.getElementById('mode-wireframe').onclick = () => applyShadingMode('wireframe');
document.getElementById('mode-normals').onclick = () => applyShadingMode('normals');
document.getElementById('mode-xray').onclick = () => applyShadingMode('xray');

// --- BLENDER CAMERA & FOV CONTROLS ---
function setCameraView(view) {
  if (!currentLoadedMesh) return;
  const box = new THREE.Box3().setFromObject(currentLoadedMesh);
  const size = new THREE.Vector3();
  box.getSize(size);
  const center = new THREE.Vector3();
  box.getCenter(center);
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
document.getElementById('cam-reset').onclick = () => {
  if (currentLoadedMesh) autoFrameModel(currentLoadedMesh);
};

// Perspective / Orthographic Toggle (Blender Numpad 5)
document.getElementById('cam-proj').onclick = () => {
  isOrtho = !isOrtho;
  document.getElementById('cam-proj').innerText = isOrtho ? 'Ortho' : 'Persp';
  document.getElementById('cam-proj').classList.toggle('active', isOrtho);

  const aspect = window.innerWidth / window.innerHeight;
  const dist = camera.position.distanceTo(controls.target);

  if (isOrtho) {
    const frustumSize = dist * 0.8;
    orthoCamera = new THREE.OrthographicCamera(
      (frustumSize * aspect) / -2,
      (frustumSize * aspect) / 2,
      frustumSize / 2,
      frustumSize / -2,
      camera.near,
      camera.far
    );
    orthoCamera.position.copy(camera.position);
    orthoCamera.quaternion.copy(camera.quaternion);
    controls.object = orthoCamera;
  } else {
    controls.object = camera;
  }
  controls.update();
};

// Focal Length / FOV Selector
document.getElementById('cam-fov-select').onchange = (e) => {
  camera.fov = parseFloat(e.target.value);
  camera.updateProjectionMatrix();
};

// Turntable Auto-Orbit
const turntableBtn = document.getElementById('cam-turntable');
turntableBtn.onclick = () => {
  controls.autoRotate = !controls.autoRotate;
  turntableBtn.classList.toggle('active', controls.autoRotate);
};

// --- BLENDER STUDIO SETTINGS PANEL ---
const studioPanel = document.getElementById('studio-settings-panel');
document.getElementById('studio-toggle-btn').onclick = () => {
  studioPanel.style.display = studioPanel.style.display === 'none' ? 'block' : 'none';
};
document.getElementById('studio-close-btn').onclick = () => {
  studioPanel.style.display = 'none';
};

// Lighting Presets
document.getElementById('light-preset-select').onchange = (e) => {
  const preset = e.target.value;
  if (preset === 'studio') {
    keyLight.color.setHex(0xffffff);
    fillLight.color.setHex(0x818cf8);
    rimLight.color.setHex(0xa78bfa);
  } else if (preset === 'warm') {
    keyLight.color.setHex(0xffedd5);
    fillLight.color.setHex(0xf97316);
    rimLight.color.setHex(0xfde047);
  } else if (preset === 'cyber') {
    keyLight.color.setHex(0x06b6d4);
    fillLight.color.setHex(0xd946ef);
    rimLight.color.setHex(0x3b82f6);
  } else if (preset === 'contrast') {
    keyLight.color.setHex(0xffffff);
    fillLight.color.setHex(0x334155);
    rimLight.color.setHex(0x0f172a);
  } else if (preset === 'flat') {
    keyLight.color.setHex(0xffffff);
    fillLight.color.setHex(0xffffff);
    rimLight.color.setHex(0xffffff);
  }
};

// Light Intensity Slider
document.getElementById('light-intensity-slider').oninput = (e) => {
  const factor = parseFloat(e.target.value);
  document.getElementById('val-light-intensity').innerText = `${factor.toFixed(1)}x`;
  ambientLight.intensity = baseLightIntensities.ambient * factor;
  hemiLight.intensity = baseLightIntensities.hemi * factor;
  keyLight.intensity = baseLightIntensities.key * factor;
  fillLight.intensity = baseLightIntensities.fill * factor;
  rimLight.intensity = baseLightIntensities.rim * factor;
};

// Background Theme Chips
const bgThemes = {
  'bg-blender': '#282828',
  'bg-black': '#080a0e',
  'bg-slate': '#1e293b',
  'bg-white': '#f1f5f9'
};

Object.keys(bgThemes).forEach(id => {
  const btn = document.getElementById(id);
  if (btn) {
    btn.onclick = () => {
      document.querySelectorAll('.theme-chip').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      scene.background.set(bgThemes[id]);
    };
  }
});

// Floor Grid Toggle
document.getElementById('grid-toggle').onclick = (e) => {
  gridHelper.visible = !gridHelper.visible;
  document.getElementById('grid-toggle').classList.toggle('active', gridHelper.visible);
};

// --- INTERACTIVE 3D BRUSH SURFACE SELECTION & SEPARATE PART EXTRACTION ---
let isBrushActive = false;
let isBrushing = false;
let brushLockedMesh = null; // Anti-bleed lock: locks onto initial part touched
let brushMode = 'paint';    // 'paint' (Brush Area) | 'flood' (Flood Feature)
let brushScope = 'part';    // 'part' (Lock to Target Part - No Bleed) | 'multi' (All Parts)
let brushColor = '#f43f5e';
let brushRadius = 0.10;
const selectedFaceIndices = new Set();

// 3D Brush Surface Helper Ring
const brushHelperGeo = new THREE.RingGeometry(0.04, 0.065, 32);
const brushHelperMat = new THREE.MeshBasicMaterial({
  color: 0xf43f5e,
  side: THREE.DoubleSide,
  transparent: true,
  opacity: 0.9,
  depthTest: false
});
const brushHelper = new THREE.Mesh(brushHelperGeo, brushHelperMat);
brushHelper.visible = false;
brushHelper.renderOrder = 999;
scene.add(brushHelper);

// 3D Painted Surface Overlay Mesh
let brushSelectionGeo = new THREE.BufferGeometry();
const brushSelectionMat = new THREE.MeshStandardMaterial({
  color: 0xf43f5e,
  emissive: 0xf43f5e,
  emissiveIntensity: 0.6,
  roughness: 0.25,
  metalness: 0.1,
  side: THREE.DoubleSide,
  polygonOffset: true,
  polygonOffsetFactor: -4,
  polygonOffsetUnits: -4,
  transparent: true,
  opacity: 0.88
});
const brushSelectionMesh = new THREE.Mesh(brushSelectionGeo, brushSelectionMat);
brushSelectionMesh.visible = false;
brushSelectionMesh.renderOrder = 998;

const brushBtn = document.getElementById('tool-brush-btn');
const brushToolbar = document.getElementById('brush-toolbar');
const brushFacesBadge = document.getElementById('brush-faces-badge');
const brushModeSelect = document.getElementById('brush-mode-select');
const brushScopeSelect = document.getElementById('brush-scope-select');
const brushRadiusSlider = document.getElementById('brush-radius-slider');
const brushRadiusVal = document.getElementById('brush-radius-val');
const brushColorPicker = document.getElementById('brush-color-picker');
const brushSplitBtn = document.getElementById('brush-split-btn');
const brushClearBtn = document.getElementById('brush-clear-btn');
const brushCloseBtn = document.getElementById('brush-close-btn');

function toggleBrushMode(forceState = null) {
  isBrushActive = (forceState !== null) ? forceState : !isBrushActive;
  if (brushBtn) brushBtn.classList.toggle('active', isBrushActive);
  if (brushToolbar) brushToolbar.style.display = isBrushActive ? 'flex' : 'none';
  brushHelper.visible = false;

  if (!isBrushActive) {
    clearBrushSelection();
  }
}

if (brushBtn) brushBtn.onclick = () => toggleBrushMode();
if (brushCloseBtn) brushCloseBtn.onclick = () => toggleBrushMode(false);

if (brushModeSelect) {
  brushModeSelect.onchange = (e) => {
    brushMode = e.target.value;
  };
}

if (brushScopeSelect) {
  brushScopeSelect.onchange = (e) => {
    brushScope = e.target.value;
  };
}

if (brushRadiusSlider) {
  brushRadiusSlider.oninput = (e) => {
    brushRadius = parseFloat(e.target.value);
    if (brushRadiusVal) brushRadiusVal.innerText = `${brushRadius.toFixed(2)}m`;
    const s = brushRadius / 0.10;
    brushHelper.scale.set(s, s, s);
  };
}

if (brushColorPicker) {
  brushColorPicker.oninput = (e) => {
    brushColor = e.target.value;
    brushHelperMat.color.set(brushColor);
    brushSelectionMat.color.set(brushColor);
    brushSelectionMat.emissive.set(brushColor);
  };
}

function clearBrushSelection() {
  selectedFaceIndices.clear();
  brushLockedMesh = null;
  isBrushing = false;
  controls.enabled = true;

  if (brushSelectionMesh.parent) {
    brushSelectionMesh.parent.remove(brushSelectionMesh);
  }
  brushSelectionMesh.visible = false;
  brushSelectionGeo.dispose();
  brushSelectionGeo = new THREE.BufferGeometry();
  brushSelectionMesh.geometry = brushSelectionGeo;

  if (brushFacesBadge) brushFacesBadge.innerText = '0 faces';
}

if (brushClearBtn) brushClearBtn.onclick = () => clearBrushSelection();

// Floating Modern Toast Notification
function showToast(msg, type = 'info') {
  const existing = document.getElementById('app-floating-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = 'app-floating-toast';
  toast.className = 'glass';
  const bg = type === 'success' ? 'rgba(16, 185, 129, 0.94)' :
             type === 'warning' ? 'rgba(245, 158, 11, 0.94)' :
             'rgba(99, 102, 241, 0.94)';
  const shadow = type === 'success' ? 'rgba(16, 185, 129, 0.45)' :
                 type === 'warning' ? 'rgba(245, 158, 11, 0.45)' :
                 'rgba(99, 102, 241, 0.45)';

  toast.style.cssText = `position: fixed; top: 110px; left: 50%; transform: translateX(-50%);
    padding: 8px 18px; border-radius: 20px; font-size: 12.5px; font-weight: 700; color: #fff;
    background: ${bg}; box-shadow: 0 4px 22px ${shadow}; border: 1px solid rgba(255,255,255,0.25);
    z-index: 2500; pointer-events: none; animation: slide-down-fade 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex; align-items: center; gap: 8px; backdrop-filter: blur(12px);`;
  toast.innerHTML = msg;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity 0.35s ease, transform 0.35s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translate(-50%, -10px)';
    setTimeout(() => toast.remove(), 350);
  }, 2800);
}

// Mesh Adjacency Cache for Semantic Feature Flood-Fill
const meshAdjacencyCache = new WeakMap();

function vertKey(pos, idx) {
  const x = Math.round(pos.getX(idx) * 10000);
  const y = Math.round(pos.getY(idx) * 10000);
  const z = Math.round(pos.getZ(idx) * 10000);
  return `${x},${y},${z}`;
}

function edgeKey(iA, iB, pos) {
  const kA = vertKey(pos, iA);
  const kB = vertKey(pos, iB);
  return kA < kB ? `${kA}|${kB}` : `${kB}|${kA}`;
}

function getMeshAdjacency(geo) {
  if (meshAdjacencyCache.has(geo)) {
    return meshAdjacencyCache.get(geo);
  }
  const isIndexed = geo.index !== null;
  const index = geo.index;
  const pos = geo.attributes.position;
  const totalFaces = isIndexed ? (index.count / 3) : (pos.count / 3);

  const edgeToFaces = new Map();
  const faceNormals = [];
  const faceNeighbors = Array.from({ length: totalFaces }, () => []);

  const pA = new THREE.Vector3();
  const pB = new THREE.Vector3();
  const pC = new THREE.Vector3();
  const cb = new THREE.Vector3();
  const ab = new THREE.Vector3();

  for (let f = 0; f < totalFaces; f++) {
    const i0 = isIndexed ? index.getX(f * 3) : f * 3;
    const i1 = isIndexed ? index.getX(f * 3 + 1) : f * 3 + 1;
    const i2 = isIndexed ? index.getX(f * 3 + 2) : f * 3 + 2;

    pA.fromBufferAttribute(pos, i0);
    pB.fromBufferAttribute(pos, i1);
    pC.fromBufferAttribute(pos, i2);

    cb.subVectors(pC, pB);
    ab.subVectors(pA, pB);
    const fn = new THREE.Vector3().crossVectors(cb, ab).normalize();
    faceNormals.push(fn);

    const edges = [
      edgeKey(i0, i1, pos),
      edgeKey(i1, i2, pos),
      edgeKey(i2, i0, pos)
    ];

    for (const ek of edges) {
      if (!edgeToFaces.has(ek)) edgeToFaces.set(ek, []);
      edgeToFaces.get(ek).push(f);
    }
  }

  for (const [_, fList] of edgeToFaces.entries()) {
    if (fList.length > 1) {
      for (let i = 0; i < fList.length; i++) {
        for (let j = i + 1; j < fList.length; j++) {
          faceNeighbors[fList[i]].push(fList[j]);
          faceNeighbors[fList[j]].push(fList[i]);
        }
      }
    }
  }

  const adjData = { totalFaces, faceNormals, faceNeighbors };
  meshAdjacencyCache.set(geo, adjData);
  return adjData;
}

// 1-Click Semantic Feature Flood-Fill (< 45 deg crease angle)
function floodFillSemanticFeature(mesh, startFaceIndex) {
  if (!mesh || !mesh.geometry) return;
  const adj = getMeshAdjacency(mesh.geometry);
  if (!adj || startFaceIndex < 0 || startFaceIndex >= adj.totalFaces) return;

  const startNorm = adj.faceNormals[startFaceIndex];
  const queue = [startFaceIndex];
  const visited = new Set([startFaceIndex]);
  selectedFaceIndices.add(startFaceIndex);

  const normalThreshold = 0.707; // ~45 degree max crease threshold

  while (queue.length > 0) {
    const curr = queue.shift();
    const currNorm = adj.faceNormals[curr];
    const neighbors = adj.faceNeighbors[curr];

    for (const nb of neighbors) {
      if (!visited.has(nb)) {
        visited.add(nb);
        const nbNorm = adj.faceNormals[nb];
        if (currNorm.dot(nbNorm) >= normalThreshold && startNorm.dot(nbNorm) >= 0.45) {
          selectedFaceIndices.add(nb);
          queue.push(nb);
        }
      }
    }
  }

  updateBrushOverlay();
}

// Freehand Surface Area Paint with Fast Bounding Box & Anti-Bleed Filtering
function paintFacesAtPoint(hit) {
  if (!brushLockedMesh || !brushLockedMesh.geometry) return;
  const geo = brushLockedMesh.geometry;
  const isIndexed = geo.index !== null;
  const index = geo.index;
  const pos = geo.attributes.position;
  const totalFaces = isIndexed ? (index.count / 3) : (pos.count / 3);

  // Convert world hit point into local coordinate space of target mesh
  const localHitPoint = brushLockedMesh.worldToLocal(hit.point.clone());

  // Determine local brush radius
  const scale = brushLockedMesh.getWorldScale(new THREE.Vector3());
  const avgScale = (scale.x + scale.y + scale.z) / 3 || 1;
  const localRadius = brushRadius / Math.max(avgScale, 1e-4);
  const r2 = localRadius * localRadius;

  const minX = localHitPoint.x - localRadius;
  const maxX = localHitPoint.x + localRadius;
  const minY = localHitPoint.y - localRadius;
  const maxY = localHitPoint.y + localRadius;
  const minZ = localHitPoint.z - localRadius;
  const maxZ = localHitPoint.z + localRadius;

  const hitNorm = hit.face ? hit.face.normal : new THREE.Vector3(0, 1, 0);

  const pA = new THREE.Vector3();
  const pB = new THREE.Vector3();
  const pC = new THREE.Vector3();
  const centroid = new THREE.Vector3();
  const cb = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const fn = new THREE.Vector3();

  let addedAny = false;

  for (let f = 0; f < totalFaces; f++) {
    if (selectedFaceIndices.has(f)) continue;

    const i0 = isIndexed ? index.getX(f * 3) : f * 3;
    const i1 = isIndexed ? index.getX(f * 3 + 1) : f * 3 + 1;
    const i2 = isIndexed ? index.getX(f * 3 + 2) : f * 3 + 2;

    pA.fromBufferAttribute(pos, i0);
    pB.fromBufferAttribute(pos, i1);
    pC.fromBufferAttribute(pos, i2);

    // Fast AABB early rejection
    if ((pA.x < minX && pB.x < minX && pC.x < minX) ||
        (pA.x > maxX && pB.x > maxX && pC.x > maxX) ||
        (pA.y < minY && pB.y < minY && pC.y < minY) ||
        (pA.y > maxY && pB.y > maxY && pC.y > maxY) ||
        (pA.z < minZ && pB.z < minZ && pC.z < minZ) ||
        (pA.z > maxZ && pB.z > maxZ && pC.z > maxZ)) {
      continue;
    }

    centroid.set(
      (pA.x + pB.x + pC.x) / 3,
      (pA.y + pB.y + pC.y) / 3,
      (pA.z + pB.z + pC.z) / 3
    );

    const dCentroid2 = centroid.distanceToSquared(localHitPoint);
    const dMin2 = Math.min(
      dCentroid2,
      pA.distanceToSquared(localHitPoint),
      pB.distanceToSquared(localHitPoint),
      pC.distanceToSquared(localHitPoint)
    );

    if (dMin2 <= r2) {
      // Normal alignment check (anti-backface check to prevent painting through thin walls)
      cb.subVectors(pC, pB);
      ab.subVectors(pA, pB);
      fn.crossVectors(cb, ab).normalize();

      if (fn.dot(hitNorm) > 0.05) {
        selectedFaceIndices.add(f);
        addedAny = true;
      }
    }
  }

  if (addedAny) {
    updateBrushOverlay();
  }
}

// Synchronize 3D Visual Mesh Overlay and Badge Counter
function updateBrushOverlay() {
  if (brushFacesBadge) {
    brushFacesBadge.innerText = `${selectedFaceIndices.size} faces`;
  }

  if (!brushLockedMesh || selectedFaceIndices.size === 0) {
    brushSelectionMesh.visible = false;
    return;
  }

  const geo = brushLockedMesh.geometry;
  const isIndexed = geo.index !== null;
  const index = geo.index;
  const pos = geo.attributes.position;
  const norm = geo.attributes.normal;

  const S = selectedFaceIndices.size;
  const selPos = new Float32Array(S * 9);
  const selNorm = norm ? new Float32Array(S * 9) : null;

  let sIdx = 0;
  for (const f of selectedFaceIndices) {
    const i0 = isIndexed ? index.getX(f * 3) : f * 3;
    const i1 = isIndexed ? index.getX(f * 3 + 1) : f * 3 + 1;
    const i2 = isIndexed ? index.getX(f * 3 + 2) : f * 3 + 2;
    const vIndices = [i0, i1, i2];

    for (let v = 0; v < 3; v++) {
      const vi = vIndices[v];
      const pBase = sIdx * 9 + v * 3;
      selPos[pBase] = pos.getX(vi);
      selPos[pBase + 1] = pos.getY(vi);
      selPos[pBase + 2] = pos.getZ(vi);

      if (norm && selNorm) {
        selNorm[pBase] = norm.getX(vi);
        selNorm[pBase + 1] = norm.getY(vi);
        selNorm[pBase + 2] = norm.getZ(vi);
      }
    }
    sIdx++;
  }

  brushSelectionGeo.dispose();
  brushSelectionGeo = new THREE.BufferGeometry();
  brushSelectionGeo.setAttribute('position', new THREE.BufferAttribute(selPos, 3));
  if (selNorm) {
    brushSelectionGeo.setAttribute('normal', new THREE.BufferAttribute(selNorm, 3));
  } else {
    brushSelectionGeo.computeVertexNormals();
  }
  brushSelectionMesh.geometry = brushSelectionGeo;

  if (brushSelectionMesh.parent !== brushLockedMesh) {
    brushLockedMesh.add(brushSelectionMesh);
  }
  brushSelectionMesh.position.set(0, 0, 0);
  brushSelectionMesh.rotation.set(0, 0, 0);
  brushSelectionMesh.scale.set(1, 1, 1);
  brushSelectionMesh.visible = true;
}

// Partition Geometry into Selected Sub-Mesh and Remaining Mesh
function partitionGeometry(sourceGeo, selectedSet) {
  const isIndexed = sourceGeo.index !== null;
  const index = sourceGeo.index;
  const pos = sourceGeo.attributes.position;
  const norm = sourceGeo.attributes.normal;
  const uv = sourceGeo.attributes.uv;
  const color = sourceGeo.attributes.color;

  const totalFaces = isIndexed ? (index.count / 3) : (pos.count / 3);
  const S = selectedSet.size;
  const R = totalFaces - S;

  const selPos = new Float32Array(S * 9);
  const remPos = new Float32Array(R * 9);

  const selNorm = norm ? new Float32Array(S * 9) : null;
  const remNorm = norm ? new Float32Array(R * 9) : null;

  const selUv = uv ? new Float32Array(S * 6) : null;
  const remUv = uv ? new Float32Array(R * 6) : null;

  const selCol = color ? new Float32Array(S * 3 * color.itemSize) : null;
  const remCol = color ? new Float32Array(R * 3 * color.itemSize) : null;

  let sIdx = 0;
  let rIdx = 0;

  for (let f = 0; f < totalFaces; f++) {
    const isSel = selectedSet.has(f);
    const targetPos = isSel ? selPos : remPos;
    const targetNorm = isSel ? selNorm : remNorm;
    const targetUv = isSel ? selUv : remUv;
    const targetCol = isSel ? selCol : remCol;
    const offset = isSel ? sIdx : rIdx;

    const i0 = isIndexed ? index.getX(f * 3) : f * 3;
    const i1 = isIndexed ? index.getX(f * 3 + 1) : f * 3 + 1;
    const i2 = isIndexed ? index.getX(f * 3 + 2) : f * 3 + 2;
    const vertIndices = [i0, i1, i2];

    for (let v = 0; v < 3; v++) {
      const vi = vertIndices[v];
      const pBase = offset * 9 + v * 3;
      targetPos[pBase] = pos.getX(vi);
      targetPos[pBase + 1] = pos.getY(vi);
      targetPos[pBase + 2] = pos.getZ(vi);

      if (norm && targetNorm) {
        targetNorm[pBase] = norm.getX(vi);
        targetNorm[pBase + 1] = norm.getY(vi);
        targetNorm[pBase + 2] = norm.getZ(vi);
      }

      if (uv && targetUv) {
        const uvBase = offset * 6 + v * 2;
        targetUv[uvBase] = uv.getX(vi);
        targetUv[uvBase + 1] = uv.getY(vi);
      }

      if (color && targetCol) {
        const cBase = offset * 3 * color.itemSize + v * color.itemSize;
        for (let c = 0; c < color.itemSize; c++) {
          targetCol[cBase + c] = color.getComponent(vi, c);
        }
      }
    }

    if (isSel) sIdx++;
    else rIdx++;
  }

  const selectedGeo = new THREE.BufferGeometry();
  selectedGeo.setAttribute('position', new THREE.BufferAttribute(selPos, 3));
  if (selNorm) selectedGeo.setAttribute('normal', new THREE.BufferAttribute(selNorm, 3));
  if (selUv) selectedGeo.setAttribute('uv', new THREE.BufferAttribute(selUv, 2));
  if (selCol) selectedGeo.setAttribute('color', new THREE.BufferAttribute(selCol, color.itemSize));

  const remainingGeo = new THREE.BufferGeometry();
  remainingGeo.setAttribute('position', new THREE.BufferAttribute(remPos, 3));
  if (remNorm) remainingGeo.setAttribute('normal', new THREE.BufferAttribute(remNorm, 3));
  if (remUv) remainingGeo.setAttribute('uv', new THREE.BufferAttribute(remUv, 2));
  if (remCol) remainingGeo.setAttribute('color', new THREE.BufferAttribute(remCol, color.itemSize));

  if (!selNorm) selectedGeo.computeVertexNormals();
  if (!remNorm) remainingGeo.computeVertexNormals();

  selectedGeo.computeBoundingBox();
  selectedGeo.computeBoundingSphere();
  remainingGeo.computeBoundingBox();
  remainingGeo.computeBoundingSphere();

  return { selectedGeo, remainingGeo };
}

// Split and Carve Brushed Surface Patch into an Independent Segmented Part
function splitBrushedAreaAsSeparatePart() {
  if (!brushLockedMesh || selectedFaceIndices.size === 0) {
    showToast('⚠️ Please brush an area or flood-select a feature on the 3D model first!', 'warning');
    return;
  }

  const parentMesh = brushLockedMesh;
  const srcGeo = parentMesh.geometry;
  const isIndexed = srcGeo.index !== null;
  const totalFaces = isIndexed ? (srcGeo.index.count / 3) : (srcGeo.attributes.position.count / 3);

  if (selectedFaceIndices.size >= totalFaces) {
    showToast('⚠️ All faces of this part are selected. The entire part is already a separate part.', 'warning');
    return;
  }

  const selectedCount = selectedFaceIndices.size;
  const { selectedGeo, remainingGeo } = partitionGeometry(srcGeo, selectedFaceIndices);

  // Update parent mesh with remaining geometry
  parentMesh.geometry.dispose();
  parentMesh.geometry = remainingGeo;
  parentMesh.geometry.computeBoundingBox();
  parentMesh.geometry.computeBoundingSphere();

  // Create new mesh with chosen brush color
  const newHex = brushColor;
  const newMat = new THREE.MeshStandardMaterial({
    color: newHex,
    roughness: 0.35,
    metalness: 0.05,
    emissive: newHex,
    emissiveIntensity: 0.15,
    side: THREE.DoubleSide
  });

  const newMesh = new THREE.Mesh(selectedGeo, newMat);
  const parentInfo = partsData.get(parentMesh);
  const nextIdx = partsData.size;
  const newPartName = `${parentInfo ? parentInfo.name : 'part'}_carved_${nextIdx + 1}`;
  newMesh.name = newPartName;

  newMesh.position.copy(parentMesh.position);
  newMesh.quaternion.copy(parentMesh.quaternion);
  newMesh.scale.copy(parentMesh.scale);

  if (parentMesh.parent) {
    parentMesh.parent.add(newMesh);
  } else if (currentLoadedMesh) {
    currentLoadedMesh.add(newMesh);
  } else {
    scene.add(newMesh);
  }

  // Calculate distinct explosion vector for new carved part
  const worldPos = new THREE.Vector3();
  newMesh.getWorldPosition(worldPos);

  const lVec = worldPos.clone().sub(globalCenter);
  let dist = lVec.length();
  if (dist < 1e-4) {
    lVec.set((Math.random() - 0.5) * 0.02, (Math.random() - 0.5) * 0.02, (Math.random() - 0.5) * 0.02);
  }
  lVec.normalize();

  const geomCenter = new THREE.Vector3();
  if (newMesh.geometry.boundingBox) {
    newMesh.geometry.boundingBox.getCenter(geomCenter);
  }

  const rotAxis = new THREE.Vector3(1, (Math.random() - 0.5) * 0.6, (Math.random() - 0.5) * 0.6).normalize();
  const rotSpeedFactor = (0.5 + Math.random()) * (Math.random() > 0.5 ? 1 : -1);

  const newPartInfo = {
    mesh: newMesh,
    index: nextIdx,
    name: newPartName,
    originalLocalPosition: newMesh.position.clone(),
    originalRotation: newMesh.quaternion.clone(),
    originalScale: newMesh.scale.clone(),
    explodeDirection: lVec.clone(),
    distanceWeight: maxDistance > 0 ? (dist / maxDistance) : 1.0,
    geometryCenterOffset: geomCenter.clone(),
    accumulatedRotation: 0,
    rotationAxis: rotAxis,
    rotationSpeedFactor: rotSpeedFactor,
    snapshotRotation: 0,
    customOffset: new THREE.Vector3(),
    customRotation: new THREE.Euler(),
    customScale: 1.0,
    visible: true
  };

  partsData.set(newMesh, newPartInfo);
  paletteMaterials.set(newMesh, newMat);
  originalMaterials.set(newMesh, cloneMaterialSafe(newMat));

  // Recalculate parent part centroid
  if (parentMesh.geometry.boundingBox && parentInfo) {
    parentMesh.geometry.boundingBox.getCenter(parentInfo.geometryCenterOffset);
  }

  // Clear active brush selection and overlay
  clearBrushSelection();

  // Dynamically update parts hierarchy, quick dropdown & badge counter
  rebuildPartsHierarchyUI();

  // Select newly created part in Blender inspector
  selectPart(newMesh);

  // Update Geometry Statistics HUD
  if (currentLoadedMesh) {
    updateMeshStatsDisplay(calculateGeometryStats(currentLoadedMesh));
  }

  showToast(`✂️ Carved & extracted <strong>${selectedCount}</strong> faces as separate part <em>"${newPartName}"</em>!`, 'success');
}

if (brushSplitBtn) brushSplitBtn.onclick = () => splitBrushedAreaAsSeparatePart();

// Hotkey: press 'B' to toggle 3D brush mode
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'b' || e.key === 'B') {
    toggleBrushMode();
  }
});

// 3D Canvas Raycasting for Brush
const brushRaycaster = new THREE.Raycaster();
const brushMouse = new THREE.Vector2();

function getModelIntersects(e, restrictMesh = null) {
  const rect = renderer.domElement.getBoundingClientRect();
  brushMouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  brushMouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  brushRaycaster.setFromCamera(brushMouse, (isOrtho && orthoCamera) ? orthoCamera : camera);

  if (!currentLoadedMesh) return [];

  if (restrictMesh) {
    return brushRaycaster.intersectObject(restrictMesh, true);
  }

  const meshes = [];
  currentLoadedMesh.traverse((child) => {
    if (child.isMesh && child.visible && child !== brushHelper && child !== brushSelectionMesh) {
      meshes.push(child);
    }
  });
  return brushRaycaster.intersectObjects(meshes, true);
}

renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!isBrushActive || e.button !== 0) return;
  const hits = getModelIntersects(e, (brushScope === 'part' && brushLockedMesh) ? brushLockedMesh : null);
  if (hits.length > 0) {
    const hit = hits[0];
    const hitMesh = hit.object;

    if (!brushLockedMesh) {
      brushLockedMesh = hitMesh;
    } else if (brushScope === 'part' && hitMesh !== brushLockedMesh) {
      // ANTI-BLEED: ignore touches on other parts
      return;
    }

    if (brushMode === 'flood') {
      floodFillSemanticFeature(brushLockedMesh, hit.faceIndex);
    } else {
      isBrushing = true;
      controls.enabled = false; // Freeze orbit camera so brushing doesn't spin viewport
      paintFacesAtPoint(hit);
    }
  }
});

renderer.domElement.addEventListener('pointermove', (e) => {
  if (!isBrushActive) return;

  const surfaceHits = getModelIntersects(e, (brushScope === 'part' && brushLockedMesh) ? brushLockedMesh : null);
  if (surfaceHits.length > 0) {
    const hit = surfaceHits[0];
    brushHelper.visible = true;
    brushHelper.position.copy(hit.point).addScaledVector(hit.face.normal, 0.003);
    brushHelper.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), hit.face.normal);

    if (isBrushing && brushMode === 'paint') {
      if (brushScope === 'part') {
        if (hit.object === brushLockedMesh) {
          paintFacesAtPoint(hit);
        }
      } else {
        if (hit.object !== brushLockedMesh) {
          brushLockedMesh = hit.object;
        }
        paintFacesAtPoint(hit);
      }
    }
  } else {
    brushHelper.visible = false;
  }
});

window.addEventListener('pointerup', () => {
  if (isBrushing) {
    isBrushing = false;
    controls.enabled = true; // Restore orbit camera
  }
});

// --- PARTS DROPDOWN TOGGLE & GLOBAL CLICK-OUTSIDE ---
const partsDropdownToggle = document.getElementById('parts-dropdown-toggle');
const partsDropdownMenu = document.getElementById('parts-dropdown-menu');
const partsDropdownClose = document.getElementById('parts-dropdown-close');

if (partsDropdownToggle && partsDropdownMenu) {
  partsDropdownToggle.onclick = (e) => {
    e.stopPropagation();
    const isVisible = partsDropdownMenu.style.display !== 'none';
    partsDropdownMenu.style.display = isVisible ? 'none' : 'flex';
    partsDropdownToggle.classList.toggle('active', !isVisible);
  };
}

if (partsDropdownClose && partsDropdownMenu) {
  partsDropdownClose.onclick = () => {
    partsDropdownMenu.style.display = 'none';
    if (partsDropdownToggle) partsDropdownToggle.classList.remove('active');
  };
}

// Click outside to close parts dropdown
document.addEventListener('click', (e) => {
  if (partsDropdownMenu && partsDropdownMenu.style.display !== 'none') {
    if (!partsDropdownMenu.contains(e.target) && (!partsDropdownToggle || !partsDropdownToggle.contains(e.target))) {
      partsDropdownMenu.style.display = 'none';
      if (partsDropdownToggle) partsDropdownToggle.classList.remove('active');
    }
  }
});

// --- TRIPO3D EXPLOSION ENGINE ---
function applyExplosion(progress, deltaSec) {
  if (partsData.size === 0) return;

  const isTumblingEnabled = document.getElementById('tumbling-toggle').checked;
  const easedProgress = currentState === States.Collapsing
    ? tripoCollapseEasing(progress)
    : tripoOvershootEasing(progress);

  partsData.forEach((info, mesh) => {
    if (isTumblingEnabled && currentState === States.Rotating) {
      info.accumulatedRotation += deltaSec * 0.8 * info.rotationSpeedFactor;
      const rotQuat = new THREE.Quaternion().setFromAxisAngle(info.rotationAxis, info.accumulatedRotation);
      mesh.quaternion.copy(info.originalRotation).multiply(rotQuat);
    }

    const rDist = maxDistance * (0.35 + 0.65 * Math.min(1, info.distanceWeight));
    const disp = info.explodeDirection.clone().multiplyScalar(easedProgress * rDist);
    mesh.position.copy(info.originalLocalPosition).add(disp).add(info.customOffset);
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
  isExploded = (val > 0.05);
  explodeBtn.classList.toggle('active', isExploded);
  explodeText.innerText = isExploded ? 'Collapse' : 'Explode';
  applyExplosion(val, 0.016);
};

// --- RECORDING STUDIO (60 FPS VIDEO & SNAPSHOTS) ---
function startRecording(type = 'explosion') {
  if (isRecording) return;
  const stream = renderer.domElement.captureStream(60);
  recordedChunks = [];

  let options = { mimeType: 'video/webm;codecs=vp9' };
  if (!MediaRecorder.isTypeSupported(options.mimeType)) {
    options = { mimeType: 'video/webm' };
  }

  try {
    mediaRecorder = new MediaRecorder(stream, options);
  } catch (err) {
    alert('MediaRecorder not supported: ' + err);
    return;
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = type === 'turntable' ? 'tripo_turntable_360.webm' : 'tripo_explosion_animation.webm';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    stopRecordingUI();
  };

  mediaRecorder.start();
  isRecording = true;
  recordStartTime = performance.now();
  startRecordingUI();

  if (type === 'explosion') {
    // Reset to 0 then animate
    explodeSlider.value = 0;
    applyExplosion(0, 0.016);
    animStartTime = performance.now();
    startProgress = 0;
    currentState = States.Expanding;
    isExploded = true;
    explodeBtn.classList.add('active');
    explodeText.innerText = 'Collapse';

    // Record expansion (1s) + 1.5s inspection + collapse (1s) = 3.5s total
    setTimeout(() => {
      animStartTime = performance.now();
      startProgress = 1.0;
      currentState = States.Collapsing;
      isExploded = false;
      explodeBtn.classList.remove('active');
      explodeText.innerText = 'Explode';
    }, 2500);

    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
    }, 3800);
  } else if (type === 'turntable') {
    controls.autoRotate = true;
    controls.autoRotateSpeed = 6.0;
    setTimeout(() => {
      controls.autoRotate = false;
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
    }, 5500);
  }
}

function startRecordingUI() {
  const badge = document.getElementById('recording-badge');
  badge.style.display = 'flex';
  recordingTimer = setInterval(() => {
    const elapsed = (performance.now() - recordStartTime) / 1000;
    const mins = Math.floor(elapsed / 60);
    const secs = Math.floor(elapsed % 60);
    document.getElementById('recording-status-text').innerText =
      `REC ${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }, 200);
}

function stopRecordingUI() {
  isRecording = false;
  if (recordingTimer) clearInterval(recordingTimer);
  document.getElementById('recording-badge').style.display = 'none';
}

document.getElementById('record-explode-btn').onclick = () => startRecording('explosion');
document.getElementById('record-turntable-btn').onclick = () => startRecording('turntable');

// High-Res Snapshot
document.getElementById('snapshot-btn').onclick = () => {
  renderer.render(scene, isOrtho && orthoCamera ? orthoCamera : camera);
  const dataUrl = renderer.domElement.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `tripo_snapshot_${Date.now()}.png`;
  a.click();
};

// --- MODEL LOADING (GLB, OBJ, PLY) WITH DRACO & MESHOPT DECOMPRESSION ---
const dracoLoader = new THREE.DRACOLoader();
dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.4.3/');
dracoLoader.setDecoderConfig({ type: 'js' });

const gltfLoader = new THREE.GLTFLoader();
gltfLoader.setDRACOLoader(dracoLoader);
if (typeof MeshoptDecoder !== 'undefined') {
  gltfLoader.setMeshoptDecoder(MeshoptDecoder);
}

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

    // Reset explode controls
    isExploded = false;
    currentState = States.Idle;
    explodeBtn.classList.remove('active');
    explodeText.innerText = 'Explode';
    explodeSlider.value = 0;
  };

  if (ext === '.obj') {
    objLoader.load(url, handleLoadedModel, undefined, (err) => {
      console.error('Error loading OBJ:', err);
      alert('Error loading OBJ: ' + (err.message || err));
    });
  } else if (ext === '.ply') {
    plyLoader.load(url, (geom) => {
      geom.computeVertexNormals();
      const mat = new THREE.MeshStandardMaterial({ color: 0x818cf8, roughness: 0.4, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geom, mat);
      const group = new THREE.Group();
      group.add(mesh);
      handleLoadedModel(group);
    }, undefined, (err) => {
      console.error('Error loading PLY:', err);
      alert('Error loading PLY: ' + (err.message || err));
    });
  } else {
    gltfLoader.load(url, (gltf) => {
      const sceneObj = gltf.scene || (gltf.scenes && gltf.scenes[0]) || gltf;
      handleLoadedModel(sceneObj);
    }, (xhr) => {
      if (xhr.lengthComputable && xhr.total > 0) {
        const percent = Math.round((xhr.loaded / xhr.total) * 100);
        const stageText = document.getElementById('progress-stage-text');
        const percentText = document.getElementById('progress-percent-text');
        const barFill = document.getElementById('progress-bar-fill');
        if (stageText && percent < 100) {
          stageText.innerText = `Loading 3D asset (${percent}%)...`;
          percentText.innerText = `${percent}%`;
          barFill.style.width = `${percent}%`;
        }
      }
    }, (err) => {
      console.error('Error loading GLB:', err);
      alert('Error loading GLB: ' + (err.message || err));
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
      stageText.innerText = 'Geometry parsed. Initiating Auto-Segmentation...';
      percentText.innerText = '100%';
      barFill.style.width = '100%';

      loadModelFromUrl(data.path, file.name, data.stats);

      // Instantly run the segmentation pipeline automatically!
      setTimeout(() => {
        runAutoSegmentationPipeline(data.path, file.name);
      }, 400);
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
  if (!badge) return;
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
  if (!badge) return;
  badge.style.color = '#fff';
  badge.style.boxShadow = 'none';
  if (finalFormatted) {
    badge.innerText = `⚡ ${finalFormatted}`;
  }
}

async function runAutoSegmentationPipeline(meshPath = null, modelTitle = null) {
  const targetPath = meshPath || currentModelPath;
  if (!targetPath) {
    alert('Please upload or select a 3D model first!');
    return;
  }

  const banner = document.getElementById('progress-banner');
  const stageText = document.getElementById('progress-stage-text');
  const percentText = document.getElementById('progress-percent-text');
  const barFill = document.getElementById('progress-bar-fill');
  if (banner) banner.style.display = 'flex';

  // Reset timing fields
  const tStep1 = document.getElementById('time-step1');
  const tStep2 = document.getElementById('time-step2');
  const tStep3 = document.getElementById('time-step3');
  const tTotal = document.getElementById('time-total');
  if (tStep1) tStep1.innerText = '-- s';
  if (tStep2) tStep2.innerText = '-- s';
  if (tStep3) tStep3.innerText = '-- s';
  if (tTotal) tTotal.innerText = '-- s';
  startLiveStopwatch();

  try {
    const res = await fetch('/api/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mesh_path: targetPath,
        mode: 'tripo_exact',
        preserve_materials: true,
        mc_algo: 'dmc',
        steps: 50,
        resolution: 512,
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

        if (stageText) stageText.innerText = sData.stage || 'Processing...';
        if (percentText) percentText.innerText = `${sData.progress || 0}%`;
        if (barFill) barFill.style.width = `${sData.progress || 0}%`;

        if (sData.timings) {
          if (tStep1 && sData.timings.step1 > 0) tStep1.innerText = `${sData.timings.step1}s`;
          if (tStep2 && sData.timings.step2 > 0) tStep2.innerText = `${sData.timings.step2}s`;
        }

        if (sData.status === 'completed') {
          clearInterval(pollInterval);
          setTimeout(() => { if (banner) banner.style.display = 'none'; }, 600);

          if (sData.result && sData.result.timing) {
            const t = sData.result.timing;
            if (tStep1) tStep1.innerText = `${t.step1}s`;
            if (tStep2) tStep2.innerText = `${t.step2}s`;
            if (tStep3) tStep3.innerText = `${t.step3}s`;
            if (tTotal) tTotal.innerText = `${t.formatted}`;
            stopLiveStopwatch(t.formatted);
          } else {
            stopLiveStopwatch();
          }

          // Load generated multi-node assembly
          const title = modelTitle || document.getElementById('model-title-text').innerText;
          loadModelFromUrl(
            sData.result.assembly_glb,
            `${title} (Segmented Assembly)`,
            sData.result.mesh_details
          );

          // Update Export Links
          const dlAssemblyBtn = document.getElementById('download-assembly-btn');
          const dlExplodedBtn = document.getElementById('download-exploded-btn');
          if (dlAssemblyBtn) {
            dlAssemblyBtn.onclick = () => { window.location.href = sData.result.assembly_glb; };
          }
          if (dlExplodedBtn) {
            dlExplodedBtn.onclick = () => { window.location.href = sData.result.exploded_glb; };
          }
        } else if (sData.status === 'failed') {
          clearInterval(pollInterval);
          if (banner) banner.style.display = 'none';
          stopLiveStopwatch();
          alert(`Pipeline execution failed: ${sData.error}`);
        }
      } catch (pollErr) {
        console.error('Polling error:', pollErr);
      }
    }, 800);
  } catch (err) {
    if (banner) banner.style.display = 'none';
    stopLiveStopwatch();
    alert('Pipeline error: ' + err);
  }
}

const runPipelineBtn = document.getElementById('run-pipeline-btn');
if (runPipelineBtn) {
  runPipelineBtn.onclick = () => runAutoSegmentationPipeline();
}

// --- KEYBOARD SHORTCUTS (Blender Standards) ---
window.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

  if (e.key === '1') setCameraView('front');
  else if (e.key === '3') setCameraView('side');
  else if (e.key === '7') setCameraView('top');
  else if (e.key === '0') setCameraView('iso');
  else if (e.key === '.') {
    if (currentLoadedMesh) autoFrameModel(currentLoadedMesh);
  } else if (e.key === '5') {
    document.getElementById('cam-proj').click();
  } else if (e.key === 'b' || e.key === 'B') {
    const bBtn = document.getElementById('tool-brush-btn');
    if (bBtn) bBtn.click();
  } else if (e.key === 'h' && !e.altKey) {
    document.getElementById('editor-hide-btn').click();
  } else if (e.key === 'H' || (e.key === 'h' && e.altKey)) {
    document.getElementById('show-all-parts-btn').click();
  } else if (e.key === '/') {
    document.getElementById('editor-isolate-btn').click();
  }
});

// --- RENDER LOOP ---
window.addEventListener('resize', () => {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  if (orthoCamera) {
    const aspect = w / h;
    const dist = camera.position.distanceTo(controls.target);
    const frustumSize = dist * 0.8;
    orthoCamera.left = (frustumSize * aspect) / -2;
    orthoCamera.right = (frustumSize * aspect) / 2;
    orthoCamera.top = frustumSize / 2;
    orthoCamera.bottom = frustumSize / -2;
    orthoCamera.updateProjectionMatrix();
  }

  renderer.setSize(w, h);
});

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  controls.update();
  updateAnimation(delta);

  const activeCam = (isOrtho && orthoCamera) ? orthoCamera : camera;
  renderer.render(scene, activeCam);
}
animate();

// Initial load: parse URL model query param or auto-load default
const urlParams = new URLSearchParams(window.location.search);
const modelFromUrl = urlParams.get('model');
if (modelFromUrl) {
  loadModelFromUrl(modelFromUrl, 'Loaded 3D Object');
} else {
  loadModelFromUrl(
    '/outputs/furn16_3k_tripo_exact/train_cfg_-1.000000_dit_boxgpt_furn16_3k_assembly.glb',
    'furn16_3k Furniture (Tripo3D Exact)'
  );
}

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

function resetExplosionToRest() {
  currentState = States.Idle;
  isExploded = false;
  const expBtn = document.getElementById('explode-toggle');
  const expTxt = document.getElementById('explode-text');
  const expSld = document.getElementById('explode-slider');
  if (expBtn) expBtn.classList.remove('active');
  if (expTxt) expTxt.innerText = 'Explode';
  if (expSld) expSld.value = 0;

  partsData.forEach((info, mesh) => {
    if (mesh && info) {
      mesh.position.copy(info.originalLocalPosition).add(info.customOffset || new THREE.Vector3());
      mesh.quaternion.copy(info.originalRotation);
      info.accumulatedRotation = 0;
      mesh.updateMatrixWorld(true);
    }
  });
}

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

// Checkbox tracking for multi-part selection & joining
const checkedParts = new Set();
let paletteColorIndex = 0;

function getNextDistinctPaletteColor() {
  const col = TRIPO_PALETTE[paletteColorIndex % TRIPO_PALETTE.length];
  paletteColorIndex++;
  return col;
}

function updateBulkActionBar() {
  const bulkBar = document.getElementById('parts-bulk-bar');
  const bulkText = document.getElementById('parts-bulk-text');
  if (bulkBar && bulkText) {
    if (checkedParts.size > 0) {
      bulkBar.style.display = 'flex';
      bulkText.innerText = `${checkedParts.size} parts checked`;
    } else {
      bulkBar.style.display = 'none';
    }
  }
}

// Bulk Join / Split / Clear wiring
const bulkJoinBtn = document.getElementById('parts-bulk-join-btn');
if (bulkJoinBtn) {
  bulkJoinBtn.onclick = () => {
    if (checkedParts.size < 2) {
      showToast('⚠️ Please check at least 2 parts to combine them!', 'warning');
      return;
    }
    joinSelectedParts(Array.from(checkedParts));
  };
}

const bulkSplitBtn = document.getElementById('parts-bulk-split-btn');
if (bulkSplitBtn) {
  bulkSplitBtn.onclick = () => {
    cutOrDivideBrushedParts();
  };
}

const bulkClearBtn = document.getElementById('parts-bulk-clear-btn');
if (bulkClearBtn) {
  bulkClearBtn.onclick = () => {
    checkedParts.clear();
    updateBulkActionBar();
    rebuildPartsHierarchyUI();
  };
}

// --- CLIENT-SIDE GLTF / GLB EXPORT ENGINE (Tripo Step 4: Export) ---
function exportMeshToGLB(mesh, filename = 'part') {
  if (!window.THREE || !THREE.GLTFExporter) {
    showToast('⚠️ GLTFExporter is loading, please try again in a moment.', 'warning');
    return;
  }
  const exporter = new THREE.GLTFExporter();
  const exportScene = new THREE.Scene();
  const clone = mesh.clone();
  clone.position.set(0, 0, 0);
  clone.rotation.set(0, 0, 0);
  clone.scale.set(1, 1, 1);
  exportScene.add(clone);

  exporter.parse(
    exportScene,
    function (result) {
      if (result instanceof ArrayBuffer) {
        const blob = new Blob([result], { type: 'model/gltf-binary' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        const safeName = (filename || 'part').replace(/[^a-zA-Z0-9_-]/g, '_');
        link.download = `${safeName}.glb`;
        link.click();
        URL.revokeObjectURL(link.href);
        showToast(`📥 Successfully exported <strong>"${safeName}.glb"</strong>!`, 'success');
      }
    },
    { binary: true }
  );
}

function exportAllSegmentedParts() {
  if (partsData.size === 0) {
    showToast('⚠️ No segmented parts to export.', 'warning');
    return;
  }
  let delay = 0;
  partsData.forEach((info, mesh) => {
    setTimeout(() => {
      exportMeshToGLB(mesh, info.name);
    }, delay);
    delay += 350;
  });
  showToast(`📥 Exporting all <strong>${partsData.size}</strong> parts as individual GLB models...`, 'info');
}

const partsExportAllBtn = document.getElementById('parts-export-all-btn');
if (partsExportAllBtn) partsExportAllBtn.onclick = () => exportAllSegmentedParts();

const partsBulkExportBtn = document.getElementById('parts-bulk-export-btn');
if (partsBulkExportBtn) {
  partsBulkExportBtn.onclick = () => {
    if (checkedParts.size === 0) {
      showToast('⚠️ Please check at least one part to export.', 'warning');
      return;
    }
    let delay = 0;
    checkedParts.forEach((mesh) => {
      const info = partsData.get(mesh);
      const name = info ? info.name : 'part';
      setTimeout(() => {
        exportMeshToGLB(mesh, name);
      }, delay);
      delay += 350;
    });
    showToast(`📥 Exporting <strong>${checkedParts.size}</strong> checked parts...`, 'info');
  };
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

  updateBulkActionBar();

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

    const isChecked = checkedParts.has(mesh);

    const card = document.createElement('div');
    card.className = 'part-item';
    card.id = `part-card-${idx}`;
    card.innerHTML = `
      <div class="part-main-row">
        <div class="part-badge">
          <input type="checkbox" class="part-checkbox" id="part-cb-${idx}" title="Check to multi-select for Joining" ${isChecked ? 'checked' : ''}>
          <input type="color" value="${hexColor}" class="part-color-dot" id="color-dot-${idx}" title="Click to Change Part Color">
          <input type="text" class="part-name-input" id="part-name-input-${idx}" value="${partInfo.name}" title="Click to Rename Part">
        </div>
        <div style="display: flex; gap: 4px;">
          <button class="icon-tool-btn" id="export-part-btn-${idx}" title="📥 Export This Part as GLB">📥</button>
          <button class="icon-tool-btn" id="divide-btn-${idx}" title="⚡ Divide Disjoint Components into Multiple Parts with Different Colors">⚡</button>
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

    const cb = card.querySelector(`#part-cb-${idx}`);
    if (cb) {
      cb.onchange = (e) => {
        e.stopPropagation();
        if (cb.checked) checkedParts.add(mesh);
        else checkedParts.delete(mesh);
        updateBulkActionBar();
      };
    }

    const expBtn = card.querySelector(`#export-part-btn-${idx}`);
    if (expBtn) {
      expBtn.onclick = (e) => {
        e.stopPropagation();
        exportMeshToGLB(mesh, partInfo.name);
      };
    }

    const nameInput = card.querySelector(`#part-name-input-${idx}`);
    if (nameInput) {
      nameInput.onchange = (e) => {
        partInfo.name = e.target.value.trim() || `part_${idx + 1}`;
        mesh.name = partInfo.name;
        updateBrushReassignDropdown();
        const opt = document.querySelector(`#quick-part-select option[value="${idx}"]`);
        if (opt) opt.textContent = `Part #${idx + 1}: ${partInfo.name}`;
      };
      nameInput.onclick = (e) => e.stopPropagation();
    }

    const divBtn = card.querySelector(`#divide-btn-${idx}`);
    if (divBtn) {
      divBtn.onclick = (e) => {
        e.stopPropagation();
        dividePartIntoConnectedComponents(mesh);
      };
    }

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

  updateBrushReassignDropdown();

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

// --- MULTI-PART 3D BRUSH, GLOBAL SEMANTIC MATCHING, JOIN & CUT/DIVIDE ---
let isBrushActive = false;
let isBrushing = false;
let brushLockedMesh = null; // Used ONLY when brushScope === 'part'
let brushMode = 'paint';    // 'paint' (Freehand Area) | 'flood' (Flood Feature) | 'semantic' (Global Semantic)
let brushScope = 'multi';   // 'multi' (Multi-Region / All Parts) [DEFAULT] | 'part' (Lock to Single Part)
let brushRadius = 0.10;
const selectedMeshFaces = new Map();  // THREE.Mesh -> Set<number> (face indices)
const brushOverlayMeshes = new Map(); // THREE.Mesh -> THREE.Mesh (child visual overlay)

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
brushHelper.raycast = function() {}; // Prevent raycaster interference
scene.add(brushHelper);

// 3D Painted Surface Overlay Material
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

// --- TRIPO3D PRECISION TIERS (Simple, Balanced, Detailed) ---
let currentPrecisionTier = 'balanced';

function setPrecisionTier(tier) {
  currentPrecisionTier = tier;
  ['tier-simple', 'tier-balanced', 'tier-detailed'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active');
  });
  const activeEl = document.getElementById(`tier-${tier}`);
  if (activeEl) activeEl.classList.add('active');

  if (partsData.size === 0) return;

  saveStateToUndoStack(`Change Tier to ${tier}`);

  if (tier === 'simple') {
    applySimpleCoarseTier();
  } else if (tier === 'detailed') {
    applyDetailedFineTier();
  } else {
    showToast('⚖️ Balanced tier: standard production segmentation active.', 'info');
  }
}

function applyDetailedFineTier() {
  let totalDivided = 0;
  const meshes = Array.from(partsData.keys());
  meshes.forEach(m => {
    if (dividePartIntoConnectedComponents(m, false)) {
      totalDivided++;
    }
  });
  stageOperation(`Detailed Tier: split into ${partsData.size} parts`);
  showToast(`🔍 Detailed Tier active: model split into <strong>${partsData.size}</strong> fine components!`, 'success');
}

function applySimpleCoarseTier() {
  if (partsData.size <= 2) {
    showToast('ℹ️ Model already has minimal parts.', 'info');
    return;
  }
  let totalVerts = 0;
  partsData.forEach((info, m) => {
    if (m.geometry.attributes.position) totalVerts += m.geometry.attributes.position.count;
  });

  const smallMeshes = [];
  const largeMeshes = [];
  partsData.forEach((info, m) => {
    const vCount = m.geometry.attributes.position ? m.geometry.attributes.position.count : 0;
    if (vCount < totalVerts * 0.08) smallMeshes.push(m);
    else largeMeshes.push(m);
  });

  if (smallMeshes.length > 0 && largeMeshes.length > 0) {
    smallMeshes.forEach(smallM => {
      const sCenter = new THREE.Vector3();
      new THREE.Box3().setFromObject(smallM).getCenter(sCenter);
      let bestDist = Infinity;
      let bestLarge = largeMeshes[0];
      largeMeshes.forEach(lM => {
        const lCenter = new THREE.Vector3();
        new THREE.Box3().setFromObject(lM).getCenter(lCenter);
        const d = sCenter.distanceTo(lCenter);
        if (d < bestDist) {
          bestDist = d;
          bestLarge = lM;
        }
      });
      mergeMeshesIntoSinglePart([bestLarge, smallM]);
    });
    stageOperation(`Simple Tier: consolidated into ${partsData.size} parts`);
    showToast(`🧩 Simple Tier active: consolidated model into <strong>${partsData.size}</strong> primary parts!`, 'success');
  } else {
    showToast('ℹ️ Parts are already balanced.', 'info');
  }
}

const tierSimpleBtn = document.getElementById('tier-simple');
const tierBalancedBtn = document.getElementById('tier-balanced');
const tierDetailedBtn = document.getElementById('tier-detailed');
if (tierSimpleBtn) tierSimpleBtn.onclick = () => setPrecisionTier('simple');
if (tierBalancedBtn) tierBalancedBtn.onclick = () => setPrecisionTier('balanced');
if (tierDetailedBtn) tierDetailedBtn.onclick = () => setPrecisionTier('detailed');

// --- UNDO STACK & TRIPO BOTTOM CONFIRMATION BAR ---
const undoStack = [];

function saveStateToUndoStack(actionName = 'Edit') {
  if (partsData.size === 0) return;
  const snapshot = [];
  partsData.forEach((info, mesh) => {
    let hex = '#6366f1';
    if (mesh.material) {
      const m = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      if (m && m.color) hex = '#' + m.color.getHexString();
    }
    snapshot.push({
      name: info.name,
      index: info.index,
      geometry: mesh.geometry.clone(),
      hex: hex,
      position: mesh.position.clone(),
      quaternion: mesh.quaternion.clone(),
      scale: mesh.scale.clone(),
      originalLocalPosition: info.originalLocalPosition.clone(),
      originalRotation: info.originalRotation.clone(),
      originalScale: info.originalScale.clone(),
      explodeDirection: info.explodeDirection.clone(),
      distanceWeight: info.distanceWeight,
      geometryCenterOffset: info.geometryCenterOffset.clone()
    });
  });
  undoStack.push({ actionName, snapshot });
  if (undoStack.length > 12) undoStack.shift();
}

function stageOperation(actionDescription) {
  const confirmBar = document.getElementById('tripo-confirm-bar');
  const confirmText = document.getElementById('tripo-confirm-text');
  if (confirmBar && confirmText) {
    confirmText.innerText = actionDescription;
    confirmBar.style.display = 'flex';
  }
}

function restoreFromSnapshot(snapshot) {
  const parentObj = currentLoadedMesh || scene;
  partsData.forEach((info, mesh) => {
    if (mesh.parent) mesh.parent.remove(mesh);
    if (mesh.geometry) mesh.geometry.dispose();
  });
  partsData.clear();
  paletteMaterials.clear();
  originalMaterials.clear();

  snapshot.forEach(item => {
    const mat = new THREE.MeshStandardMaterial({
      color: item.hex,
      roughness: 0.35,
      metalness: 0.05,
      emissive: item.hex,
      emissiveIntensity: 0.15,
      side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(item.geometry, mat);
    mesh.name = item.name;
    mesh.position.copy(item.position);
    mesh.quaternion.copy(item.quaternion);
    mesh.scale.copy(item.scale);
    parentObj.add(mesh);

    const info = {
      mesh: mesh,
      index: item.index,
      name: item.name,
      originalLocalPosition: item.originalLocalPosition.clone(),
      originalRotation: item.originalRotation.clone(),
      originalScale: item.originalScale.clone(),
      explodeDirection: item.explodeDirection.clone(),
      distanceWeight: item.distanceWeight,
      geometryCenterOffset: item.geometryCenterOffset.clone(),
      accumulatedRotation: 0,
      rotationAxis: new THREE.Vector3(1, 0, 0),
      rotationSpeedFactor: 1.0,
      snapshotRotation: 0,
      customOffset: new THREE.Vector3(),
      customRotation: new THREE.Euler(),
      customScale: 1.0,
      visible: true
    };
    partsData.set(mesh, info);
    paletteMaterials.set(mesh, mat);
    originalMaterials.set(mesh, cloneMaterialSafe(mat));
  });

  rebuildPartsHierarchyUI();
  updateBrushReassignDropdown();
  if (currentLoadedMesh) updateMeshStatsDisplay(calculateGeometryStats(currentLoadedMesh));
}

const confirmBtn = document.getElementById('tripo-confirm-btn');
if (confirmBtn) {
  confirmBtn.onclick = () => {
    const confirmBar = document.getElementById('tripo-confirm-bar');
    if (confirmBar) confirmBar.style.display = 'none';
    showToast('✅ Changes confirmed and locked in Tripo workbench!', 'success');
  };
}

const undoBtn = document.getElementById('tripo-undo-btn');
if (undoBtn) {
  undoBtn.onclick = () => {
    if (undoStack.length === 0) {
      showToast('ℹ️ No previous state to undo.', 'info');
      return;
    }
    const last = undoStack.pop();
    restoreFromSnapshot(last.snapshot);
    const confirmBar = document.getElementById('tripo-confirm-bar');
    if (confirmBar) confirmBar.style.display = 'none';
    showToast(`↩️ Reverted: ${last.actionName}`, 'info');
  };
}

// --- 3D BRUSH WORKBENCH CONTROLS & SUBMODES ---
let brushSubMode = 'paint'; // 'paint' | 'erase'

const brushBtn = document.getElementById('tool-brush-btn');
const brushToolbar = document.getElementById('brush-toolbar');
const brushFacesBadge = document.getElementById('brush-faces-badge');
const brushModeSelect = document.getElementById('brush-mode-select');
const brushScopeSelect = document.getElementById('brush-scope-select');
const brushRadiusSlider = document.getElementById('brush-radius-slider');
const brushRadiusVal = document.getElementById('brush-radius-val');
const brushCutBtn = document.getElementById('brush-cut-btn');
const brushJoinBtn = document.getElementById('brush-join-btn');
const brushSemanticBtn = document.getElementById('brush-semantic-match-btn');
const brushInvertBtn = document.getElementById('brush-invert-btn');
const brushClearBtn = document.getElementById('brush-clear-btn');
const brushCloseBtn = document.getElementById('brush-close-btn');

const brushSubmodePaintBtn = document.getElementById('brush-submode-paint');
const brushSubmodeEraseBtn = document.getElementById('brush-submode-erase');
const brushReassignSelect = document.getElementById('brush-reassign-select');
const brushApplyReassignBtn = document.getElementById('brush-apply-reassign-btn');

function setBrushSubMode(mode) {
  brushSubMode = mode;
  if (brushSubmodePaintBtn) brushSubmodePaintBtn.classList.toggle('active', mode === 'paint');
  if (brushSubmodeEraseBtn) brushSubmodeEraseBtn.classList.toggle('active', mode === 'erase');
  
  if (brushHelperMat) {
    brushHelperMat.color.setHex(mode === 'paint' ? 0xf43f5e : 0xeab308);
  }
}

if (brushSubmodePaintBtn) brushSubmodePaintBtn.onclick = () => setBrushSubMode('paint');
if (brushSubmodeEraseBtn) brushSubmodeEraseBtn.onclick = () => setBrushSubMode('erase');

// Shift / Alt hotkey to erase while holding
window.addEventListener('keydown', (e) => {
  if (isBrushActive && (e.key === 'Shift' || e.key === 'Alt')) {
    setBrushSubMode('erase');
  }
});
window.addEventListener('keyup', (e) => {
  if (isBrushActive && (e.key === 'Shift' || e.key === 'Alt')) {
    setBrushSubMode('paint');
  }
});

// Update Target Part Dropdown in Brush Bar
function updateBrushReassignDropdown() {
  if (!brushReassignSelect) return;
  const currentVal = brushReassignSelect.value;
  brushReassignSelect.innerHTML = '<option value="new">+ New Separate Part</option>';
  let i = 1;
  partsData.forEach((info, m) => {
    const opt = document.createElement('option');
    opt.value = info.index;
    opt.textContent = `Part #${i}: ${info.name}`;
    brushReassignSelect.appendChild(opt);
    i++;
  });
  if (currentVal) brushReassignSelect.value = currentVal;
}

// Invert Selection
if (brushInvertBtn) {
  brushInvertBtn.onclick = () => {
    const meshes = (brushScope === 'part' && brushLockedMesh) 
      ? [brushLockedMesh] 
      : Array.from(partsData.keys());
    
    if (meshes.length === 0) return;
    
    meshes.forEach(mesh => {
      if (!selectedMeshFaces.has(mesh)) selectedMeshFaces.set(mesh, new Set());
      const set = selectedMeshFaces.get(mesh);
      const totalFaces = mesh.geometry.index ? (mesh.geometry.index.count / 3) : (mesh.geometry.attributes.position.count / 3);
      for (let f = 0; f < totalFaces; f++) {
        if (set.has(f)) set.delete(f);
        else set.add(f);
      }
    });
    updateBrushOverlay();
    showToast('🔄 Inverted selection on model surfaces!', 'info');
  };
}

// Reassign Brushed Faces into Target Part or New Part
function reassignBrushedFaces() {
  const targetVal = brushReassignSelect ? brushReassignSelect.value : 'new';
  if (targetVal === 'new') {
    cutOrDivideBrushedParts();
    return;
  }
  
  const targetIndex = parseInt(targetVal);
  let targetMesh = null;
  for (const [m, info] of partsData.entries()) {
    if (info.index === targetIndex) {
      targetMesh = m;
      break;
    }
  }
  
  if (!targetMesh) {
    showToast('⚠️ Target part not found.', 'warning');
    return;
  }

  const activeMeshes = Array.from(selectedMeshFaces.keys()).filter(m => selectedMeshFaces.get(m).size > 0);
  if (activeMeshes.length === 0) {
    showToast('⚠️ Please brush faces on the model to reassign them to this part!', 'warning');
    return;
  }

  resetExplosionToRest();
  saveStateToUndoStack(`Reassign brushed faces to ${partsData.get(targetMesh).name}`);

  const piecesToMerge = [targetMesh];
  activeMeshes.forEach(mesh => {
    if (mesh === targetMesh) return;
    const faceSet = selectedMeshFaces.get(mesh);
    const totalFaces = mesh.geometry.index ? (mesh.geometry.index.count / 3) : (mesh.geometry.attributes.position.count / 3);
    
    if (faceSet.size >= totalFaces) {
      piecesToMerge.push(mesh);
    } else {
      const { selectedGeo, remainingGeo } = partitionGeometry(mesh.geometry, faceSet);
      mesh.geometry.dispose();
      mesh.geometry = remainingGeo;
      mesh.geometry.computeBoundingBox();
      mesh.geometry.computeBoundingSphere();
      mesh.updateMatrixWorld(true);
      
      const pieceMesh = new THREE.Mesh(selectedGeo, targetMesh.material);
      pieceMesh.position.copy(mesh.position);
      pieceMesh.quaternion.copy(mesh.quaternion);
      pieceMesh.scale.copy(mesh.scale);
      (mesh.parent || currentLoadedMesh || scene).add(pieceMesh);
      piecesToMerge.push(pieceMesh);
    }
  });

  if (piecesToMerge.length > 1) {
    const targetInfo = partsData.get(targetMesh);
    const targetName = targetInfo.name;
    const targetHex = '#' + (Array.isArray(targetMesh.material) ? targetMesh.material[0].color : targetMesh.material.color).getHexString();
    mergeMeshesIntoSinglePart(piecesToMerge, targetName, targetHex);
    showToast(`🎯 Reassigned brushed faces directly into <strong>"${targetName}"</strong>!`, 'success');
    stageOperation(`Reassigned faces to "${targetName}"`);
  }
  clearBrushSelection();
}

if (brushApplyReassignBtn) brushApplyReassignBtn.onclick = () => reassignBrushedFaces();

function toggleBrushMode(forceState = null) {
  isBrushActive = (forceState !== null) ? forceState : !isBrushActive;
  if (brushBtn) brushBtn.classList.toggle('active', isBrushActive);
  if (brushToolbar) brushToolbar.style.display = isBrushActive ? 'flex' : 'none';
  brushHelper.visible = false;

  if (isBrushActive) {
    resetExplosionToRest();
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.NONE, // In brush mode, left mouse button paints without camera rotation!
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.ROTATE
    };
  } else {
    clearBrushSelection();
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    };
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
    if (brushScope === 'multi') {
      brushLockedMesh = null;
    }
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

function clearBrushSelection() {
  selectedMeshFaces.clear();
  brushLockedMesh = null;
  isBrushing = false;
  controls.enabled = true;

  brushOverlayMeshes.forEach((oMesh) => {
    oMesh.visible = false;
    if (oMesh.parent) oMesh.parent.remove(oMesh);
    if (oMesh.geometry) oMesh.geometry.dispose();
  });
  brushOverlayMeshes.clear();

  if (brushFacesBadge) brushFacesBadge.innerText = '0 faces (0 parts)';
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
  }, 3200);
}

// Mesh Adjacency Cache for Semantic Feature Flood-Fill and Connected Components
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

// Find Connected Triangle Components for Dividing Parts into Multiple Pieces
function getConnectedComponents(geo) {
  const isIndexed = geo.index !== null;
  const index = geo.index;
  const pos = geo.attributes.position;
  const totalFaces = isIndexed ? (index.count / 3) : (pos.count / 3);

  const adj = getMeshAdjacency(geo);
  const visited = new Uint8Array(totalFaces);
  const components = [];

  for (let i = 0; i < totalFaces; i++) {
    if (visited[i]) continue;
    const compFaces = [];
    const queue = [i];
    visited[i] = 1;

    while (queue.length > 0) {
      const f = queue.shift();
      compFaces.push(f);
      const neighbors = adj.faceNeighbors[f];
      for (const nb of neighbors) {
        if (!visited[nb]) {
          visited[nb] = 1;
          queue.push(nb);
        }
      }
    }
    components.push(compFaces);
  }

  return components;
}

// 1-Click Semantic Feature Flood-Fill (< 45 deg crease angle)
function floodFillSemanticFeature(mesh, startFaceIndex) {
  if (!mesh || !mesh.geometry) return;
  const adj = getMeshAdjacency(mesh.geometry);
  if (!adj || startFaceIndex < 0 || startFaceIndex >= adj.totalFaces) return;

  if (!selectedMeshFaces.has(mesh)) selectedMeshFaces.set(mesh, new Set());
  const faceSet = selectedMeshFaces.get(mesh);

  const startNorm = adj.faceNormals[startFaceIndex];
  const queue = [startFaceIndex];
  const visited = new Set([startFaceIndex]);
  faceSet.add(startFaceIndex);

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
          faceSet.add(nb);
          queue.push(nb);
        }
      }
    }
  }

  updateBrushOverlay();
}

// Multi-Part Unrestricted Surface Paint
function paintFacesOnMesh(mesh, worldHitPoint, worldHitNormal) {
  if (!mesh || !mesh.geometry) return;
  const geo = mesh.geometry;
  const isIndexed = geo.index !== null;
  const index = geo.index;
  const pos = geo.attributes.position;
  const totalFaces = isIndexed ? (index.count / 3) : (pos.count / 3);

  if (!selectedMeshFaces.has(mesh)) {
    selectedMeshFaces.set(mesh, new Set());
  }
  const faceSet = selectedMeshFaces.get(mesh);

  mesh.updateMatrixWorld(true);
  const localHitPoint = mesh.worldToLocal(worldHitPoint.clone());
  const scale = mesh.getWorldScale(new THREE.Vector3());
  const avgScale = (scale.x + scale.y + scale.z) / 3 || 1;
  const localRadius = brushRadius / Math.max(avgScale, 1e-4);
  const r2 = localRadius * localRadius;

  const minX = localHitPoint.x - localRadius;
  const maxX = localHitPoint.x + localRadius;
  const minY = localHitPoint.y - localRadius;
  const maxY = localHitPoint.y + localRadius;
  const minZ = localHitPoint.z - localRadius;
  const maxZ = localHitPoint.z + localRadius;

  let localHitNorm = null;
  if (worldHitNormal) {
    const normalMatrix = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
    localHitNorm = worldHitNormal.clone().applyMatrix3(normalMatrix).normalize();
  }

  const pA = new THREE.Vector3();
  const pB = new THREE.Vector3();
  const pC = new THREE.Vector3();
  const centroid = new THREE.Vector3();
  const cb = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const fn = new THREE.Vector3();

  let changed = false;

  for (let f = 0; f < totalFaces; f++) {
    const isSel = faceSet.has(f);
    if (brushSubMode === 'paint' && isSel) continue;
    if (brushSubMode === 'erase' && !isSel) continue;

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

    const dMin2 = Math.min(
      centroid.distanceToSquared(localHitPoint),
      pA.distanceToSquared(localHitPoint),
      pB.distanceToSquared(localHitPoint),
      pC.distanceToSquared(localHitPoint)
    );

    if (dMin2 <= r2) {
      if (localHitNorm) {
        cb.subVectors(pC, pB);
        ab.subVectors(pA, pB);
        fn.crossVectors(cb, ab).normalize();
        if (fn.dot(localHitNorm) < -0.2) {
          continue; // Don't paint opposite side backfaces
        }
      }

      if (brushSubMode === 'paint') {
        faceSet.add(f);
        changed = true;
      } else if (brushSubMode === 'erase') {
        faceSet.delete(f);
        changed = true;
      }
    }
  }

  if (changed) {
    updateBrushOverlay();
  }
}

function getTotalSelectedFaces() {
  let count = 0;
  selectedMeshFaces.forEach(set => { count += set.size; });
  return count;
}

// Synchronize Multi-Part 3D Visual Mesh Overlay and Badge Counter
function updateBrushOverlay() {
  const total = getTotalSelectedFaces();
  const partCount = Array.from(selectedMeshFaces.values()).filter(s => s && s.size > 0).length;
  if (brushFacesBadge) {
    brushFacesBadge.innerText = `${total} faces (${partCount} part${partCount !== 1 ? 's' : ''})`;
  }

  selectedMeshFaces.forEach((faceSet, mesh) => {
    let overlayMesh = brushOverlayMeshes.get(mesh);
    if (!faceSet || faceSet.size === 0) {
      if (overlayMesh) {
        overlayMesh.visible = false;
        if (overlayMesh.parent) overlayMesh.parent.remove(overlayMesh);
      }
      return;
    }

    const geo = mesh.geometry;
    const isIndexed = geo.index !== null;
    const index = geo.index;
    const pos = geo.attributes.position;
    const norm = geo.attributes.normal;

    const S = faceSet.size;
    const selPos = new Float32Array(S * 9);
    const selNorm = norm ? new Float32Array(S * 9) : null;

    let sIdx = 0;
    for (const f of faceSet) {
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

    if (!overlayMesh) {
      const oGeo = new THREE.BufferGeometry();
      overlayMesh = new THREE.Mesh(oGeo, brushSelectionMat);
      overlayMesh.renderOrder = 998;
      overlayMesh.raycast = function() {}; // Prevent raycast interference
      overlayMesh.isBrushOverlay = true;
      brushOverlayMeshes.set(mesh, overlayMesh);
    }

    overlayMesh.geometry.dispose();
    const newGeo = new THREE.BufferGeometry();
    newGeo.setAttribute('position', new THREE.BufferAttribute(selPos, 3));
    if (selNorm) {
      newGeo.setAttribute('normal', new THREE.BufferAttribute(selNorm, 3));
    } else {
      newGeo.computeVertexNormals();
    }
    overlayMesh.geometry = newGeo;

    if (overlayMesh.parent !== mesh) {
      mesh.add(overlayMesh);
    }
    overlayMesh.position.set(0, 0, 0);
    overlayMesh.rotation.set(0, 0, 0);
    overlayMesh.scale.set(1, 1, 1);
    overlayMesh.visible = true;
  });

  // Clean up any stale overlays
  brushOverlayMeshes.forEach((oMesh, m) => {
    if (!selectedMeshFaces.has(m) || selectedMeshFaces.get(m).size === 0) {
      oMesh.visible = false;
      if (oMesh.parent) oMesh.parent.remove(oMesh);
    }
  });
}

// Global Semantic Feature Auto-Detection across the whole object
function detectAndSelectGlobalSemanticFeatures() {
  const activeMeshes = Array.from(selectedMeshFaces.keys()).filter(m => selectedMeshFaces.get(m).size > 0);
  if (activeMeshes.length === 0) {
    showToast('⚠️ Please brush an area or feature first so we can analyze its semantic features!', 'warning');
    return;
  }

  // Calculate global bounding box & aspect signature of brushed area
  const combinedBox = new THREE.Box3();
  const pVec = new THREE.Vector3();

  activeMeshes.forEach(mesh => {
    const geo = mesh.geometry;
    const isIndexed = geo.index !== null;
    const index = geo.index;
    const pos = geo.attributes.position;
    const faceSet = selectedMeshFaces.get(mesh);

    mesh.updateMatrixWorld(true);

    for (const f of faceSet) {
      const i0 = isIndexed ? index.getX(f * 3) : f * 3;
      const i1 = isIndexed ? index.getX(f * 3 + 1) : f * 3 + 1;
      const i2 = isIndexed ? index.getX(f * 3 + 2) : f * 3 + 2;

      for (const vi of [i0, i1, i2]) {
        pVec.fromBufferAttribute(pos, vi);
        pVec.applyMatrix4(mesh.matrixWorld);
        combinedBox.expandByPoint(pVec);
      }
    }
  });

  const bSize = new THREE.Vector3();
  combinedBox.getSize(bSize);
  const bCenter = new THREE.Vector3();
  combinedBox.getCenter(bCenter);

  // Symmetrical reflection coordinates across global center
  const gCenter = globalCenter || new THREE.Vector3(0, 0, 0);
  const symX = new THREE.Vector3(- (bCenter.x - gCenter.x) + gCenter.x, bCenter.y, bCenter.z);
  const symZ = new THREE.Vector3(bCenter.x, bCenter.y, - (bCenter.z - gCenter.z) + gCenter.z);
  const symXZ = new THREE.Vector3(- (bCenter.x - gCenter.x) + gCenter.x, bCenter.y, - (bCenter.z - gCenter.z) + gCenter.z);

  const sortedDims = [bSize.x, bSize.y, bSize.z].sort((a, b) => a - b);
  let matchedCount = 0;

  // Scan all parts across the 3D model
  partsData.forEach((info, candidateMesh) => {
    if (activeMeshes.includes(candidateMesh) && selectedMeshFaces.get(candidateMesh).size > 10) return;

    candidateMesh.updateMatrixWorld(true);
    const cBox = new THREE.Box3().setFromObject(candidateMesh);
    const cSize = new THREE.Vector3();
    cBox.getSize(cSize);
    const cCenter = new THREE.Vector3();
    cBox.getCenter(cCenter);

    const cSorted = [cSize.x, cSize.y, cSize.z].sort((a, b) => a - b);

    // Dimension similarity check (within 38%)
    const dimMatch = Math.abs(cSorted[0] - sortedDims[0]) / Math.max(sortedDims[0], 0.02) < 0.38 &&
                     Math.abs(cSorted[1] - sortedDims[1]) / Math.max(sortedDims[1], 0.02) < 0.38 &&
                     Math.abs(cSorted[2] - sortedDims[2]) / Math.max(sortedDims[2], 0.02) < 0.38;

    // Symmetry check
    const symDistX = cCenter.distanceTo(symX);
    const symDistZ = cCenter.distanceTo(symZ);
    const symDistXZ = cCenter.distanceTo(symXZ);
    const isSymmetrical = symDistX < 0.18 || symDistZ < 0.18 || symDistXZ < 0.18;

    // Elevation / Y match
    const yMatch = Math.abs(cCenter.y - bCenter.y) < 0.20;

    if ((dimMatch && (isSymmetrical || yMatch)) || (isSymmetrical && yMatch)) {
      if (!selectedMeshFaces.has(candidateMesh)) selectedMeshFaces.set(candidateMesh, new Set());
      const cSet = selectedMeshFaces.get(candidateMesh);
      const totalFaces = candidateMesh.geometry.index
        ? (candidateMesh.geometry.index.count / 3)
        : (candidateMesh.geometry.attributes.position.count / 3);
      for (let f = 0; f < totalFaces; f++) cSet.add(f);
      matchedCount++;
    }
  });

  updateBrushOverlay();
  showToast(`🌐 Semantic Match: auto-selected <strong>${matchedCount}</strong> matching component(s) across the global object!`, 'success');
}

if (brushSemanticBtn) brushSemanticBtn.onclick = () => detectAndSelectGlobalSemanticFeatures();

// Extract arbitrary face list into independent non-indexed BufferGeometry
function extractSubGeometry(sourceGeo, faceList) {
  const isIndexed = sourceGeo.index !== null;
  const index = sourceGeo.index;
  const pos = sourceGeo.attributes.position;
  const norm = sourceGeo.attributes.normal;
  const uv = sourceGeo.attributes.uv;
  const color = sourceGeo.attributes.color;

  const S = faceList.length;
  const selPos = new Float32Array(S * 9);
  const selNorm = norm ? new Float32Array(S * 9) : null;
  const selUv = uv ? new Float32Array(S * 6) : null;
  const selCol = color ? new Float32Array(S * 3 * color.itemSize) : null;

  for (let sIdx = 0; sIdx < S; sIdx++) {
    const f = faceList[sIdx];
    const i0 = isIndexed ? index.getX(f * 3) : f * 3;
    const i1 = isIndexed ? index.getX(f * 3 + 1) : f * 3 + 1;
    const i2 = isIndexed ? index.getX(f * 3 + 2) : f * 3 + 2;
    const vertIndices = [i0, i1, i2];

    for (let v = 0; v < 3; v++) {
      const vi = vertIndices[v];
      const pBase = sIdx * 9 + v * 3;
      selPos[pBase] = pos.getX(vi);
      selPos[pBase + 1] = pos.getY(vi);
      selPos[pBase + 2] = pos.getZ(vi);

      if (norm && selNorm) {
        selNorm[pBase] = norm.getX(vi);
        selNorm[pBase + 1] = norm.getY(vi);
        selNorm[pBase + 2] = norm.getZ(vi);
      }

      if (uv && selUv) {
        const uvBase = sIdx * 6 + v * 2;
        selUv[uvBase] = uv.getX(vi);
        selUv[uvBase + 1] = uv.getY(vi);
      }

      if (color && selCol) {
        const cBase = sIdx * 3 * color.itemSize + v * color.itemSize;
        const viBase = vi * color.itemSize;
        for (let c = 0; c < color.itemSize; c++) {
          selCol[cBase + c] = color.array[viBase + c];
        }
      }
    }
  }

  const subGeo = new THREE.BufferGeometry();
  subGeo.setAttribute('position', new THREE.BufferAttribute(selPos, 3));
  if (selNorm) subGeo.setAttribute('normal', new THREE.BufferAttribute(selNorm, 3));
  if (selUv) subGeo.setAttribute('uv', new THREE.BufferAttribute(selUv, 2));
  if (selCol) subGeo.setAttribute('color', new THREE.BufferAttribute(selCol, color.itemSize, color.normalized));

  if (!selNorm) subGeo.computeVertexNormals();
  subGeo.computeBoundingBox();
  subGeo.computeBoundingSphere();

  return subGeo;
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
        const viBase = vi * color.itemSize;
        for (let c = 0; c < color.itemSize; c++) {
          targetCol[cBase + c] = color.array[viBase + c];
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
  if (selCol) selectedGeo.setAttribute('color', new THREE.BufferAttribute(selCol, color.itemSize, color.normalized));

  const remainingGeo = new THREE.BufferGeometry();
  remainingGeo.setAttribute('position', new THREE.BufferAttribute(remPos, 3));
  if (remNorm) remainingGeo.setAttribute('normal', new THREE.BufferAttribute(remNorm, 3));
  if (remUv) remainingGeo.setAttribute('uv', new THREE.BufferAttribute(remUv, 2));
  if (remCol) remainingGeo.setAttribute('color', new THREE.BufferAttribute(remCol, color.itemSize, color.normalized));

  if (!selNorm) selectedGeo.computeVertexNormals();
  if (!remNorm) remainingGeo.computeVertexNormals();

  selectedGeo.computeBoundingBox();
  selectedGeo.computeBoundingSphere();
  remainingGeo.computeBoundingBox();
  remainingGeo.computeBoundingSphere();

  return { selectedGeo, remainingGeo };
}

// FEATURE 1: JOIN / MERGE Multiple Segmented Parts into One Unified Part
function mergeMeshesIntoSinglePart(meshesToMerge, customName = null, customHex = null) {
  if (!meshesToMerge || meshesToMerge.length < 2) return null;

  resetExplosionToRest();
  saveStateToUndoStack(`Combine ${meshesToMerge.length} parts`);

  const parentObj = meshesToMerge[0].parent || currentLoadedMesh || scene;
  parentObj.updateMatrixWorld(true);
  const parentWorldInv = parentObj.matrixWorld.clone().invert();

  // Transform all meshes to common parent coordinates
  const transformedGeos = [];
  meshesToMerge.forEach(mesh => {
    mesh.updateMatrixWorld(true);
    const toParentMatrix = parentWorldInv.clone().multiply(mesh.matrixWorld);
    const gClone = mesh.geometry.clone();
    gClone.applyMatrix4(toParentMatrix);
    const nonIndexed = gClone.index ? gClone.toNonIndexed() : gClone;
    transformedGeos.push(nonIndexed);
  });

  let totalVerts = 0;
  transformedGeos.forEach(g => { totalVerts += g.attributes.position.count; });

  const mergedPos = new Float32Array(totalVerts * 3);
  const hasNorm = transformedGeos.every(g => g.attributes.normal);
  const mergedNorm = hasNorm ? new Float32Array(totalVerts * 3) : null;
  const hasUv = transformedGeos.every(g => g.attributes.uv);
  const mergedUv = hasUv ? new Float32Array(totalVerts * 2) : null;

  let offset = 0;
  transformedGeos.forEach(g => {
    mergedPos.set(g.attributes.position.array, offset * 3);
    if (mergedNorm && g.attributes.normal) {
      mergedNorm.set(g.attributes.normal.array, offset * 3);
    }
    if (mergedUv && g.attributes.uv) {
      mergedUv.set(g.attributes.uv.array, offset * 2);
    }
    offset += g.attributes.position.count;
  });

  const mergedGeo = new THREE.BufferGeometry();
  mergedGeo.setAttribute('position', new THREE.BufferAttribute(mergedPos, 3));
  if (mergedNorm) {
    mergedGeo.setAttribute('normal', new THREE.BufferAttribute(mergedNorm, 3));
  } else {
    mergedGeo.computeVertexNormals();
  }
  if (mergedUv) {
    mergedGeo.setAttribute('uv', new THREE.BufferAttribute(mergedUv, 2));
  }

  mergedGeo.computeBoundingBox();
  mergedGeo.computeBoundingSphere();

  const newHex = customHex || getNextDistinctPaletteColor();
  const joinedMat = new THREE.MeshStandardMaterial({
    color: newHex,
    roughness: 0.35,
    metalness: 0.05,
    emissive: newHex,
    emissiveIntensity: 0.15,
    side: THREE.DoubleSide
  });

  const nextIdx = partsData.size;
  const joinedName = customName || `joined_part_${nextIdx + 1}`;
  const joinedMesh = new THREE.Mesh(mergedGeo, joinedMat);
  joinedMesh.name = joinedName;

  // Remove old individual meshes
  meshesToMerge.forEach(oldMesh => {
    if (oldMesh.parent) oldMesh.parent.remove(oldMesh);
    partsData.delete(oldMesh);
    paletteMaterials.delete(oldMesh);
    originalMaterials.delete(oldMesh);
    if (oldMesh.geometry) oldMesh.geometry.dispose();
  });

  parentObj.add(joinedMesh);
  joinedMesh.updateMatrixWorld(true);

  // Joined part center & explosion direction
  const jBox = new THREE.Box3().setFromObject(joinedMesh);
  const jCenter = new THREE.Vector3();
  jBox.getCenter(jCenter);
  const jDir = jCenter.clone().sub(globalCenter);
  if (jDir.length() < 1e-4) jDir.set(0, 1, 0);
  jDir.normalize();

  const joinedPartInfo = {
    mesh: joinedMesh,
    index: nextIdx,
    name: joinedName,
    originalLocalPosition: joinedMesh.position.clone(),
    originalRotation: joinedMesh.quaternion.clone(),
    originalScale: joinedMesh.scale.clone(),
    explodeDirection: jDir,
    distanceWeight: 1.0,
    geometryCenterOffset: jCenter.clone(),
    accumulatedRotation: 0,
    rotationAxis: new THREE.Vector3(1, 0, 0),
    rotationSpeedFactor: 1.0,
    snapshotRotation: 0,
    customOffset: new THREE.Vector3(),
    customRotation: new THREE.Euler(),
    customScale: 1.0,
    visible: true
  };

  partsData.set(joinedMesh, joinedPartInfo);
  paletteMaterials.set(joinedMesh, joinedMat);
  originalMaterials.set(joinedMesh, cloneMaterialSafe(joinedMat));

  clearBrushSelection();
  rebuildPartsHierarchyUI();
  selectPart(joinedMesh);
  if (currentLoadedMesh) updateMeshStatsDisplay(calculateGeometryStats(currentLoadedMesh));

  stageOperation(`Combined ${meshesToMerge.length} parts into "${joinedName}"`);
  showToast(`🔗 Successfully joined <strong>${meshesToMerge.length}</strong> parts into <em>"${joinedName}"</em>!`, 'success');
  return joinedMesh;
}

function joinSelectedParts(targetMeshes = null) {
  resetExplosionToRest();

  let meshesToJoin = targetMeshes;
  const activeBrushedMeshes = Array.from(selectedMeshFaces.keys()).filter(m => selectedMeshFaces.get(m).size > 0);

  // Case 1: User brushed faces across 2 or more parts
  if ((!meshesToJoin || meshesToJoin.length === 0) && activeBrushedMeshes.length >= 2) {
    saveStateToUndoStack('Combine brushed regions across parts');
    const piecesToMerge = [];

    activeBrushedMeshes.forEach(mesh => {
      const faceSet = selectedMeshFaces.get(mesh);
      const srcGeo = mesh.geometry;
      const isIndexed = srcGeo.index !== null;
      const totalFaces = isIndexed ? (srcGeo.index.count / 3) : (srcGeo.attributes.position.count / 3);
      const parentObj = mesh.parent || currentLoadedMesh || scene;

      if (faceSet.size >= totalFaces) {
        // All faces brushed on this mesh: include the entire mesh
        piecesToMerge.push(mesh);
      } else {
        // Carve out brushed faces
        const { selectedGeo, remainingGeo } = partitionGeometry(srcGeo, faceSet);
        mesh.geometry.dispose();
        mesh.geometry = remainingGeo;
        mesh.geometry.computeBoundingBox();
        mesh.geometry.computeBoundingSphere();
        mesh.updateMatrixWorld(true);

        const pieceMesh = new THREE.Mesh(selectedGeo, mesh.material);
        pieceMesh.position.copy(mesh.position);
        pieceMesh.quaternion.copy(mesh.quaternion);
        pieceMesh.scale.copy(mesh.scale);
        parentObj.add(pieceMesh);
        piecesToMerge.push(pieceMesh);
      }
    });

    const newJoinedMesh = mergeMeshesIntoSinglePart(piecesToMerge, 'combined_part');
    clearBrushSelection();
    checkedParts.clear();
    updateBulkActionBar();
    return newJoinedMesh;
  }

  // Case 2: User checked 2+ parts in hierarchy
  if (!meshesToJoin || meshesToJoin.length === 0) {
    if (checkedParts.size >= 2) {
      meshesToJoin = Array.from(checkedParts);
    }
  }

  if (!meshesToJoin || meshesToJoin.length < 2) {
    showToast('⚠️ Please select or brush across at least 2 parts to combine them!', 'warning');
    return null;
  }

  const newJoinedMesh = mergeMeshesIntoSinglePart(meshesToJoin);
  checkedParts.clear();
  updateBulkActionBar();
  return newJoinedMesh;
}

if (brushJoinBtn) brushJoinBtn.onclick = () => joinSelectedParts();

// FEATURE 2: SEPARATE / CUT / DIVIDE Segmented Parts into Multiple Independent Pieces with DIFFERENT VIBRANT COLORS
function bisectMeshIntoTwoPieces(targetMesh) {
  if (!targetMesh || !targetMesh.geometry) return false;
  const geo = targetMesh.geometry;
  const isIndexed = geo.index !== null;
  const index = geo.index;
  const pos = geo.attributes.position;
  const totalFaces = isIndexed ? (index.count / 3) : (pos.count / 3);
  if (totalFaces < 2) return false;

  geo.computeBoundingBox();
  const box = geo.boundingBox;
  const size = new THREE.Vector3().subVectors(box.max, box.min);

  let axis = 'x';
  if (size.y >= size.x && size.y >= size.z) axis = 'y';
  else if (size.z >= size.x && size.z >= size.y) axis = 'z';

  const mid = (box.min[axis] + box.max[axis]) / 2;

  const setA = new Set();
  for (let f = 0; f < totalFaces; f++) {
    const i0 = isIndexed ? index.getX(f * 3) : f * 3;
    const i1 = isIndexed ? index.getX(f * 3 + 1) : f * 3 + 1;
    const i2 = isIndexed ? index.getX(f * 3 + 2) : f * 3 + 2;
    const v0 = pos[axis === 'x' ? 'getX' : (axis === 'y' ? 'getY' : 'getZ')](i0);
    const v1 = pos[axis === 'x' ? 'getX' : (axis === 'y' ? 'getY' : 'getZ')](i1);
    const v2 = pos[axis === 'x' ? 'getX' : (axis === 'y' ? 'getY' : 'getZ')](i2);
    const centerAxis = (v0 + v1 + v2) / 3;
    if (centerAxis < mid) {
      setA.add(f);
    }
  }

  if (setA.size === 0 || setA.size === totalFaces) {
    setA.clear();
    const half = Math.floor(totalFaces / 2);
    for (let f = 0; f < half; f++) setA.add(f);
  }

  const { selectedGeo: geo1, remainingGeo: geo2 } = partitionGeometry(geo, setA);

  const parentInfo = partsData.get(targetMesh);
  const baseName = parentInfo ? parentInfo.name : 'part';
  const parentObj = targetMesh.parent || currentLoadedMesh || scene;

  parentObj.remove(targetMesh);
  partsData.delete(targetMesh);
  paletteMaterials.delete(targetMesh);
  originalMaterials.delete(targetMesh);

  const newMeshes = [];
  [geo1, geo2].forEach((pieceGeo, pIdx) => {
    const col = getNextDistinctPaletteColor();
    const mat = new THREE.MeshStandardMaterial({
      color: col,
      roughness: 0.35,
      metalness: 0.05,
      emissive: col,
      emissiveIntensity: 0.15,
      side: THREE.DoubleSide
    });
    const nextIdx = partsData.size;
    const pieceName = `${baseName}_half_${pIdx + 1}`;
    const pMesh = new THREE.Mesh(pieceGeo, mat);
    pMesh.name = pieceName;
    pMesh.position.copy(targetMesh.position);
    pMesh.quaternion.copy(targetMesh.quaternion);
    pMesh.scale.copy(targetMesh.scale);
    parentObj.add(pMesh);

    pMesh.updateMatrixWorld(true);
    const pBox = new THREE.Box3().setFromObject(pMesh);
    const pCenter = new THREE.Vector3();
    pBox.getCenter(pCenter);
    const pDir = pCenter.clone().sub(globalCenter);
    if (pDir.length() < 1e-4) pDir.set(0, 1, 0);
    pDir.normalize();

    const info = {
      mesh: pMesh,
      index: nextIdx,
      name: pieceName,
      originalLocalPosition: pMesh.position.clone(),
      originalRotation: pMesh.quaternion.clone(),
      originalScale: pMesh.scale.clone(),
      explodeDirection: pDir,
      distanceWeight: 1.0,
      geometryCenterOffset: pCenter.clone(),
      accumulatedRotation: 0,
      rotationAxis: new THREE.Vector3(1, 0, 0),
      rotationSpeedFactor: 1.0,
      snapshotRotation: 0,
      customOffset: new THREE.Vector3(),
      customRotation: new THREE.Euler(),
      customScale: 1.0,
      visible: true
    };
    partsData.set(pMesh, info);
    paletteMaterials.set(pMesh, mat);
    originalMaterials.set(pMesh, cloneMaterialSafe(mat));
    newMeshes.push(pMesh);
  });

  targetMesh.geometry.dispose();
  return newMeshes;
}

function dividePartIntoIndependentPieces(targetMesh) {
  if (!targetMesh || !targetMesh.geometry) return false;
  const components = getConnectedComponents(targetMesh.geometry);
  if (components.length >= 2) {
    return dividePartIntoConnectedComponents(targetMesh, false);
  } else {
    const res = bisectMeshIntoTwoPieces(targetMesh);
    return res ? true : false;
  }
}

// Cluster connected components that belong to the same brushed area
function clusterConnectedComponents(geo, components, maxClusterDist = 0.08) {
  if (!components || components.length <= 1) return components;

  const isIndexed = geo.index !== null;
  const index = geo.index;
  const pos = geo.attributes.position;

  // Compute 3D centroid for each component
  const compCenters = components.map(faceList => {
    let cx = 0, cy = 0, cz = 0, count = 0;
    for (const f of faceList) {
      const i0 = isIndexed ? index.getX(f * 3) : f * 3;
      const i1 = isIndexed ? index.getX(f * 3 + 1) : f * 3 + 1;
      const i2 = isIndexed ? index.getX(f * 3 + 2) : f * 3 + 2;
      for (const vi of [i0, i1, i2]) {
        cx += pos.getX(vi);
        cy += pos.getY(vi);
        cz += pos.getZ(vi);
        count++;
      }
    }
    return count > 0 ? new THREE.Vector3(cx / count, cy / count, cz / count) : new THREE.Vector3();
  });

  // Union components within maxClusterDist
  const parent = components.map((_, i) => i);
  function find(i) {
    if (parent[i] === i) return i;
    parent[i] = find(parent[i]);
    return parent[i];
  }
  function union(i, j) {
    const rootI = find(i);
    const rootJ = find(j);
    if (rootI !== rootJ) parent[rootI] = rootJ;
  }

  for (let i = 0; i < components.length; i++) {
    for (let j = i + 1; j < components.length; j++) {
      if (compCenters[i].distanceTo(compCenters[j]) <= maxClusterDist) {
        union(i, j);
      }
    }
  }

  const grouped = new Map();
  for (let i = 0; i < components.length; i++) {
    const root = find(i);
    if (!grouped.has(root)) grouped.set(root, []);
    grouped.get(root).push(...components[i]);
  }

  return Array.from(grouped.values());
}

function cutOrDivideBrushedParts() {
  resetExplosionToRest();

  const activeMeshes = Array.from(selectedMeshFaces.keys()).filter(m => selectedMeshFaces.get(m).size > 0);
  const checkedList = Array.from(checkedParts);

  // Require explicit brush selection or checked multi-shell parts
  if (activeMeshes.length === 0) {
    if (checkedList.length > 0) {
      let dividedAny = false;
      checkedList.forEach(targetMesh => {
        if (!targetMesh || !targetMesh.geometry || !partsData.has(targetMesh)) return;
        const comps = getConnectedComponents(targetMesh.geometry);
        if (comps.length >= 2) {
          dividePartIntoConnectedComponents(targetMesh, false);
          dividedAny = true;
        }
      });
      if (dividedAny) {
        checkedParts.clear();
        updateBulkActionBar();
        rebuildPartsHierarchyUI();
        return;
      }
    }
    showToast('⚠️ Please use the 3D Brush to paint the area or region you want to separate!', 'warning');
    return;
  }

  saveStateToUndoStack('Separate / Split selected region(s)');

  let createdCount = 0;
  const newlyCreatedMeshes = [];

  // 1. Separate ONLY the brushed faces from each touched mesh
  activeMeshes.forEach(parentMesh => {
    const faceSet = selectedMeshFaces.get(parentMesh);
    if (!faceSet || faceSet.size === 0) return;

    const srcGeo = parentMesh.geometry;
    const isIndexed = srcGeo.index !== null;
    const totalFaces = isIndexed ? (srcGeo.index.count / 3) : (srcGeo.attributes.position.count / 3);
    const parentInfo = partsData.get(parentMesh);
    const parentName = parentInfo ? parentInfo.name : 'part';
    const parentObj = parentMesh.parent || currentLoadedMesh || scene;

    const { selectedGeo, remainingGeo } = partitionGeometry(srcGeo, faceSet);

    // Identify distinct disjoint regions in selection (e.g. 2 knobs on the same drawer)
    const rawComponents = getConnectedComponents(selectedGeo);
    const clusterDist = Math.max(brushRadius * 1.25, 0.08);
    let selectedComponents = clusterConnectedComponents(selectedGeo, rawComponents, clusterDist);
    if (!selectedComponents || selectedComponents.length === 0) {
      selectedComponents = [Array.from({ length: faceSet.size }, (_, i) => i)];
    }

    // Separate EACH distinct brushed region as an independent part with its own VIBRANT COLOR
    selectedComponents.forEach((compFaces, cIdx) => {
      let pieceGeo = selectedComponents.length === 1
        ? selectedGeo
        : extractSubGeometry(selectedGeo, compFaces);

      const newColor = getNextDistinctPaletteColor();
      const newMat = new THREE.MeshStandardMaterial({
        color: newColor,
        roughness: 0.35,
        metalness: 0.05,
        emissive: newColor,
        emissiveIntensity: 0.15,
        side: THREE.DoubleSide
      });

      const nextIdx = partsData.size;
      const compSuffix = selectedComponents.length > 1 ? `_piece_${cIdx + 1}` : `_sep_${nextIdx + 1}`;
      const newName = `${parentName}${compSuffix}`;
      const newMesh = new THREE.Mesh(pieceGeo, newMat);
      newMesh.name = newName;

      newMesh.position.copy(parentMesh.position);
      newMesh.quaternion.copy(parentMesh.quaternion);
      newMesh.scale.copy(parentMesh.scale);

      parentObj.add(newMesh);

      newMesh.updateMatrixWorld(true);
      const nBox = new THREE.Box3().setFromObject(newMesh);
      const nCenter = new THREE.Vector3();
      nBox.getCenter(nCenter);
      const nDir = nCenter.clone().sub(globalCenter);
      if (nDir.length() < 1e-4) nDir.set(0, 1, 0);
      nDir.normalize();

      const newInfo = {
        mesh: newMesh,
        index: nextIdx,
        name: newName,
        originalLocalPosition: newMesh.position.clone(),
        originalRotation: newMesh.quaternion.clone(),
        originalScale: newMesh.scale.clone(),
        explodeDirection: nDir,
        distanceWeight: 1.0,
        geometryCenterOffset: nCenter.clone(),
        accumulatedRotation: 0,
        rotationAxis: new THREE.Vector3(1, 0, 0),
        rotationSpeedFactor: 1.0,
        snapshotRotation: 0,
        customOffset: new THREE.Vector3(),
        customRotation: new THREE.Euler(),
        customScale: 1.0,
        visible: true
      };

      partsData.set(newMesh, newInfo);
      paletteMaterials.set(newMesh, newMat);
      originalMaterials.set(newMesh, cloneMaterialSafe(newMat));
      newlyCreatedMeshes.push(newMesh);
      createdCount++;
    });

    // Handle parent mesh: PRESERVE intact with original color and identity!
    if (faceSet.size >= totalFaces) {
      // If 100% of faces were brushed, remove parent mesh
      parentObj.remove(parentMesh);
      partsData.delete(parentMesh);
      paletteMaterials.delete(parentMesh);
      originalMaterials.delete(parentMesh);
      srcGeo.dispose();
    } else {
      // ONLY the brushed region was carved out.
      // The parent mesh retains all remaining geometry with its original color and material!
      parentMesh.geometry.dispose();
      parentMesh.geometry = remainingGeo;
      parentMesh.geometry.computeBoundingBox();
      parentMesh.geometry.computeBoundingSphere();
      parentMesh.updateMatrixWorld(true);

      if (parentInfo) {
        const pBox = new THREE.Box3().setFromObject(parentMesh);
        const pCenter = new THREE.Vector3();
        pBox.getCenter(pCenter);
        parentInfo.geometryCenterOffset.copy(pCenter);
        const pDir = pCenter.clone().sub(globalCenter);
        if (pDir.length() < 1e-4) pDir.set(0, 1, 0);
        parentInfo.explodeDirection.copy(pDir.normalize());
      }
    }
  });

  clearBrushSelection();
  checkedParts.clear();
  updateBulkActionBar();
  rebuildPartsHierarchyUI();

  if (newlyCreatedMeshes.length > 0) {
    selectPart(newlyCreatedMeshes[0]);
  }
  if (currentLoadedMesh) {
    updateMeshStatsDisplay(calculateGeometryStats(currentLoadedMesh));
  }

  stageOperation(`Separated ${createdCount} region(s) into distinct part(s)`);
  showToast(`✂️ Separated selected region into <strong>${createdCount} independent part(s)</strong> with <strong>new vibrant color(s)</strong>!`, 'success');
}

if (brushCutBtn) brushCutBtn.onclick = () => cutOrDivideBrushedParts();

// FEATURE 2b: DIVIDE Part into Disjoint Connected Components with DIFFERENT COLORS
function dividePartIntoConnectedComponents(targetMesh, showInfoIfSingle = true) {
  if (!targetMesh || !targetMesh.geometry) return false;
  const components = getConnectedComponents(targetMesh.geometry);

  if (components.length < 2) {
    if (showInfoIfSingle) {
      showToast('ℹ️ This part is already a single continuous component.', 'info');
    }
    return false;
  }

  const parentInfo = partsData.get(targetMesh);
  const baseName = parentInfo ? parentInfo.name : 'part';
  const parentObj = targetMesh.parent || currentLoadedMesh || scene;

  saveStateToUndoStack(`Divide "${baseName}" into components`);

  parentObj.remove(targetMesh);
  partsData.delete(targetMesh);
  paletteMaterials.delete(targetMesh);
  originalMaterials.delete(targetMesh);

  const newMeshes = [];

  components.forEach((faceList, cIdx) => {
    const compGeo = extractSubGeometry(targetMesh.geometry, faceList);
    // EACH piece is assigned a DIFFERENT vibrant color from TRIPO_PALETTE!
    const compColor = getNextDistinctPaletteColor();
    const compMat = new THREE.MeshStandardMaterial({
      color: compColor,
      roughness: 0.35,
      metalness: 0.05,
      emissive: compColor,
      emissiveIntensity: 0.15,
      side: THREE.DoubleSide
    });

    const compMesh = new THREE.Mesh(compGeo, compMat);
    const compIdx = partsData.size;
    const compName = `${baseName}_div_${cIdx + 1}`;
    compMesh.name = compName;

    compMesh.position.copy(targetMesh.position);
    compMesh.quaternion.copy(targetMesh.quaternion);
    compMesh.scale.copy(targetMesh.scale);

    parentObj.add(compMesh);

    compMesh.updateMatrixWorld(true);
    const cBox = new THREE.Box3().setFromObject(compMesh);
    const cCenter = new THREE.Vector3();
    cBox.getCenter(cCenter);
    const cDir = cCenter.clone().sub(globalCenter);
    if (cDir.length() < 1e-4) cDir.set(0, 1, 0);
    cDir.normalize();

    const compInfo = {
      mesh: compMesh,
      index: compIdx,
      name: compName,
      originalLocalPosition: compMesh.position.clone(),
      originalRotation: compMesh.quaternion.clone(),
      originalScale: compMesh.scale.clone(),
      explodeDirection: cDir,
      distanceWeight: 1.0,
      geometryCenterOffset: cCenter.clone(),
      accumulatedRotation: 0,
      rotationAxis: new THREE.Vector3(1, 0, 0),
      rotationSpeedFactor: 1.0,
      snapshotRotation: 0,
      customOffset: new THREE.Vector3(),
      customRotation: new THREE.Euler(),
      customScale: 1.0,
      visible: true
    };

    partsData.set(compMesh, compInfo);
    paletteMaterials.set(compMesh, compMat);
    originalMaterials.set(compMesh, cloneMaterialSafe(compMat));
    newMeshes.push(compMesh);
  });

  targetMesh.geometry.dispose();

  clearBrushSelection();
  rebuildPartsHierarchyUI();
  if (newMeshes.length > 0) selectPart(newMeshes[0]);
  stageOperation(`Divided "${baseName}" into ${components.length} components`);
  showToast(`⚡ Divided <em>"${baseName}"</em> into <strong>${components.length}</strong> separate parts, each with a <strong>different color</strong>!`, 'success');
  return true;
}

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

  if (restrictMesh) {
    restrictMesh.updateMatrixWorld(true);
    return brushRaycaster.intersectObject(restrictMesh, false);
  }

  const meshes = [];
  partsData.forEach((info, m) => {
    if (m && m.isMesh && m.visible && !m.isBrushOverlay && m !== brushHelper) {
      m.updateMatrixWorld(true);
      meshes.push(m);
    }
  });

  if (meshes.length === 0 && currentLoadedMesh) {
    currentLoadedMesh.traverse((child) => {
      if (child.isMesh && child.visible && !child.isBrushOverlay && child !== brushHelper) {
        child.updateMatrixWorld(true);
        meshes.push(child);
      }
    });
  }

  const rawHits = brushRaycaster.intersectObjects(meshes, false);
  return rawHits.filter(h => h.object && !h.object.isBrushOverlay && h.object !== brushHelper);
}

function handleBrushDown(e) {
  if (!isBrushActive || e.button !== 0) return;
  const hits = getModelIntersects(e, (brushScope === 'part' && brushLockedMesh) ? brushLockedMesh : null);
  if (hits.length > 0) {
    const hit = hits[0];
    const hitMesh = hit.object;

    if (brushScope === 'part') {
      if (!brushLockedMesh) {
        brushLockedMesh = hitMesh;
      } else if (hitMesh !== brushLockedMesh) {
        return; // Anti-bleed lock
      }
    }

    const hitNorm = (hit.face && hit.face.normal) ? hit.face.normal : new THREE.Vector3(0, 1, 0);

    if (brushMode === 'flood') {
      floodFillSemanticFeature(hitMesh, hit.faceIndex);
    } else if (brushMode === 'semantic') {
      paintFacesOnMesh(hitMesh, hit.point, hitNorm);
      detectAndSelectGlobalSemanticFeatures();
    } else {
      isBrushing = true;
      controls.enabled = false;
      paintFacesOnMesh(hitMesh, hit.point, hitNorm);
    }
  }
}

function handleBrushMove(e) {
  if (!isBrushActive) return;

  const surfaceHits = getModelIntersects(e, (brushScope === 'part' && brushLockedMesh) ? brushLockedMesh : null);
  if (surfaceHits.length > 0) {
    const hit = surfaceHits[0];
    const hitNorm = (hit.face && hit.face.normal) ? hit.face.normal : new THREE.Vector3(0, 1, 0);

    brushHelper.visible = true;
    brushHelper.position.copy(hit.point).addScaledVector(hitNorm, 0.003);
    brushHelper.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), hitNorm);

    if (isBrushing && brushMode === 'paint') {
      if (brushScope === 'part') {
        if (hit.object === brushLockedMesh) {
          paintFacesOnMesh(brushLockedMesh, hit.point, hitNorm);
        }
      } else {
        // Multi-Region / All Parts: paints any mesh hit without restrictions!
        paintFacesOnMesh(hit.object, hit.point, hitNorm);
      }
    }
  } else {
    brushHelper.visible = false;
  }
}

function handleBrushUp() {
  if (isBrushing) {
    isBrushing = false;
    controls.enabled = true;
  }
}

renderer.domElement.addEventListener('pointerdown', handleBrushDown);
renderer.domElement.addEventListener('mousedown', handleBrushDown);

renderer.domElement.addEventListener('pointermove', handleBrushMove);
renderer.domElement.addEventListener('mousemove', handleBrushMove);

window.addEventListener('pointerup', handleBrushUp);
window.addEventListener('mouseup', handleBrushUp);

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

// --- PIPELINE SELECTION & CONFIGURATION ENGINE ---
const PIPELINE_CONFIGS = {
  tripo_exact: {
    id: 'tripo_exact',
    label: 'Tripo Exact Partitioning',
    timeEst: '~10.8s Total',
    desc: 'High-speed exact mesh partitioning via P3-SAM face assignment. Preserves 100% of authentic PBR paint, vertex colors, and textures.',
    params: { mode: 'tripo_exact', steps: 50, resolution: 512, mc_algo: 'dmc', chunk_size: 16, preserve_materials: true },
    step2Label: 'Tripo3D Exact Partitioning:'
  },
  xpart_fast: {
    id: 'xpart_fast',
    label: 'X-Part Fast DiT',
    timeEst: '~35.0s Total',
    desc: 'Generative DiT flow-matching with 20 sampling steps, 384 octree grid, and CUDA GPU DiffDMC marching cubes.',
    params: { mode: 'xpart_fast', steps: 20, resolution: 384, mc_algo: 'dmc', chunk_size: 16, preserve_materials: false },
    step2Label: 'X-Part Fast DiT (20 steps, 384 res):'
  },
  xpart_lossless: {
    id: 'xpart_lossless',
    label: 'X-Part Lossless GPU',
    timeEst: '~71.9s Total',
    desc: 'Full 50-step DiT diffusion, 512 octree resolution, single-pass batch chunking (chunk=16), and GPU DiffDMC (bit-identical to original).',
    params: { mode: 'xpart_lossless', steps: 50, resolution: 512, mc_algo: 'dmc', chunk_size: 16, preserve_materials: false },
    step2Label: 'X-Part Lossless DiT (50 steps, 512 res):'
  },
  xpart_baseline: {
    id: 'xpart_baseline',
    label: 'Original X-Part Baseline',
    timeEst: '~185.5s Total',
    desc: 'Original unoptimized generative pipeline with 50 diffusion steps, 512 octree resolution, small chunking (chunk=4), and single-threaded CPU marching cubes.',
    params: { mode: 'xpart_baseline', steps: 50, resolution: 512, mc_algo: 'mc', chunk_size: 4, preserve_materials: false },
    step2Label: 'Original X-Part Baseline (CPU MC):'
  }
};

let currentSelectedPipelineMode = 'tripo_exact';

function setPipelineMode(mode) {
  if (!PIPELINE_CONFIGS[mode]) mode = 'tripo_exact';
  currentSelectedPipelineMode = mode;
  const cfg = PIPELINE_CONFIGS[mode];

  const topSelect = document.getElementById('topbar-pipeline-select');
  const sideSelect = document.getElementById('sidebar-pipeline-select');
  const latencyBadge = document.getElementById('pipeline-latency-badge');
  const descText = document.getElementById('pipeline-desc-text');
  const step2Label = document.getElementById('step2-title-label');

  if (topSelect && topSelect.value !== mode) topSelect.value = mode;
  if (sideSelect && sideSelect.value !== mode) sideSelect.value = mode;
  if (latencyBadge) latencyBadge.innerText = cfg.timeEst;
  if (descText) descText.innerText = cfg.desc;
  if (step2Label) step2Label.innerText = cfg.step2Label;

  showToast(`⚙️ Active Pipeline set to <strong>${cfg.label}</strong> (${cfg.timeEst})`, 'info');
}

const topbarPipelineSelect = document.getElementById('topbar-pipeline-select');
if (topbarPipelineSelect) {
  topbarPipelineSelect.onchange = (e) => setPipelineMode(e.target.value);
}

const sidebarPipelineSelect = document.getElementById('sidebar-pipeline-select');
if (sidebarPipelineSelect) {
  sidebarPipelineSelect.onchange = (e) => setPipelineMode(e.target.value);
}

const topbarRunPipelineBtn = document.getElementById('topbar-run-pipeline-btn');
if (topbarRunPipelineBtn) {
  topbarRunPipelineBtn.onclick = () => runAutoSegmentationPipeline();
}

const sidebarRunPipelineBtn = document.getElementById('sidebar-run-pipeline-btn');
if (sidebarRunPipelineBtn) {
  sidebarRunPipelineBtn.onclick = () => runAutoSegmentationPipeline();
}

async function runAutoSegmentationPipeline(meshPath = null, modelTitle = null, overrideMode = null) {
  const targetPath = meshPath || currentModelPath;
  if (!targetPath) {
    showToast('⚠️ Please upload or select a 3D model first!', 'warning');
    return;
  }

  const selectedMode = overrideMode || currentSelectedPipelineMode || 'tripo_exact';
  const pipelineCfg = PIPELINE_CONFIGS[selectedMode] || PIPELINE_CONFIGS['tripo_exact'];

  const banner = document.getElementById('progress-banner');
  const stageText = document.getElementById('progress-stage-text');
  const percentText = document.getElementById('progress-percent-text');
  const barFill = document.getElementById('progress-bar-fill');
  if (banner) banner.style.display = 'flex';

  const step2Label = document.getElementById('step2-title-label');
  if (step2Label) step2Label.innerText = pipelineCfg.step2Label;

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
    const postBody = {
      mesh_path: targetPath,
      ...pipelineCfg.params,
      explosion_scale: 0.35
    };

    const res = await fetch('/api/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(postBody)
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
        } else if (sData.status === 'cancelled') {
          clearInterval(pollInterval);
          if (banner) banner.style.display = 'none';
          stopLiveStopwatch();
          showToast('⏸️ Preempted: Previous task paused to free GPU for new pipeline run', 'info');
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

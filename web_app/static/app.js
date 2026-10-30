// --- THREE.JS SCENE SETUP ---
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0b0d13');

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 100);
camera.position.set(2.8, 2.0, 3.4);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.outputEncoding = THREE.sRGBEncoding;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Studio Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
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

// --- REVERSE-ENGINEERED TRIPO3D EXPLOSION ENGINE (CRvlez9U.js) ---
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

let partsData = new Map();
let globalCenter = new THREE.Vector3();
let maxDistance = 1.0;
let isExploded = false;
let currentLoadedMesh = null;
let currentModelPath = null;

// Exact Tripo3D easing polynomials
const tripoOvershootEasing = (e) => 1 + 2.2 * Math.pow(e - 1, 3) + 1.2 * Math.pow(e - 1, 2);
const tripoCollapseEasing = (e) => e < 0.5 ? 4 * e * e * e : 1 - Math.pow(-2 * e + 2, 3) / 2;

function initExplosionData(model) {
  const bBoxUnion = new THREE.Box3();
  const meshes = [];

  model.traverse((child) => {
    if (child.isMesh) {
      meshes.push(child);
      child.geometry.computeBoundingBox();
      const partBox = child.geometry.boundingBox.clone();
      partBox.applyMatrix4(child.matrixWorld);
      bBoxUnion.union(partBox);
    }
  });

  bBoxUnion.getCenter(globalCenter);
  const bBoxSize = new THREE.Vector3();
  bBoxUnion.getSize(bBoxSize);
  gridHelper.position.y = bBoxUnion.min.y - 0.02;

  // Frame Camera
  const maxDim = Math.max(bBoxSize.x, bBoxSize.y, bBoxSize.z);
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
      originalLocalPosition: mesh.position.clone(),
      originalRotation: mesh.quaternion.clone(),
      explodeDirection: lVec.clone(),
      distanceWeight: dist / maxDistance,
      geometryCenterOffset: geomCenter.clone(),
      accumulatedRotation: 0,
      rotationAxis: rotAxis,
      rotationSpeedFactor: rotSpeedFactor,
      snapshotRotation: 0
    };
    partsData.set(mesh, partInfo);

    // Build sidebar item
    const item = document.createElement('div');
    item.className = 'part-item';
    const colorHex = '#' + (mesh.material?.color?.getHexString() || '818cf8');
    const partName = mesh.name || `tripo_part_${idx}`;
    item.innerHTML = `
      <div class="part-badge">
        <span class="part-color-dot" style="background: ${colorHex}"></span>
        <span>${partName}</span>
      </div>
      <span class="eye-icon">👁</span>
    `;
    item.onclick = (e) => {
      if (e.target.classList.contains('eye-icon')) {
        mesh.visible = !mesh.visible;
        item.style.opacity = mesh.visible ? '1' : '0.3';
      } else {
        // Focus on part
        controls.target.copy(worldPos);
        controls.update();
      }
    };
    partsContainer.appendChild(item);
  });
}

// Apply Tripo3D explosion transforms
function applyExplosion(tValue, deltaSec = 0.016) {
  const allowTumbling = document.getElementById('tumbling-toggle').checked;
  const rotQuat = new THREE.Quaternion();
  const isCollapsing = (currentState === States.Collapsing);
  const collapseProgress = isCollapsing ? Math.min((performance.now() - animStartTime) / ANIM_DURATION, 1) : 0;
  const collapseEase = tripoCollapseEasing(collapseProgress);

  partsData.forEach((info, mesh) => {
    if (isCollapsing) {
      info.accumulatedRotation = info.snapshotRotation * (1 - collapseEase);
      rotQuat.setFromAxisAngle(info.rotationAxis, info.accumulatedRotation);
      rotQuat.premultiply(info.originalRotation);

      const rOrig = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(info.originalRotation);
      const rCur = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(rotQuat);
      const pivotComp = rOrig.sub(rCur);

      const rDist = maxDistance * (0.35 + 0.65 * Math.min(1, info.distanceWeight));
      const disp = info.explodeDirection.clone().multiplyScalar((1 - collapseEase) * rDist);

      mesh.position.copy(info.originalLocalPosition).add(pivotComp).add(disp);
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

      mesh.position.copy(info.originalLocalPosition).add(pivotComp).add(disp);
      mesh.quaternion.copy(rotQuat);
    } else {
      const rDist = maxDistance * (0.35 + 0.65 * Math.min(1, info.distanceWeight));
      const disp = info.explodeDirection.clone().multiplyScalar(easedProgress * rDist);
      mesh.position.copy(info.originalLocalPosition).add(disp);
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
        mesh.position.copy(info.originalLocalPosition);
        mesh.quaternion.copy(info.originalRotation);
        info.accumulatedRotation = 0;
      });
    }
  }
}

// Load Model Helper
const loader = new THREE.GLTFLoader();
function loadModelFromUrl(url, title = null) {
  currentModelPath = url;
  if (title) {
    document.getElementById('model-title-text').innerText = title;
  }
  
  // Show minimal loading state
  loader.load(url, (gltf) => {
    if (currentLoadedMesh) {
      scene.remove(currentLoadedMesh);
    }
    currentLoadedMesh = gltf.scene;
    scene.add(currentLoadedMesh);
    initExplosionData(currentLoadedMesh);

    // Reset controls
    isExploded = false;
    currentState = States.Idle;
    document.getElementById('explode-toggle').classList.remove('active');
    document.getElementById('explode-text').innerText = 'Explode';
    document.getElementById('explode-slider').value = 0;
  }, undefined, (err) => {
    console.error('Error loading GLB:', err);
  });
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
  applyExplosion(val, 0);
};

// --- PRESETS & PIPELINE API CLIENT ---

async function fetchPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    const list = document.getElementById('presets-list');
    list.innerHTML = '';
    
    data.presets.forEach((p, idx) => {
      const chip = document.createElement('div');
      chip.className = `preset-chip ${idx === 0 ? 'active' : ''}`;
      chip.innerHTML = `
        <span style="font-weight: 500;">${p.name}</span>
        <span style="font-size: 10px; opacity: 0.6;">${p.type === 'segmented_assembly' ? 'READY' : 'RAW'}</span>
      `;
      chip.onclick = () => {
        document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        loadModelFromUrl(p.path, p.name);
      };
      list.appendChild(chip);
    });

    // Auto-load first preset if available
    if (data.presets.length > 0) {
      loadModelFromUrl(data.presets[0].path, data.presets[0].name);
    }
  } catch (err) {
    console.warn('Could not fetch presets:', err);
  }
}

// File Upload
const fileInput = document.getElementById('file-input');
document.getElementById('upload-btn').onclick = () => fileInput.click();

fileInput.onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.status === 'success') {
      loadModelFromUrl(data.path, file.name);
    } else {
      alert(data.error || 'Upload failed');
    }
  } catch (err) {
    alert('Upload error: ' + err);
  }
};

// Run Segmentation Pipeline
document.getElementById('run-pipeline-btn').onclick = async () => {
  if (!currentModelPath) {
    alert('Please load or upload a 3D model first!');
    return;
  }

  const mcAlgo = document.getElementById('mc-algo-select').value;
  const steps = parseInt(document.getElementById('steps-select').value);
  const resolution = parseInt(document.getElementById('res-select').value);

  const banner = document.getElementById('progress-banner');
  const stageText = document.getElementById('progress-stage-text');
  const percentText = document.getElementById('progress-percent-text');
  const barFill = document.getElementById('progress-bar-fill');
  banner.style.display = 'flex';

  try {
    const res = await fetch('/api/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mesh_path: currentModelPath,
        mc_algo: mcAlgo,
        steps: steps,
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

        if (sData.status === 'completed') {
          clearInterval(pollInterval);
          setTimeout(() => { banner.style.display = 'none'; }, 800);
          // Load generated multi-node assembly
          loadModelFromUrl(sData.result.assembly_glb, `${document.getElementById('model-title-text').innerText} (Segmented)`);
          
          // Configure download buttons
          document.getElementById('download-assembly-btn').onclick = () => window.open(sData.result.assembly_glb, '_blank');
          document.getElementById('download-exploded-btn').onclick = () => window.open(sData.result.exploded_glb, '_blank');
        } else if (sData.status === 'error') {
          clearInterval(pollInterval);
          banner.style.display = 'none';
          alert('Pipeline error: ' + sData.error);
        }
      } catch (err) {
        clearInterval(pollInterval);
        banner.style.display = 'none';
        console.error('Polling error:', err);
      }
    }, 1000);

  } catch (err) {
    banner.style.display = 'none';
    alert('Pipeline launch error: ' + err.message);
  }
};

// Download Current Model
document.getElementById('download-assembly-btn').onclick = () => {
  if (currentModelPath) window.open(currentModelPath, '_blank');
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

// Initialize Presets on load
fetchPresets();

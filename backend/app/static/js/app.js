// Global Variables
let webcamStream = null;
let genderChart = null;
let ageChart = null;
let currentDetections = []; // Store current image upload detections for resize redraws
let currentWebcamDetections = [];

// DOM Elements
const doc = document;
const navTabs = doc.querySelectorAll('.nav-custom-link');
const fileInput = doc.getElementById('file-input');
const dropzone = doc.getElementById('image-dropzone');
const uploadStatus = doc.getElementById('upload-status');
const uploadResultContainer = doc.getElementById('upload-result-container');
const uploadResultsTbody = doc.getElementById('upload-results-tbody');
const noUploadMsg = doc.getElementById('no-upload-msg');
const previewImg = doc.getElementById('image-preview');
const previewWrapper = doc.getElementById('image-preview-wrapper');

// Webcam Elements
const webcamVideo = doc.getElementById('webcam-video');
const btnStartWebcam = doc.getElementById('btn-start-webcam');
const btnStopWebcam = doc.getElementById('btn-stop-webcam');
const btnCaptureWebcam = doc.getElementById('btn-capture-webcam');
const webcamStatus = doc.getElementById('webcam-status');
const webcamResultContainer = doc.getElementById('webcam-result-container');
const webcamSnapshotPreview = doc.getElementById('webcam-snapshot-preview');
const webcamResultsTbody = doc.getElementById('webcam-results-tbody');
const noWebcamMsg = doc.getElementById('no-webcam-msg');

// Initialize Application
doc.addEventListener("DOMContentLoaded", () => {
    // 1. Tab Navigation Routing and Lifecycle
    navTabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', (e) => {
            const targetId = e.target.getAttribute('href');
            
            // Stop webcam if navigating away from the webcam view
            if (targetId !== '#webcam') {
                stopWebcam();
            }
            
            if (targetId === '#dashboard') {
                fetchDashboardStats();
            } else if (targetId === '#history') {
                fetchHistory();
            }
        });
    });

    // 2. Drag & Drop Upload Handlers
    dropzone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageUpload(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });

    // 3. Webcam Handlers
    btnStartWebcam.addEventListener('click', startWebcam);
    btnStopWebcam.addEventListener('click', stopWebcam);
    btnCaptureWebcam.addEventListener('click', captureWebcamFrame);

    // 4. Redraw Bounding Boxes on Resize
    window.addEventListener('resize', () => {
        if (currentDetections.length > 0) {
            renderBoundingBoxes(previewWrapper, previewImg, currentDetections);
        }
        if (currentWebcamDetections.length > 0) {
            const webcamWrapper = webcamSnapshotPreview.parentElement;
            renderBoundingBoxes(webcamWrapper, webcamSnapshotPreview, currentWebcamDetections);
        }
    });

    // Initial load
    fetchDashboardStats();
});

// ==========================================
// 1. DASHBOARD SERVICE
// ==========================================
async function fetchDashboardStats() {
    try {
        const response = await fetch('/api/dashboard');
        const stats = await response.json();
        
        // Update total detections indicator
        doc.getElementById('stat-total-detections').textContent = stats.total_detections;
        
        // Update distribution summary indicators
        const mCount = stats.gender_distribution.Male || 0;
        const fCount = stats.gender_distribution.Female || 0;
        doc.getElementById('stat-gender-distribution').textContent = `M: ${mCount} | F: ${fCount}`;
        
        // Find top age group
        let maxAgeGroup = '--';
        let maxAgeCount = -1;
        for (const [group, count] of Object.entries(stats.age_distribution)) {
            if (count > maxAgeCount && count > 0) {
                maxAgeCount = count;
                maxAgeGroup = group;
            }
        }
        doc.getElementById('stat-top-age').textContent = maxAgeGroup === '--' ? '--' : `${maxAgeGroup} (${maxAgeCount})`;
        
        // Render/Update charts
        renderGenderChart(stats.gender_distribution);
        renderAgeChart(stats.age_distribution);
    } catch (err) {
        console.error("Error fetching dashboard statistics: ", err);
    }
}

function renderGenderChart(genderData) {
    const ctx = doc.getElementById('genderChart').getContext('2d');
    const dataValues = [genderData.Male || 0, genderData.Female || 0];
    
    if (genderChart) {
        genderChart.data.datasets[0].data = dataValues;
        genderChart.update();
        return;
    }

    genderChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Male', 'Female'],
            datasets: [{
                data: dataValues,
                backgroundColor: ['#06b6d4', '#f43f5e'],
                borderColor: 'rgba(255,255,255,0.05)',
                borderWidth: 2,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', weight: 'bold' } }
                }
            },
            cutout: '70%'
        }
    });
}

function renderAgeChart(ageData) {
    const ctx = doc.getElementById('ageChart').getContext('2d');
    const labels = Object.keys(ageData);
    const dataValues = Object.values(ageData);

    if (ageChart) {
        ageChart.data.datasets[0].data = dataValues;
        ageChart.update();
        return;
    }

    ageChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Detections',
                data: dataValues,
                backgroundColor: 'rgba(99, 102, 241, 0.65)',
                borderColor: '#6366f1',
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { precision: 0, color: '#94a3b8' }
                }
            }
        }
    });
}

// ==========================================
// 2. FILE UPLOAD SERVICE
// ==========================================
async function handleImageUpload(file) {
    uploadStatus.classList.remove('d-none');
    uploadResultContainer.classList.add('d-none');
    noUploadMsg.classList.add('d-none');
    
    // Clear previous bounding boxes
    clearBoundingBoxes(previewWrapper);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/predict-image', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error("HTTP error on uploading image file.");
        }
        
        const data = await response.json();
        currentDetections = data.detections;
        
        // Show prediction container
        uploadResultContainer.classList.remove('d-none');
        
        // Set Image Source
        previewImg.src = `/uploads/${data.image_name}`;
        
        // Wait for image to load to render bounding boxes accurately
        previewImg.onload = () => {
            renderBoundingBoxes(previewWrapper, previewImg, currentDetections);
        };
        
        // Populate results table
        populateResultsTable(uploadResultsTbody, currentDetections);
        
    } catch (err) {
        console.error("Error analyzing uploaded image: ", err);
        alert("Face analysis failed. Please verify the image contains clear visible human faces.");
        noUploadMsg.classList.remove('d-none');
    } finally {
        uploadStatus.classList.add('d-none');
    }
}

// ==========================================
// 3. LIVE WEBCAM SERVICE
// ==========================================
async function startWebcam() {
    noWebcamMsg.classList.add('d-none');
    webcamResultContainer.classList.add('d-none');
    clearBoundingBoxes(webcamSnapshotPreview.parentElement);
    currentWebcamDetections = [];

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" }
        });
        webcamVideo.srcObject = webcamStream;
        
        btnStartWebcam.disabled = true;
        btnStopWebcam.disabled = false;
        btnCaptureWebcam.disabled = false;
    } catch (err) {
        console.error("Webcam activation error: ", err);
        alert("Could not access webcam. Please verify camera permissions in your browser.");
        noWebcamMsg.classList.remove('d-none');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    webcamVideo.srcObject = null;
    btnStartWebcam.disabled = false;
    btnStopWebcam.disabled = true;
    btnCaptureWebcam.disabled = true;
}

async function captureWebcamFrame() {
    if (!webcamStream) return;
    
    webcamStatus.classList.remove('d-none');
    webcamResultContainer.classList.add('d-none');
    
    const canvas = doc.createElement('canvas');
    canvas.width = webcamVideo.videoWidth;
    canvas.height = webcamVideo.videoHeight;
    
    const ctx = canvas.getContext('2d');
    // Mirror the frame horizontally so it matches the mirrored preview
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
    
    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    
    const base64Image = canvas.toDataURL('image/jpeg');
    const webcamWrapper = webcamSnapshotPreview.parentElement;
    clearBoundingBoxes(webcamWrapper);
    
    try {
        const response = await fetch('/api/predict-webcam', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });
        
        if (!response.ok) {
            throw new Error("HTTP error predicting webcam frame.");
        }
        
        const data = await response.json();
        currentWebcamDetections = data.detections;
        
        webcamResultContainer.classList.remove('d-none');
        webcamSnapshotPreview.src = `/uploads/${data.image_name}`;
        
        webcamSnapshotPreview.onload = () => {
            renderBoundingBoxes(webcamWrapper, webcamSnapshotPreview, currentWebcamDetections);
        };
        
        populateResultsTable(webcamResultsTbody, currentWebcamDetections);
        
    } catch (err) {
        console.error("Webcam prediction error: ", err);
        alert("Analysis failed. Please align your face in front of the camera.");
        noWebcamMsg.classList.remove('d-none');
    } finally {
        webcamStatus.classList.add('d-none');
    }
}

// ==========================================
// 4. BOUNDING BOX & INTERACTIVE RENDERING
// ==========================================
function renderBoundingBoxes(container, imageEl, detections) {
    clearBoundingBoxes(container);
    
    const displayW = imageEl.clientWidth;
    const displayH = imageEl.clientHeight;
    
    const naturalW = imageEl.naturalWidth;
    const naturalH = imageEl.naturalHeight;
    
    if (!naturalW || !naturalH) return; // Guard against unloaded state
    
    const scaleX = displayW / naturalW;
    const scaleY = displayH / naturalH;
    
    detections.forEach((det, idx) => {
        const [x, y, w, h] = det.box;
        
        // Calculate relative coordinates in display pixels
        const left = x * scaleX;
        const top = y * scaleY;
        const width = w * scaleX;
        const height = h * scaleY;
        
        // Create Bounding Box element
        const boxDiv = doc.createElement('div');
        boxDiv.className = 'bounding-box';
        boxDiv.style.left = `${left}px`;
        boxDiv.style.top = `${top}px`;
        boxDiv.style.width = `${width}px`;
        boxDiv.style.height = `${height}px`;
        
        // Create Label
        const labelDiv = doc.createElement('div');
        labelDiv.className = 'bounding-box-label';
        labelDiv.innerHTML = `Face #${idx + 1}: ${det.gender} (${Math.round(det.gender_confidence * 100)}%), Age ${det.age_group}`;
        boxDiv.appendChild(labelDiv);
        
        container.appendChild(boxDiv);
    });
}

function clearBoundingBoxes(container) {
    const boxes = container.querySelectorAll('.bounding-box');
    boxes.forEach(box => box.remove());
}

function populateResultsTable(tbody, detections) {
    tbody.innerHTML = '';
    
    if (detections.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-warning">No faces detected.</td></tr>';
        return;
    }
    
    detections.forEach((det, idx) => {
        const genderBadgeClass = det.gender.toLowerCase() === 'male' ? 'male' : 'female';
        const tr = doc.createElement('tr');
        tr.innerHTML = `
            <td><strong class="text-white-50">#${idx + 1}</strong></td>
            <td><span class="badge-gender ${genderBadgeClass}"><i class="fa-solid ${det.gender.toLowerCase() === 'male' ? 'fa-mars' : 'fa-venus'} me-1"></i>${det.gender}</span></td>
            <td><span class="badge-age">${det.age_group}</span></td>
            <td><div class="fw-semibold">${Math.round(det.confidence * 100)}%</div></td>
        `;
        tbody.appendChild(tr);
    });
}

// ==========================================
// 5. HISTORY MANAGEMENT SERVICE
// ==========================================
async function fetchHistory() {
    const historyTbody = doc.getElementById('history-tbody');
    const noHistoryMsg = doc.getElementById('no-history-msg');
    
    historyTbody.innerHTML = '<tr><td colspan="7" class="text-center py-4"><div class="spinner-border spinner-border-sm me-2" role="status"></div>Loading logs...</td></tr>';
    noHistoryMsg.classList.add('d-none');
    
    try {
        const response = await fetch('/api/history?limit=50');
        const historyData = await response.json();
        
        historyTbody.innerHTML = '';
        
        if (historyData.length === 0) {
            noHistoryMsg.classList.remove('d-none');
            return;
        }
        
        historyData.forEach(item => {
            const genderBadgeClass = item.predicted_gender.toLowerCase() === 'male' ? 'male' : 'female';
            const tr = doc.createElement('tr');
            tr.id = `history-item-${item.id}`;
            
            // Format datetime
            const dt = new Date(item.prediction_time);
            const timeFormatted = dt.toLocaleString();
            
            tr.innerHTML = `
                <td>
                    <a href="/uploads/${item.image_name}" target="_blank">
                        <img class="history-img-thumb" src="/uploads/${item.image_name}" alt="Thumbnail">
                    </a>
                </td>
                <td><small class="text-secondary">#${item.id}</small></td>
                <td><span class="badge-gender ${genderBadgeClass}">${item.predicted_gender}</span></td>
                <td><span class="badge-age">${item.predicted_age_group}</span></td>
                <td><div class="fw-semibold">${Math.round(item.confidence * 100)}%</div></td>
                <td><small class="text-secondary">${timeFormatted}</small></td>
                <td>
                    <button class="btn btn-danger-custom btn-sm" onclick="deleteHistoryItem(${item.id})">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </td>
            `;
            historyTbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Error loading predictions history: ", err);
        historyTbody.innerHTML = '<tr><td colspan="7" class="text-center text-danger">Failed to retrieve history logs.</td></tr>';
    }
}

async function deleteHistoryItem(id) {
    if (!confirm("Are you sure you want to delete this prediction log from history?")) return;
    
    try {
        const response = await fetch(`/api/history/${id}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            // Remove DOM element
            const tr = doc.getElementById(`history-item-${id}`);
            if (tr) tr.remove();
            
            // Refresh counts and graphics
            fetchDashboardStats();
            
            // Show no logs placeholder if table empty
            const historyTbody = doc.getElementById('history-tbody');
            if (historyTbody.children.length === 0) {
                doc.getElementById('no-history-msg').classList.remove('d-none');
            }
        } else {
            alert("Delete operation failed.");
        }
    } catch (err) {
        console.error("Error deleting history record: ", err);
        alert("Network failure trying to delete the prediction record.");
    }
}

/**
 * F&B Anomaly Detection System - Version 2.0 JavaScript
 * Enhanced dashboard functionality for dual-module architecture
 */

// Global variables
let systemStatus = {};
let currentModule = null;
let chartInstances = {};

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

/**
 * Initialize the dashboard
 */
function initializeDashboard() {
    console.log('Initializing F&B Anomaly Detection Dashboard v2.0...');
    
    // Check system status
    checkSystemStatus();
    
    // Initialize charts if elements exist
    initializeCharts();
    
    // Setup event listeners
    setupEventListeners();
    
    // Auto-refresh status every 30 seconds
    setInterval(checkSystemStatus, 30000);
    
    console.log('Dashboard initialized successfully');
}

/**
 * Check system status for both modules
 */
async function checkSystemStatus() {
    try {
        const response = await axios.get('/api/status');
        systemStatus = response.data;
        
        updateSystemStatusDisplay(systemStatus);
        
    } catch (error) {
        console.error('Error checking system status:', error);
        displayErrorStatus();
    }
}

/**
 * Update system status display
 */
function updateSystemStatusDisplay(status) {
    // Update main status indicators
    const systemStatusEl = document.getElementById('system-status');
    if (systemStatusEl) {
        systemStatusEl.textContent = status.status === 'operational' ? 'Operational' : 'Offline';
        systemStatusEl.className = status.status === 'operational' ? 'status-operational' : 'status-error';
    }
    
    // Update Module 1 status
    const module1StatusEl = document.getElementById('module1-status');
    if (module1StatusEl) {
        const module1Status = status.module1;
        if (module1Status) {
            module1StatusEl.textContent = module1Status.data_loaded ? 'Data Loaded' : 'Ready';
        }
    }
    
    // Update Module 2 status
    const module2StatusEl = document.getElementById('module2-status');
    const modelsCountEl = document.getElementById('models-count');
    
    if (module2StatusEl && status.module2) {
        const module2Status = status.module2;
        if (module2Status.loaded) {
            module2StatusEl.textContent = `${module2Status.models_count} Models Loaded`;
            if (modelsCountEl) {
                modelsCountEl.textContent = module2Status.models_count;
            }
        } else {
            module2StatusEl.textContent = 'Not Available';
            if (modelsCountEl) {
                modelsCountEl.textContent = '0';
            }
        }
    }
    
    // Update performance status
    const performanceStatusEl = document.getElementById('performance-status');
    if (performanceStatusEl) {
        performanceStatusEl.textContent = 'Optimal';
    }
}

/**
 * Display error status when API call fails
 */
function displayErrorStatus() {
    const systemStatusEl = document.getElementById('system-status');
    if (systemStatusEl) {
        systemStatusEl.textContent = 'Connection Error';
        systemStatusEl.className = 'status-error';
    }
}

/**
 * Initialize charts
 */
function initializeCharts() {
    // Hero chart
    const heroChartEl = document.getElementById('heroChart');
    if (heroChartEl) {
        initializeHeroChart();
    }
    
    // Performance charts
    const performanceChartEl = document.getElementById('performanceChart');
    if (performanceChartEl) {
        initializePerformanceChart();
    }
    
    // Quality trend chart
    const qualityChartEl = document.getElementById('qualityChart');
    if (qualityChartEl) {
        initializeQualityChart();
    }
}

/**
 * Initialize hero chart
 */
function initializeHeroChart() {
    const ctx = document.getElementById('heroChart').getContext('2d');
    
    chartInstances.heroChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 30}, (_, i) => i),
            datasets: [{
                label: 'Quality Score',
                data: generateRandomData(30, 85, 95),
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4
            }, {
                label: 'Process Efficiency',
                data: generateRandomData(30, 80, 95),
                borderColor: '#764ba2',
                backgroundColor: 'rgba(118, 75, 162, 0.1)',
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    display: false,
                    min: 70,
                    max: 100
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
    
    // Animate chart data
    setInterval(() => {
        chartInstances.heroChart.data.datasets.forEach(dataset => {
            dataset.data.shift();
            dataset.data.push(generateRandomValue(85, 95));
        });
        chartInstances.heroChart.update('none');
    }, 2000);
}

/**
 * Initialize performance chart
 */
function initializePerformanceChart() {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    chartInstances.performanceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Excellent', 'Good', 'Fair', 'Poor'],
            datasets: [{
                data: [65, 25, 8, 2],
                backgroundColor: [
                    '#10b981',
                    '#3b82f6',
                    '#f59e0b',
                    '#ef4444'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

/**
 * Initialize quality trend chart
 */
function initializeQualityChart() {
    const ctx = document.getElementById('qualityChart').getContext('2d');
    
    chartInstances.qualityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Average Quality Score',
                data: [92, 88, 94, 91, 89, 93, 90],
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderColor: '#667eea',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 80,
                    max: 100
                }
            }
        }
    });
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Mobile menu toggle
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navToggle.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }
    
    // Close mobile menu when clicking on a link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            if (navToggle && navMenu) {
                navToggle.classList.remove('active');
                navMenu.classList.remove('active');
            }
        });
    });
    
    // Module selection handlers
    document.querySelectorAll('[data-module]').forEach(element => {
        element.addEventListener('click', (e) => {
            const module = e.currentTarget.getAttribute('data-module');
            selectModule(module);
        });
    });
    
    // File upload handlers
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', handleFileSelection);
    });
    
    // Drag and drop handlers
    setupDragAndDrop();
}

/**
 * Setup drag and drop functionality
 */
function setupDragAndDrop() {
    const dropZones = document.querySelectorAll('.upload-area');
    
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', handleDragOver);
        zone.addEventListener('drop', handleDrop);
        zone.addEventListener('dragleave', handleDragLeave);
        zone.addEventListener('dragenter', handleDragEnter);
    });
}

/**
 * Handle drag over event
 */
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-over');
}

/**
 * Handle drag enter event
 */
function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
}

/**
 * Handle drag leave event
 */
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
}

/**
 * Handle drop event
 */
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (isValidFile(file)) {
            handleFileUpload(file);
        } else {
            showNotification('Invalid file type. Please upload Excel or CSV files only.', 'error');
        }
    }
}

/**
 * Handle file selection
 */
function handleFileSelection(e) {
    const file = e.target.files[0];
    if (file) {
        if (isValidFile(file)) {
            handleFileUpload(file);
        } else {
            showNotification('Invalid file type. Please upload Excel or CSV files only.', 'error');
        }
    }
}

/**
 * Check if file is valid
 */
function isValidFile(file) {
    const validTypes = [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'text/csv'
    ];
    
    const validExtensions = ['.xlsx', '.xls', '.csv'];
    
    return validTypes.includes(file.type) || 
           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
}

/**
 * Handle file upload
 */
async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading(`Uploading ${file.name}...`);
    
    try {
        // Determine which module to use based on current page
        const currentPath = window.location.pathname;
        let endpoint = '/api/module1/upload';
        
        if (currentPath.includes('module2')) {
            endpoint = '/api/module2/predict/batch';
        }
        
        const response = await axios.post(endpoint, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        
        hideLoading();
        
        if (response.data.success) {
            showNotification('File uploaded successfully!', 'success');
            handleUploadSuccess(response.data);
        } else {
            showNotification('Upload failed: ' + response.data.error, 'error');
        }
        
    } catch (error) {
        hideLoading();
        showNotification('Upload failed: ' + (error.response?.data?.error || error.message), 'error');
    }
}

/**
 * Handle successful upload
 */
function handleUploadSuccess(data) {
    // Update UI based on the response
    console.log('Upload successful:', data);
    
    // Trigger appropriate actions based on current module
    const currentPath = window.location.pathname;
    
    if (currentPath.includes('module1')) {
        // Module 1: Show data summary and enable training
        displayDataSummary(data);
    } else if (currentPath.includes('module2')) {
        // Module 2: Show prediction results
        displayPredictionResults(data);
    }
}

/**
 * Display data summary for Module 1
 */
function displayDataSummary(data) {
    const summaryEl = document.getElementById('dataSummary');
    if (summaryEl && data.data_info) {
        const info = data.data_info;
        summaryEl.innerHTML = `
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-number">${info.rows_processed || 0}</div>
                    <div class="summary-label">Rows</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number">${info.features_extracted || 0}</div>
                    <div class="summary-label">Features</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number">${info.outliers_detected || 0}</div>
                    <div class="summary-label">Outliers</div>
                </div>
            </div>
        `;
    }
}

/**
 * Display prediction results for Module 2
 */
function displayPredictionResults(data) {
    const resultsEl = document.getElementById('predictionResults');
    if (resultsEl && data.result) {
        // Display results based on the data structure
        console.log('Displaying prediction results:', data.result);
    }
}

/**
 * Select module
 */
function selectModule(moduleNumber) {
    currentModule = moduleNumber;
    
    // Redirect to appropriate module page
    if (moduleNumber === 1) {
        window.location.href = '/module1';
    } else if (moduleNumber === 2) {
        window.location.href = '/module2';
    }
}

/**
 * Show loading overlay
 */
function showLoading(message = 'Processing...') {
    const loadingEl = document.getElementById('loading');
    const loadingTextEl = document.getElementById('loadingText');
    
    if (loadingEl) {
        if (loadingTextEl) {
            loadingTextEl.textContent = message;
        }
        loadingEl.style.display = 'flex';
    }
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const loadingEl = document.getElementById('loading');
    if (loadingEl) {
        loadingEl.style.display = 'none';
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <span class="notification-icon">
                ${getNotificationIcon(type)}
            </span>
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                &times;
            </button>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
    
    // Add styles if not already present
    addNotificationStyles();
}

/**
 * Get notification icon based on type
 */
function getNotificationIcon(type) {
    const icons = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ'
    };
    return icons[type] || icons.info;
}

/**
 * Add notification styles
 */
function addNotificationStyles() {
    if (document.getElementById('notification-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'notification-styles';
    styles.textContent = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            min-width: 300px;
            max-width: 500px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            animation: slideInRight 0.3s ease-out;
        }
        
        .notification-success {
            border-left: 4px solid #10b981;
        }
        
        .notification-error {
            border-left: 4px solid #ef4444;
        }
        
        .notification-warning {
            border-left: 4px solid #f59e0b;
        }
        
        .notification-info {
            border-left: 4px solid #3b82f6;
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            padding: 16px;
            gap: 12px;
        }
        
        .notification-icon {
            font-size: 18px;
            font-weight: bold;
        }
        
        .notification-success .notification-icon {
            color: #10b981;
        }
        
        .notification-error .notification-icon {
            color: #ef4444;
        }
        
        .notification-warning .notification-icon {
            color: #f59e0b;
        }
        
        .notification-info .notification-icon {
            color: #3b82f6;
        }
        
        .notification-message {
            flex: 1;
            color: #374151;
        }
        
        .notification-close {
            background: none;
            border: none;
            font-size: 18px;
            color: #9ca3af;
            cursor: pointer;
            padding: 0;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .notification-close:hover {
            color: #6b7280;
        }
        
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    `;
    
    document.head.appendChild(styles);
}

/**
 * Show modal
 */
function showModal(title, content, actions = null) {
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    
    if (modal && modalTitle && modalBody) {
        modalTitle.textContent = title;
        modalBody.innerHTML = content;
        modal.style.display = 'block';
    }
}

/**
 * Close modal
 */
function closeModal() {
    const modal = document.getElementById('modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

/**
 * Generate random data for charts
 */
function generateRandomData(length, min, max) {
    return Array.from({length}, () => generateRandomValue(min, max));
}

/**
 * Generate random value within range
 */
function generateRandomValue(min, max) {
    return min + Math.random() * (max - min);
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Export data as JSON
 */
function exportAsJSON(data, filename) {
    const dataStr = JSON.stringify(data, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = filename || `export_${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
}

/**
 * Export data as CSV
 */
function exportAsCSV(data, filename) {
    let csvContent = '';
    
    if (Array.isArray(data) && data.length > 0) {
        // Get headers
        const headers = Object.keys(data[0]);
        csvContent += headers.join(',') + '\n';
        
        // Add data rows
        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header];
                return typeof value === 'string' ? `"${value}"` : value;
            });
            csvContent += values.join(',') + '\n';
        });
    }
    
    const dataUri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent);
    const exportFileDefaultName = filename || `export_${new Date().toISOString().split('T')[0]}.csv`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
}

// Global functions for backward compatibility
window.showModal = showModal;
window.closeModal = closeModal;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.showNotification = showNotification;
window.exportAsJSON = exportAsJSON;
window.exportAsCSV = exportAsCSV;

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('modal');
    if (event.target === modal) {
        closeModal();
    }
}

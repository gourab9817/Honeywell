// F&B Anomaly Detection System - Dashboard JavaScript

// Global variables
let charts = {};
let updateIntervals = [];
let alertQueue = [];

// Utility functions
function formatNumber(num, decimals = 1) {
    return Number(num).toFixed(decimals);
}

function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function getStatusColor(status) {
    const colors = {
        'normal': '#28a745',
        'warning': '#ffc107',
        'critical': '#dc3545',
        'low': '#17a2b8',
        'medium': '#ffc107',
        'high': '#dc3545'
    };
    return colors[status] || '#666';
}

// API Functions
async function fetchData(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        throw error;
    }
}

// Chart Functions
function createChart(elementId, type, data, options) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return null;
    
    // Destroy existing chart if it exists
    if (charts[elementId]) {
        charts[elementId].destroy();
    }
    
    charts[elementId] = new Chart(ctx.getContext('2d'), {
        type: type,
        data: data,
        options: options
    });
    
    return charts[elementId];
}

function updateChart(chartId, newData) {
    if (!charts[chartId]) return;
    
    const chart = charts[chartId];
    
    // Update data
    if (newData.labels) {
        chart.data.labels = newData.labels;
    }
    
    if (newData.datasets) {
        newData.datasets.forEach((dataset, index) => {
            if (chart.data.datasets[index]) {
                Object.assign(chart.data.datasets[index], dataset);
            }
        });
    }
    
    chart.update('none'); // Update without animation
}

// Real-time Data Functions
async function updateProcessParameters() {
    try {
        const data = await fetchData('/api/process-parameters');
        const container = document.getElementById('parameters-list');
        
        if (!container) return;
        
        container.innerHTML = '';
        
        Object.entries(data).forEach(([name, info]) => {
            const element = createParameterElement(name, info);
            container.appendChild(element);
        });
    } catch (error) {
        console.error('Failed to update process parameters:', error);
    }
}

function createParameterElement(name, info) {
    const div = document.createElement('div');
    div.className = `parameter-item ${info.status}`;
    
    const fillPercentage = Math.min(100, Math.max(0, (info.current / info.ideal) * 100));
    
    div.innerHTML = `
        <div class="parameter-name">${name}</div>
        <div class="parameter-value">
            ${formatNumber(info.current)} ${info.unit}
            <span class="parameter-ideal">(ideal: ${info.ideal})</span>
        </div>
        <div class="parameter-bar">
            <div class="parameter-bar-fill" style="width: ${fillPercentage}%"></div>
        </div>
    `;
    
    return div;
}

async function updateQualityPrediction() {
    try {
        const data = await fetchData('/api/predict/realtime', {
            method: 'POST',
            body: JSON.stringify({
                process_data: [{}] // Empty data for simulation
            })
        });
        
        // Update quality score
        const qualityScore = data.predictions?.expected_quality || 87;
        updateElement('quality-score', `${formatNumber(qualityScore)}%`);
        updateElement('gauge-score', Math.round(qualityScore));
        
        // Update weight prediction
        const weight = data.predictions?.expected_weight || 50;
        updateElement('predicted-weight', `${formatNumber(weight)} kg`);
        
        // Update risk level
        const riskLevel = data.risk_level || 'low';
        const riskElement = document.getElementById('anomaly-risk');
        if (riskElement) {
            riskElement.textContent = riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1);
            riskElement.className = `kpi-value risk-${riskLevel}`;
        }
        
        // Update confidence
        const confidence = data.predictions?.confidence || 94;
        updateElement('confidence', `${formatNumber(confidence)}%`);
        
        // Check for alerts
        if (data.alerts && data.alerts.length > 0) {
            data.alerts.forEach(alert => {
                if (!alertQueue.some(a => a.message === alert.message)) {
                    alertQueue.push(alert);
                    showNotification(alert);
                }
            });
        }
    } catch (error) {
        console.error('Failed to update quality prediction:', error);
    }
}

async function updateAlerts() {
    try {
        const alerts = await fetchData('/api/alerts?hours=24');
        const container = document.getElementById('alerts-list');
        
        if (!container) return;
        
        updateElement('alert-count', alerts.length);
        
        container.innerHTML = '';
        
        if (alerts.length === 0) {
            container.innerHTML = '<div class="no-alerts">No recent alerts</div>';
            return;
        }
        
        alerts.slice(0, 5).forEach(alert => {
            const element = createAlertElement(alert);
            container.appendChild(element);
        });
    } catch (error) {
        console.error('Failed to update alerts:', error);
    }
}

function createAlertElement(alert) {
    const div = document.createElement('div');
    div.className = `alert-item ${alert.severity}`;
    
    div.innerHTML = `
        <div class="alert-time">${formatDateTime(alert.timestamp)}</div>
        <div class="alert-message">${alert.message}</div>
        ${alert.parameter ? `<div class="alert-param">${alert.parameter}</div>` : ''}
    `;
    
    return div;
}

function showNotification(alert) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
        border-left: 4px solid ${getStatusColor(alert.severity)};
        max-width: 400px;
    `;
    
    notification.innerHTML = `
        <div style="display: flex; align-items: start; gap: 1rem;">
            <span style="font-size: 1.5rem;">⚠️</span>
            <div>
                <strong style="color: #333;">${alert.severity.toUpperCase()} Alert</strong>
                <p style="margin: 0.5rem 0 0 0; color: #666;">${alert.message}</p>
            </div>
            <button onclick="this.parentElement.parentElement.remove()" 
                    style="background: none; border: none; cursor: pointer; font-size: 1.25rem; color: #999;">
                ×
            </button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Helper Functions
function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function showLoading(show = true) {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = show ? 'flex' : 'none';
    }
}

// Export Functions
async function exportData(format = 'csv') {
    try {
        showLoading(true);
        
        const response = await fetchData('/api/export-report', {
            method: 'POST',
            body: JSON.stringify({ format: format })
        });
        
        showLoading(false);
        
        // Handle download
        if (response.download_url) {
            window.location.href = response.download_url;
        } else {
            // Create download from JSON
            const dataStr = JSON.stringify(response, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', `report_${Date.now()}.json`);
            linkElement.click();
        }
        
        showNotification({
            severity: 'low',
            message: 'Report exported successfully'
        });
    } catch (error) {
        showLoading(false);
        showNotification({
            severity: 'high',
            message: 'Failed to export report: ' + error.message
        });
    }
}

// Initialize Dashboard Components
function initializeDashboard() {
    // Set up periodic updates
    updateIntervals.push(setInterval(updateProcessParameters, 5000));
    updateIntervals.push(setInterval(updateQualityPrediction, 5000));
    updateIntervals.push(setInterval(updateAlerts, 30000));
    
    // Initial load
    updateProcessParameters();
    updateQualityPrediction();
    updateAlerts();
}

// Cleanup function
function cleanup() {
    // Clear all intervals
    updateIntervals.forEach(interval => clearInterval(interval));
    updateIntervals = [];
    
    // Destroy all charts
    Object.values(charts).forEach(chart => chart.destroy());
    charts = {};
}

// Page visibility handling
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Pause updates when page is hidden
        cleanup();
    } else {
        // Resume updates when page is visible
        initializeDashboard();
    }
});

// Export functions for use in HTML
window.dashboardFunctions = {
    initializeDashboard,
    updateProcessParameters,
    updateQualityPrediction,
    updateAlerts,
    exportData,
    showLoading,
    showNotification,
    cleanup
};
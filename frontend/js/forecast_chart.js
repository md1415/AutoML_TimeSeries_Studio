const forecastBtn = document.getElementById('forecastBtn');
const horizonInput = document.getElementById('horizon');
const modelInfoDiv = document.getElementById('modelInfo');
const predictionsTable = document.getElementById('predictionsTable');
const includeConfidence = document.getElementById('includeConfidence');
const compareModels = document.getElementById('compareModels');

let chartInstance = null;
let currentData = null;

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (currentData) {
            renderCurrentChart(btn.dataset.tab);
        }
    });
});

function renderCurrentChart(tabName) {
    if (!currentData) return;

    if (tabName === 'comparison' && currentData.model_comparison) {
        renderComparisonChart(currentData);
    } else if (tabName === 'historical' && currentData.historical_data) {
        renderHistoricalChart(currentData);
    } else {
        renderForecastChart(currentData);
    }
}

function renderForecastChart(data) {
    const predictions = data.predictions;
    const canvas = document.getElementById('forecastChart');
    const fallbackDiv = document.getElementById('chartFallback');

    if (typeof Chart === 'undefined') {
        canvas.style.display = 'none';
        fallbackDiv.style.display = 'block';
        fallbackDiv.innerHTML = '<div style="padding: 20px; background: #fff3cd;">Chart.js not available. Using table view.</div>';
        return;
    }

    canvas.style.display = 'block';
    fallbackDiv.style.display = 'none';

    if (chartInstance) chartInstance.destroy();

    const datasets = [{
        label: 'Forecast',
        data: predictions,
        borderColor: '#ff6b6b',
        backgroundColor: 'rgba(255, 107, 107, 0.1)',
        borderWidth: 3,
        fill: true
    }];

    if (data.confidence_intervals && includeConfidence.checked) {
        const lower = data.confidence_intervals.lower;
        const upper = data.confidence_intervals.upper;

        datasets.push({
            label: 'Confidence Upper',
            data: upper,
            borderColor: 'rgba(255, 107, 107, 0.3)',
            backgroundColor: 'rgba(255, 107, 107, 0.05)',
            borderWidth: 1,
            fill: '+1'
        });

        datasets.push({
            label: 'Confidence Lower',
            data: lower,
            borderColor: 'rgba(255, 107, 107, 0.3)',
            backgroundColor: 'rgba(255, 107, 107, 0.05)',
            borderWidth: 1,
            fill: false
        });
    }

    const ctx = canvas.getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: predictions.length}, (_, i) => `Step ${i + 1}`),
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: `Forecast - Next ${data.horizon} Steps` },
                tooltip: { mode: 'index', intersect: false }
            }
        }
    });
}

function renderComparisonChart(data) {
    const comparison = data.model_comparison;
    const canvas = document.getElementById('forecastChart');
    const fallbackDiv = document.getElementById('chartFallback');

    if (typeof Chart === 'undefined') {
        canvas.style.display = 'none';
        fallbackDiv.style.display = 'block';
        fallbackDiv.innerHTML = '<div style="padding: 20px;">Chart.js not available</div>';
        return;
    }

    canvas.style.display = 'block';
    fallbackDiv.style.display = 'none';

    if (chartInstance) chartInstance.destroy();

    // Define colors for each model
    const modelColors = {
        'XGBoost': '#ff6b6b',
        'Prophet': '#4ecdc4',
        'Random Forest': '#45b7d1',
        'LSTM': '#96ceb4'
    };

    const datasets = [];
    const labels = Array.from({length: comparison.XGBoost?.length || 10}, (_, i) => `Step ${i + 1}`);

    for (const [modelName, predictions] of Object.entries(comparison)) {
        if (predictions && predictions.length > 0) {
            datasets.push({
                label: modelName,
                data: predictions,
                borderColor: modelColors[modelName] || '#888',
                backgroundColor: 'transparent',
                borderWidth: 2,
                fill: false,
                tension: 0.1
            });
        }
    }

    const ctx = canvas.getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Comparison - All 4 Models',
                    font: { size: 16 }
                },
                tooltip: { mode: 'index', intersect: false },
                legend: { position: 'top' }
            }
        }
    });
}

function renderHistoricalChart(data) {
    const historical = data.historical_data;
    const predictions = data.predictions;
    const canvas = document.getElementById('forecastChart');

    if (chartInstance) chartInstance.destroy();

    const allValues = [...historical.values, ...predictions];
    const allLabels = [...historical.dates, ...Array.from({length: predictions.length}, (_, i) => `Forecast ${i + 1}`)];

    const ctx = canvas.getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                { label: 'Historical Data', data: [...historical.values, ...Array(predictions.length).fill(null)], borderColor: '#667eea', borderWidth: 3, fill: false },
                { label: 'Forecast', data: [...Array(historical.values.length).fill(null), ...predictions], borderColor: '#ff6b6b', borderWidth: 3, fill: false }
            ]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'Historical + Forecast' } }
        }
    });
}

function displayModelInfo(data) {
    let infoHtml = `<strong>🤖 Model Selected:</strong> ${data.model_used}<br>`;
    if (data.training_info.model_used) {
        infoHtml += `<strong>📊 Data Points:</strong> ${data.training_info.data_points}<br>`;
        infoHtml += `<strong>⚠️ Anomalies Removed:</strong> ${data.training_info.anomalies_removed}<br>`;
        infoHtml += `<strong>📈 Seasonality Score:</strong> ${(data.training_info.seasonality_score || 0).toFixed(3)}<br>`;
    }
    modelInfoDiv.innerHTML = infoHtml;
}

function displayPredictionsTable(data) {
    let tableHtml = '<h3>📋 Prediction Details</h3><table>';
    tableHtml += '<thead><tr><th>Step</th><th>Value</th>';
    if (data.confidence_intervals) tableHtml += '<th>Lower (95%)</th><th>Upper (95%)</th>';
    tableHtml += '</tr></thead><tbody>';

    data.predictions.forEach((value, i) => {
        tableHtml += `<tr><td>${i + 1}</td><td>${value.toFixed(4)}</td>`;
        if (data.confidence_intervals) {
            tableHtml += `<td>${data.confidence_intervals.lower[i].toFixed(4)}</td>`;
            tableHtml += `<td>${data.confidence_intervals.upper[i].toFixed(4)}</td>`;
        }
        tableHtml += '</tr>';
    });
    tableHtml += '</tbody></table>';
    predictionsTable.innerHTML = tableHtml;
}

forecastBtn.addEventListener('click', async () => {
    if (!currentFileId) return alert('Upload CSV first');

    forecastBtn.disabled = true;
    forecastBtn.textContent = '⏳ Processing...';

    try {
        const response = await fetch('/api/forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_id: currentFileId,
                horizon: parseInt(horizonInput.value),
                include_confidence: includeConfidence.checked,
                compare_models: compareModels.checked
            })
        });

        const data = await response.json();
        if (response.ok) {
            currentData = data;
            displayModelInfo(data);
            displayPredictionsTable(data);
            renderCurrentChart(document.querySelector('.tab-btn.active').dataset.tab);
        } else alert(`Error: ${data.detail}`);
    } catch (error) {
        alert(`Failed: ${error.message}`);
    } finally {
        forecastBtn.disabled = false;
        forecastBtn.textContent = '🔮 Generate Forecast';
    }
});

// Download buttons
document.getElementById('downloadExcelBtn')?.addEventListener('click', async () => {
    if (!currentFileId) return alert('Please upload a file first');
    const horizon = document.getElementById('horizon').value;
    window.open(`/api/export/excel/${currentFileId}?horizon=${horizon}`, '_blank');
});

document.getElementById('downloadPdfBtn')?.addEventListener('click', async () => {
    if (!currentFileId) return alert('Please upload a file first');
    const horizon = document.getElementById('horizon').value;
    window.open(`/api/export/pdf/${currentFileId}?horizon=${horizon}`, '_blank');
});
const forecastBtn = document.getElementById('forecastBtn');
const horizonInput = document.getElementById('horizon');
const modelInfoDiv = document.getElementById('modelInfo');
const predictionsTable = document.getElementById('predictionsTable');

let chartInstance = null;

// Function to check if Chart.js is loaded
function isChartLoaded() {
    return typeof Chart !== 'undefined';
}

// Function to display simple table-based chart if Chart.js is not available
function displaySimpleTableChart(data) {
    const predictions = data.predictions;
    const container = document.getElementById('chartFallback');
    const canvas = document.getElementById('forecastChart');

    canvas.style.display = 'none';
    container.style.display = 'block';

    let html = '<h3>📊 Forecast Values (Chart.js not available)</h3>';
    html += '<table style="width: 100%; border-collapse: collapse;">';
    html += '<thead><tr style="background: #667eea; color: white;">';
    html += '<th style="padding: 8px;">Step</th>';
    html += '<th style="padding: 8px;">Predicted Value</th>';
    html += '<th style="padding: 8px;">Visualization</th>';
    html += '</tr></thead><tbody>';

    const maxValue = Math.max(...predictions);

    predictions.forEach((value, index) => {
        const barWidth = (value / maxValue) * 100;
        html += '<tr>';
        html += `<td style="padding: 8px; text-align: center;">${index + 1}</td>`;
        html += `<td style="padding: 8px; text-align: center;">${value.toFixed(4)}</td>`;
        html += `<td style="padding: 8px;">
                    <div style="background: #667eea; width: ${barWidth}%; height: 20px; border-radius: 10px; color: white; padding-left: 5px; line-height: 20px;">
                        ${value.toFixed(2)}
                    </div>
                  </td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function displayModelInfo(data) {
    let infoHtml = `<strong>🤖 Model Selected:</strong> ${data.model_used}<br>`;

    if (data.training_info.model_used) {
        infoHtml += `<strong>📊 Data Points:</strong> ${data.training_info.data_points}<br>`;
        infoHtml += `<strong>⚠️ Anomalies Removed:</strong> ${data.training_info.anomalies_removed}<br>`;
        if (data.training_info.seasonality_score !== undefined) {
            infoHtml += `<strong>📈 Seasonality Score:</strong> ${data.training_info.seasonality_score.toFixed(3)}<br>`;
        }
    } else if (data.training_info.status) {
        infoHtml += `<strong>💾 Status:</strong> ${data.training_info.status}<br>`;
    }

    modelInfoDiv.innerHTML = infoHtml;
}

function displayChart(data) {
    const predictions = data.predictions;
    const horizon = data.horizon;

    // Check if Chart.js is loaded
    if (!isChartLoaded()) {
        displaySimpleTableChart(data);
        return;
    }

    // Destroy existing chart if it exists
    if (chartInstance) {
        chartInstance.destroy();
    }

    const canvas = document.getElementById('forecastChart');
    const fallbackDiv = document.getElementById('chartFallback');

    canvas.style.display = 'block';
    fallbackDiv.style.display = 'none';

    const ctx = canvas.getContext('2d');

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: predictions.length}, (_, i) => `Step ${i + 1}`),
            datasets: [{
                label: 'Forecast Values',
                data: predictions,
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                borderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#ff6b6b',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: `Time Series Forecast - Next ${horizon} Steps`,
                    font: {
                        size: 18,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 14
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Predicted Value',
                        font: {
                            size: 14
                        }
                    },
                    grid: {
                        color: '#e0e0e0'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time Steps (Future)',
                        font: {
                            size: 14
                        }
                    },
                    grid: {
                        color: '#e0e0e0'
                    }
                }
            }
        }
    });
}

function displayPredictionsTable(data) {
    const predictions = data.predictions;

    let tableHtml = '<h3>📋 Prediction Details</h3><table style="width: 100%; border-collapse: collapse;">';
    tableHtml += '<thead><tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">';
    tableHtml += '<th style="padding: 10px; border: 1px solid #ddd;">Step</th>';
    tableHtml += '<th style="padding: 10px; border: 1px solid #ddd;">Predicted Value</th>';
    tableHtml += '<tr></thead><tbody>';

    predictions.forEach((value, index) => {
        const rowColor = index % 2 === 0 ? '#f9f9f9' : 'white';
        tableHtml += `<tr style="background-color: ${rowColor};">`;
        tableHtml += `<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">${index + 1}</td>`;
        tableHtml += `<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">${value.toFixed(4)}</td>`;
        tableHtml += '</tr>';
    });

    tableHtml += '</tbody></table>';
    predictionsTable.innerHTML = tableHtml;
}

forecastBtn.addEventListener('click', async () => {
    if (!currentFileId) {
        alert('Please upload a CSV file first');
        return;
    }

    const horizon = parseInt(horizonInput.value);
    if (horizon < 1 || horizon > 365) {
        alert('Horizon must be between 1 and 365');
        return;
    }

    forecastBtn.disabled = true;
    forecastBtn.textContent = '⏳ Processing...';

    try {
        const response = await fetch('/api/forecast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_id: currentFileId,
                horizon: horizon
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayModelInfo(data);
            displayChart(data);
            displayPredictionsTable(data);
        } else {
            alert(`Error: ${data.detail}`);
        }
    } catch (error) {
        alert(`Forecast failed: ${error.message}`);
    } finally {
        forecastBtn.disabled = false;
        forecastBtn.textContent = '🔮 Generate Forecast';
    }
});
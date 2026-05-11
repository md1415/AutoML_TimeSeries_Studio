let currentFileId = null;

const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const forecastSection = document.getElementById('forecastSection');

uploadBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
        showStatus('Please select a CSV file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    showStatus('Uploading...', 'info');

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            currentFileId = data.file_id;
            showStatus(`✓ ${data.message}`, 'success');
            forecastSection.style.display = 'block';
        } else {
            showStatus(`✗ Error: ${data.detail}`, 'error');
        }
    } catch (error) {
        showStatus(`✗ Upload failed: ${error.message}`, 'error');
    }
});

function showStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = type;
    setTimeout(() => {
        if (type !== 'error') {
            setTimeout(() => {
                uploadStatus.textContent = '';
                uploadStatus.className = '';
            }, 3000);
        }
    }, 1000);
}
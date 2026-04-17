function switchTab(event, tabId) {
    const panels = document.querySelectorAll('.tab-panel');
    const links = document.querySelectorAll('.tab-link');
    
    panels.forEach(p => p.classList.remove('active'));
    links.forEach(l => l.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

document.getElementById('file-input').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        processAnalysis(e.target.files[0].name);
    }
});

function setResultState({ titleText, color, width, description }) {
    const title = document.getElementById('risk-title');
    const bar = document.getElementById('meter-fill');
    const desc = document.getElementById('risk-desc');

    title.innerText = titleText;
    title.style.color = color;
    bar.style.width = width;
    bar.style.backgroundColor = color;
    desc.innerText = description;
}

async function processAnalysis(source) {
    const box = document.getElementById('result-display');
    const spinner = document.getElementById('spinner');
    const content = document.getElementById('analysis-content');

    box.classList.remove('hidden');
    spinner.classList.remove('hidden');
    content.classList.add('hidden');

    try {
        let payload;

        if (source === 'url') {
            const url = document.getElementById('url-field').value.trim();
            if (!url) {
                throw new Error('Enter a URL to analyze.');
            }

            const response = await fetch('/api/url/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });

            payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.error || 'URL analysis failed.');
            }
        } else {
            const filename = source;
            const isAudio = /\.(wav|mp3|m4a|ogg|flac)$/i.test(filename);
            const endpoint = isAudio ? '/api/voice/detect' : '/api/image/detect';
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ filename })
            });

            payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.error || 'File analysis failed.');
            }
            payload.reason = isAudio
                ? `Audio file ${filename} was analyzed by the backend voice pipeline.`
                : `Image file ${filename} was analyzed by the backend image pipeline.`;
        }

        spinner.classList.add('hidden');
        content.classList.remove('hidden');
        const isSuspicious = payload.result === 'FAKE';
        const score = Math.round((payload.risk_score ?? payload.confidence ?? 0.5) * 100);

        setResultState({
            titleText: isSuspicious ? 'High Risk Detected' : 'Content Authentic',
            color: isSuspicious ? '#ff4d4d' : '#00ff88',
            width: `${Math.max(score, 8)}%`,
            description: payload.reason || 'Analysis completed.'
        });
    } catch (error) {
        spinner.classList.add('hidden');
        content.classList.remove('hidden');
        setResultState({
            titleText: 'Analysis Failed',
            color: '#ffb347',
            width: '100%',
            description: error.message || 'Something went wrong while analyzing this content.'
        });
    }
}

const dropArea = document.getElementById('drop-area');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
    dropArea.addEventListener(name, e => {
        e.preventDefault();
        e.stopPropagation();
    });
});

dropArea.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    processAnalysis(file.name);
});

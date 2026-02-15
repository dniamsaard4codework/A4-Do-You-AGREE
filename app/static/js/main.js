/**
 * NLI Prediction - Client-side logic
 * Handles form submission and result display
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('nli-form');
    const btn = document.getElementById('submit-btn');
    const spinner = document.getElementById('spinner');
    const result = document.getElementById('result');
    const resultTag = document.getElementById('result-tag');
    const confidence = document.getElementById('confidence');
    const similarity = document.getElementById('similarity');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const premise = document.getElementById('premise').value.trim();
        const hypothesis = document.getElementById('hypothesis').value.trim();

        if (!premise || !hypothesis) {
            return;
        }

        // UI: loading state
        btn.disabled = true;
        spinner.style.display = 'block';
        result.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ premise, hypothesis }),
            });

            if (!response.ok) {
                throw new Error(`Server error (${response.status})`);
            }

            const data = await response.json();

            // Update result tag
            resultTag.textContent = data.label;
            resultTag.style.backgroundColor = data.color;

            // Update metrics
            confidence.textContent = (data.confidence * 100).toFixed(1) + '%';
            similarity.textContent = data.similarity.toFixed(4);

            // Show result
            result.style.display = 'block';
        } catch (err) {
            alert('Prediction failed: ' + err.message);
        } finally {
            btn.disabled = false;
            spinner.style.display = 'none';
        }
    });
});

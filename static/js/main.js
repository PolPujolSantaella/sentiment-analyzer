async function analyze() {
    const text = document.getElementById('inputText').value;
    const resultDiv = document.getElementById('resultado');
    const spinner = document.getElementById('spinner');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (!text.trim()) {
        alert('Please enter some text to analyze.');
        return;
    }

    resultDiv.style.display = 'none';
    spinner.style.display = 'inline-block';
    analyzeBtn.disabled = true;

    try {
        const respone = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text, keywords: true })
        });

        if (!respone.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await respone.json();
        const sentiment = data.sentiment.toLowerCase();
        const keywords = data.keywords || [];

        resultDiv.style.display = 'block';

        if (sentiment.includes('positive')) {
            resultDiv.className = 'alert alert-success';
            resultDiv.innerText = '✅ Positive Sentiment';
        } else if (sentiment.includes('negative')) {
            resultDiv.className = 'alert alert-danger';
            resultDiv.innerText = '❌ Negative Sentiment';
        } else {
            resultDiv.className = 'alert alert-warning';
            resultDiv.innerText = '⚠️ Neutral Sentiment';
        }

        if (keywords && keywords.length > 0) {
            resultDiv.innerHTML += `<br><strong>🔑 Keywords:</strong> ${keywords.join(", ")}`;
        }

    } catch (error) {
        resultDiv.style.display = 'block';
        resultDiv.className = 'alert alert-warning';
        resultDiv.innerText = '⚠️ An error occurred while analyzing the text. Please try again later.';
    } finally {
        spinner.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

document.getElementById('analyzeBtn').addEventListener('click', analyze);
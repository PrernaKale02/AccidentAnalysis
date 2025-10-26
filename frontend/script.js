document.addEventListener('DOMContentLoaded', () => {

    // --- Get Element References ---
    const predictionForm = document.getElementById('prediction-form');
    const submitButton = document.getElementById('submit-button');
    
    // Prediction elements
    const resultText = document.getElementById('result-text');
    const chartCanvas = document.getElementById('probability-chart');
    
    // Analysis elements
    const clusterNumText = document.getElementById('cluster-num');
    const clusterDescriptionText = document.getElementById('cluster-description');

    const apiBaseUrl = 'http://127.0.0.1:5000';
    
    // This variable will hold our chart instance so we can destroy it
    let probabilityChart = null;

    // --- Form Submit Event Listener ---
    predictionForm.addEventListener('submit', (event) => {
        event.preventDefault();

        // --- 1. Collect Form Data ---
        const formData = new FormData(predictionForm);
        const inputData = {};
        formData.forEach((value, key) => {
            if (key === 'crash_hour' || key === 'crash_day_of_week') {
                inputData[key] = parseInt(value, 10);
            } else {
                inputData[key] = value;
            }
        });

        console.log('Sending data:', inputData);
        setLoadingState(true);

        // --- 2. Call Both API Endpoints ---
        Promise.all([
            fetch(`${apiBaseUrl}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData),
            }),
            fetch(`${apiBaseUrl}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData),
            })
        ])
        .then(async ([predictResponse, analyzeResponse]) => {
            if (!predictResponse.ok) throw new Error(`Prediction API Error: ${predictResponse.statusText}`);
            if (!analyzeResponse.ok) throw new Error(`Analysis API Error: ${analyzeResponse.statusText}`);

            const predictionData = await predictResponse.json();
            const analysisData = await analyzeResponse.json();
            
            return { predictionData, analysisData };
        })
        .then(({ predictionData, analysisData }) => {
            // --- 3. Update UI with Results ---
            console.log('Prediction:', predictionData);
            console.log('Analysis:', analysisData);

            // Update prediction text
            resultText.textContent = predictionData.predicted_severity || 'Error';
            
            // Update probability chart
            displayProbabilityChart(predictionData.probabilities);

            // Update analysis text
            clusterNumText.textContent = `Cluster #${analysisData.cluster_number}`;
            clusterDescriptionText.textContent = analysisData.description;
        })
        .catch(error => {
            // --- 4. Handle Errors ---
            console.error('Fetch Error:', error);
            resultText.textContent = 'Error: Connection failed.';
            clusterNumText.textContent = '---';
            clusterDescriptionText.textContent = '---';
        })
        .finally(() => {
            setLoadingState(false);
        });
    });

    // --- Helper Function: Update Probability Chart (NEW) ---
    function displayProbabilityChart(probabilities) {
        // Get the canvas context
        const ctx = chartCanvas.getContext('2d');
        
        // Data for the chart
        const labels = probabilities.map(p => p.class);
        const data = probabilities.map(p => p.probability);

        // Check if a chart already exists and destroy it
        if (probabilityChart) {
            probabilityChart.destroy();
        }

        // Create the new chart
        probabilityChart = new Chart(ctx, {
            type: 'bar', // Horizontal bar chart
            data: {
                labels: labels,
                datasets: [{
                    label: 'Model Confidence (%)',
                    data: data,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',  // Red (for top one, 'Fatal')
                        'rgba(255, 159, 64, 0.7)', // Orange
                        'rgba(255, 205, 86, 0.7)', // Yellow
                        'rgba(75, 192, 192, 0.7)',  // Green
                        'rgba(54, 162, 235, 0.7)'   // Blue
                    ],
                    borderColor: 'rgba(255, 255, 255, 0)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y', // This makes the bar chart horizontal
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100, // Percentage
                        title: {
                            display: true,
                            text: 'Probability (%)'
                        }
                    },
                    y: {
                        ticks: {
                            font: {
                                size: 10
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false // Hide the legend
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return ` ${context.dataset.label}: ${context.formattedValue}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    // --- Helper Function: Set Loading State ---
    function setLoadingState(isLoading) {
        if (isLoading) {
            submitButton.disabled = true;
            submitButton.textContent = 'Analyzing...';
            resultText.textContent = 'Loading...';
            // Clear old analysis
            clusterNumText.textContent = '---';
            clusterDescriptionText.textContent = '---';
            // Hide the old chart
            if (probabilityChart) {
                probabilityChart.destroy();
            }
        } else {
            submitButton.disabled = false;
            submitButton.textContent = 'Analyze Accident';
        }
    }
});
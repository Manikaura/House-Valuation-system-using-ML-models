
  // Function to handle form submission
  function submitForm(event) {
    event.preventDefault(); // Prevent the form from submitting normally

    // Get the form data
    var formData = new FormData(event.target);

    // Send a POST request to the '/predict' route
    fetch('/predict', {
      method: 'POST',
      body: formData
    })
      .then(response => response.json()) // Parse the response as JSON
      .then(data => {
        // Update the 'prediction' element with the predicted price
        var predictionElement = document.getElementById('prediction');
        predictionElement.textContent = 'Predicted Price: ' + data[0];
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }


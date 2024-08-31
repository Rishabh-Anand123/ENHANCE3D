// Wait for the DOM to load
document.addEventListener('DOMContentLoaded', () => {
    // Event listener for "Use Image" button
    document.getElementById('image-option').addEventListener('click', () => {
        window.location.href = 'use-image.html'; // Redirect to the image page
    });

    // Event listener for "Use Camera" button
    document.getElementById('camera-option').addEventListener('click', () => {
        window.location.href = 'use-camera.html'; // Redirect to the camera page
    });
});

// Add this code to camera-script.js and script.js

document.getElementById('home-button').addEventListener('click', () => {
    window.location.href = 'index.html'; // Replace 'index.html' with the URL of your home page
});


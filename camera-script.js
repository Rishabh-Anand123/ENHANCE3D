const video = document.getElementById('camera-stream');
const cameraSelect = document.getElementById('camera-select');

// Request access to available video input devices (cameras)
navigator.mediaDevices.enumerateDevices()
    .then(devices => {
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        // Populate the camera selection dropdown
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });

        // Set the initial camera stream
        if (videoDevices.length > 0) {
            startCamera(videoDevices[0].deviceId);
        }
    })
    .catch(error => {
        console.error('Error accessing media devices:', error);
    });

// Function to start the camera stream
function startCamera(deviceId) {
    navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId } } })
        .then(stream => {
            video.srcObject = stream;
            video.play();
        })
        .catch(error => {
            console.error('Error starting camera:', error);
        });
}

// Event listener for camera selection change
cameraSelect.addEventListener('change', () => {
    const selectedDeviceId = cameraSelect.value;
    startCamera(selectedDeviceId);
});


// Add this code to camera-script.js and script.js

document.getElementById('home-button').addEventListener('click', () => {
    window.location.href = 'index.html'; // Replace 'index.html' with the URL of your home page
});

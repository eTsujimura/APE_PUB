/*
camera func
*/

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap = document.getElementById('snap');
const fileInput = document.getElementById('fileInput');
const uploadForm = document.getElementById('uploadForm');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing the camera: ", err);
    });

snap.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, 640, 480);
    canvas.toBlob(blob => {
        const file = new File([blob], "photo.png", { type: "image/png" });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
    });
});

uploadForm.addEventListener('submit', (event) => {
    if (fileInput.files.length === 0) {
        event.preventDefault();
        alert("Please snap a photo first!");
    }
});
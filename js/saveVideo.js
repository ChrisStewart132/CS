// creates a btn to shadow record roughly more than the last 'x' seconds (firefox bug mutes audio channel use chrome/brave)
const durationInSeconds = 180;
const options = {
  mimeType: 'video/webm',// video/webm, video/mp4, video/quicktime, video/mpeg, video/x-matroska, video/ogg
  videoBitsPerSecond: 4 * 10 ** 6,// Mbits / s
  audioBitsPerSecond: 128 * 10 ** 3// Kbits / s
};
const videoElement = document.getElementsByTagName("video")[0];
let mediaRecorder;
let recordedChunks = [];

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const anchorElement = document.createElement('a');
  anchorElement.href = url;
  anchorElement.download = fileName;

  document.body.appendChild(anchorElement);
  anchorElement.click();
  document.body.removeChild(anchorElement);
  URL.revokeObjectURL(url);
}

function startRecording() {
  try {
    mediaRecorder = new MediaRecorder(videoElement.captureStream(), options);
  } catch (e) {
    console.log('Save Video: firefox has a bug that mutes the stream audio in browser but records fine')
    try {
      mediaRecorder = new MediaRecorder(videoElement.mozCaptureStream(), options);
    } catch (error) {
      console.error('Error creating MediaRecorder:', error);
      return;
    }
  }

  mediaRecorder.ondataavailable = event => {
    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data);

      // Remove the first half of the recordedChunks array if its size exceeds a threshold
      const maxSize = 1 * 10 ** 6;
      if (recordedChunks.length > maxSize)
        recordedChunks.splice(0, Math.floor(maxSize / 2));// deletes elements 0->x
    }
  };

  mediaRecorder.start();
}

function stopRecordingAndSave() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') {
    console.warn('No active recording to save.');
    return;
  }
  mediaRecorder.stop();
  mediaRecorder.onstop = () => {
    const maxDuration = durationInSeconds * 1000;
    const endTime = recordedChunks[recordedChunks.length - 1].startTime + recordedChunks[recordedChunks.length - 1].duration;// starts 
    let index30sFromEnd = recordedChunks.length - 1;

    // Find the index 30 seconds from the end
    while (index30sFromEnd >= 0 && endTime - recordedChunks[index30sFromEnd].startTime > maxDuration) {
      index30sFromEnd--;
    }

    // Splice the array from the index 30 seconds from the end and save the recording
    const recordingChunks = recordedChunks.slice(index30sFromEnd);
    const fullBlob = new Blob(recordingChunks, { type: options.mimeType });
    downloadBlob(fullBlob, options.mimeType);
  };
}

function saveVideo() {
  stopRecordingAndSave()
  startRecording()
}

function createStickyButton() {
  // Create the button element
  const button = document.createElement('saveVideoBtn');
  button.textContent = 'Save Video';
  // Apply CSS styles to the button
  button.style.position = 'fixed';
  button.style.top = '50%';
  button.style.left = '20%';
  button.style.transform = 'translate(-50%, -20%)';
  button.style.backgroundColor = 'grey'
  button.style.zIndex = '33';
  button.onclick = saveVideo
  // Append the button to the body of the page
  document.body.appendChild(button);
}

// Call the function to create the sticky button
createStickyButton();

// Start recording
setTimeout(startRecording, 100);
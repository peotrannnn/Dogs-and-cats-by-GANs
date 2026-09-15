const generateButton = document.getElementById("generateButton");
const progressBar = document.getElementById("progressBar");
const statusText = document.getElementById("status");
const placeholder = document.getElementById("placeholder");
const seedText = document.getElementById("seedText");

const imageA = document.getElementById("imageA");
const imageB = document.getElementById("imageB");

const CROSSFADE_MS = 950;
const FRAME_HOLD_MS = 180;

let currentImage = imageA;
let nextImage = imageB;


function selectedSpecies() {
  return document.querySelector('input[name="species"]:checked').value;
}


function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}


function setProgress(percent) {
  const safe = Math.max(0, Math.min(100, percent));
  progressBar.style.width = `${safe}%`;
}


function resetImages() {
  imageA.hidden = true;
  imageB.hidden = true;

  imageA.classList.remove("active");
  imageB.classList.remove("active");

  imageA.src = "";
  imageB.src = "";

  currentImage = imageA;
  nextImage = imageB;
}


function preload(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = resolve;
    img.onerror = reject;
    img.src = src;
  });
}


async function showFrame(frame, firstFrame) {
  await preload(frame.image);

  placeholder.hidden = true;

  if (firstFrame) {
    currentImage.src = frame.image;
    currentImage.alt = frame.label_text;
    currentImage.hidden = false;

    // Force style calculation before enabling opacity.
    currentImage.getBoundingClientRect();
    currentImage.classList.add("active");

    await sleep(CROSSFADE_MS);
    return;
  }

  nextImage.src = frame.image;
  nextImage.alt = frame.label_text;
  nextImage.hidden = false;
  nextImage.classList.remove("active");

  nextImage.getBoundingClientRect();

  nextImage.classList.add("active");
  currentImage.classList.remove("active");

  await sleep(CROSSFADE_MS);

  currentImage.hidden = true;
  currentImage.src = "";

  const oldCurrent = currentImage;
  currentImage = nextImage;
  nextImage = oldCurrent;

  await sleep(FRAME_HOLD_MS);
}


async function generateEvolution() {
  const species = selectedSpecies();

  generateButton.disabled = true;
  document.querySelectorAll('input[name="species"]').forEach((input) => {
    input.disabled = true;
  });

  resetImages();
  placeholder.hidden = false;
  placeholder.textContent = "Preparing...";
  seedText.textContent = "";

  setProgress(0);
  statusText.textContent = "Starting sobelv5...";

  let firstFrame = true;

  try {
    const response = await fetch("/api/evolution", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        species: species,
      }),
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "Request failed");
    }

    if (!response.body) {
      throw new Error("Streaming response is unavailable in this browser");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();

      if (value) {
        buffer += decoder.decode(value, { stream: true });

        let newlineIndex;

        while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);

          if (!line) {
            continue;
          }

          const message = JSON.parse(line);

          if (message.type === "start") {
            seedText.textContent = `seed: ${message.seed}`;
            statusText.textContent =
              `Preparing ${message.total_frames} training checkpoints...`;
          }

          if (message.type === "frame") {
            statusText.textContent = message.label_text;
            setProgress(message.progress);

            await showFrame(message, firstFrame);
            firstFrame = false;
          }

          if (message.type === "error") {
            throw new Error(message.error || "Generation failed");
          }

          if (message.type === "done") {
            setProgress(100);
            statusText.textContent = "Final model";
          }
        }
      }

      if (done) {
        break;
      }
    }
  } catch (error) {
    console.error(error);

    resetImages();
    placeholder.hidden = false;
    placeholder.textContent = "Failed";

    setProgress(0);
    statusText.textContent = error.message;
  } finally {
    generateButton.disabled = false;

    document.querySelectorAll('input[name="species"]').forEach((input) => {
      input.disabled = false;
    });
  }
}


generateButton.addEventListener("click", generateEvolution);

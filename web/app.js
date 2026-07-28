const sampleResult = {
  title: "M31 (Andromeda Galaxy)",
  messierId: "M31",
  category: "Galaxy",
  confidence: "100.0%",
  confidenceLabel: "Very High",
  description:
    "A massive system of stars, gas, dust, and dark matter held together by gravity.",
  referenceImage: "/Dataset/Galaxy/M31.jpg",
  starMapImage: "/Map_Dataset/Galaxy/M31.jpg",
  matches: [
    {
      rank: 1,
      title: "M31 (Andromeda Galaxy)",
      category: "Galaxy",
      confidence: "100.0%",
      image: "/Dataset/Galaxy/M31.jpg",
    },
    {
      rank: 2,
      title: "M109 (Vacuum Cleaner Galaxy)",
      category: "Galaxy",
      confidence: "86.69%",
      image: "/Dataset/Galaxy/M109.jpg",
    },
    {
      rank: 3,
      title: "M66 (Leo Triplet Galaxy)",
      category: "Galaxy",
      confidence: "85.63%",
      image: "/Dataset/Galaxy/M66.jpg",
    },
  ],
};

const fileInput = document.getElementById("fileInput");
const dropInput = document.getElementById("dropInput");
const demoButton = document.getElementById("demoButton");
const dropzone = document.getElementById("dropzone");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const errorMessage = document.getElementById("errorMessage");
const retryButton = document.getElementById("retryButton");
const resultsSection = document.getElementById("results");
const uploadedPreview = document.getElementById("uploadedPreview");
const referencePreview = document.getElementById("referencePreview");
const starMapPreview = document.getElementById("starMapPreview");
const matchesGrid = document.getElementById("matchesGrid");
const stellarInfoTitle = document.getElementById("stellarInfoTitle");
const stellarInfoName = document.getElementById("stellarInfoName");
const stellarInfoConstellation = document.getElementById("stellarInfoConstellation");
const stellarInfoDistance = document.getElementById("stellarInfoDistance");
const stellarInfoDiameter = document.getElementById("stellarInfoDiameter");
const stellarInfoDiscovery = document.getElementById("stellarInfoDiscovery");

function showLoading() {
  loadingState.classList.remove("hidden");
  errorState.classList.add("hidden");
  resultsSection.classList.add("hidden");
}

function hideLoading() {
  loadingState.classList.add("hidden");
}

function showError(message) {
  hideLoading();
  resultsSection.classList.add("hidden");
  errorMessage.textContent = message;
  errorState.classList.remove("hidden");
}

function titleCase(value) {
  return value.replace(/\b\w/g, (char) => char.toUpperCase());
}

function hideError() {
  errorState.classList.add("hidden");
}

function showResults() {
  hideError();
  resultsSection.classList.remove("hidden");
}

function renderStellarInfoBox(stellarInfo, messierId) {
  if (stellarInfoTitle) {
    stellarInfoTitle.textContent = `${messierId} | ${stellarInfo.name || "Unknown"}`;
  }
  if (stellarInfoName) {
    stellarInfoName.textContent = stellarInfo.name || "Unknown";
  }
  if (stellarInfoConstellation) {
    stellarInfoConstellation.textContent = stellarInfo.constellation || "Unknown";
  }
  if (stellarInfoDistance) {
    stellarInfoDistance.textContent = stellarInfo.distance_ly || "Unknown";
  }
  if (stellarInfoDiameter) {
    stellarInfoDiameter.textContent = stellarInfo.diameter_ly || "Unknown";
  }
  if (stellarInfoDiscovery) {
    stellarInfoDiscovery.textContent = stellarInfo.year_of_discovery || "Unknown";
  }
}

function renderResult(result) {
  document.getElementById("predictedTitle").textContent = result.title;
  document.getElementById("messierId").textContent = result.messierId;
  document.getElementById("categoryValue").textContent = result.category;
  document.getElementById("confidenceValue").textContent = result.confidence;
  document.getElementById("confidenceLabel").textContent = titleCase(result.confidenceLabel);
  document.getElementById("descriptionText").textContent = result.description;
  document.getElementById("detailObjectName").textContent = result.title;
  document.getElementById("detailMessierId").textContent = result.messierId;
  document.getElementById("detailCategory").textContent = result.category;
  document.getElementById("detailConfidence").textContent = result.confidence;
  referencePreview.src = result.referenceImage;
  starMapPreview.src = result.starMapImage;

  matchesGrid.innerHTML = result.matches
    .map(
      (match, index) => `
        <article class="match-item ${index === 0 ? "primary" : ""}">
          <span class="match-rank">Rank ${match.rank}</span>
          <h4>${match.title}</h4>
          <p>${match.category}</p>
          <div class="match-footer">
            <strong>${match.confidence}</strong>
            <span class="match-link">View Details</span>
          </div>
          <div class="match-thumb">
            <img src="${match.image}" alt="${match.title}" />
          </div>
        </article>
      `
    )
    .join("");

  renderStellarInfoBox(result.stellarInfo, result.messierId);
  hideLoading();
  showResults();
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyzeImage(file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    uploadedPreview.src = event.target.result;
  };
  reader.readAsDataURL(file);

  showLoading();

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();

    // 🔥 DEBUG START
    console.log("==== BACKEND RESPONSE ====");
    console.log(payload);
    console.log("STELLAR INFO:", payload.stellar_info);
    // 🔥 DEBUG END

    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed");
    }

    const topMatch = payload.top_match;

    // ✅ SAFE stellar info handling
    const stellarInfo =
      payload.stellar_info && !payload.stellar_info.error
        ? payload.stellar_info
        : {
            name: "Unavailable",
            constellation: "-",
            distance_ly: "-",
            diameter_ly: "-",
            year_of_discovery: "-",
          };

    renderResult({
      title: topMatch.title,
      messierId: topMatch.messier_id,
      category: topMatch.category,
      confidence: `${topMatch.similarity_score.toFixed(2)}%`,
      confidenceLabel: topMatch.confidence_label,
      description: topMatch.description,
      referenceImage: topMatch.reference_image,
      starMapImage: topMatch.star_map_image,
      matches: payload.matches.map((match, index) => ({
        rank: index + 1,
        title: match.title,
        category: match.category,
        confidence: `${match.similarity_score.toFixed(2)}%`,
        image: match.reference_image,
      })),
      stellarInfo: stellarInfo,
    });
  } catch (error) {
    console.error("ERROR:", error);
    showError(error.message);
  }
}

function handleFileSelect(event) {
  const [file] = event.target.files || [];
  analyzeImage(file);
}

fileInput.addEventListener("change", handleFileSelect);
dropInput.addEventListener("change", handleFileSelect);

demoButton.addEventListener("click", () => {
  uploadedPreview.src = sampleResult.referenceImage;
  showLoading();
  window.setTimeout(() => renderResult(sampleResult), 900);
});

retryButton.addEventListener("click", () => {
  hideError();
  document.getElementById("upload").scrollIntoView({ behavior: "smooth", block: "start" });
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files || [];
  analyzeImage(file);
});


renderResult(sampleResult);

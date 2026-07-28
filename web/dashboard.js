const dashboardElements = {
  kpiObjects: document.getElementById("kpiObjects"),
  kpiMaps: document.getElementById("kpiMaps"),
  kpiApproach: document.getElementById("kpiApproach"),
  kpiApproachDesc: document.getElementById("kpiApproachDesc"),
  kpiInference: document.getElementById("kpiInference"),
  kpiInferenceDesc: document.getElementById("kpiInferenceDesc"),
  metricAccuracy: document.getElementById("metricAccuracy"),
  metricRunnerUp: document.getElementById("metricRunnerUp"),
  metricGap: document.getElementById("metricGap"),
  metricCoverage: document.getElementById("metricCoverage"),
  evalPcaComponents: document.getElementById("evalPcaComponents"),
  evalPcaVariance: document.getElementById("evalPcaVariance"),
  evalTrainSamples: document.getElementById("evalTrainSamples"),
  evalTestSamples: document.getElementById("evalTestSamples"),
  trainingSamples: document.getElementById("trainingSamples"),
  trainingSmoteSamples: document.getElementById("trainingSmoteSamples"),
  pipelineSteps: document.getElementById("pipelineSteps"),
  trainingChart: document.getElementById("trainingChart"),
  modelComparisonChart: document.getElementById("modelComparisonChart"),
  categoryChart: document.getElementById("categoryChart"),
  featureChart: document.getElementById("featureChart"),
  weightsChart: document.getElementById("weightsChart"),
  thresholdGrid: document.getElementById("thresholdGrid"),
  avgResolution: document.getElementById("avgResolution"),
  resolutionRange: document.getElementById("resolutionRange"),
  lowResCount: document.getElementById("lowResCount"),
  grayscaleCount: document.getElementById("grayscaleCount"),
  strengthList: document.getElementById("strengthList"),
  limitationList: document.getElementById("limitationList"),
  metricNotes: document.getElementById("metricNotes"),
  modelMetricsGrid: document.getElementById("modelMetricsGrid"),
  validationProtocol: document.getElementById("validationProtocol"),
  evaluationNotes: document.getElementById("evaluationNotes"),
};

function renderBarChart(target, entries, valueKey, suffix = "") {
  const maxValue = Math.max(...entries.map((entry) => entry[valueKey]));
  target.innerHTML = entries
    .map((entry) => {
      const width = `${(entry[valueKey] / maxValue) * 100}%`;
      return `
        <div class="chart-row">
          <div class="chart-labels">
            <span>${entry.label}</span>
            <strong>${entry[valueKey]}${suffix}</strong>
          </div>
          <div class="chart-track">
            <div class="chart-fill" style="width: ${width}"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderDashboard(payload) {
  const { dataset, metrics, model, ml_evaluation } = payload;

  dashboardElements.kpiObjects.textContent = dataset.total_objects;
  dashboardElements.kpiMaps.textContent = dataset.star_map_count;
  dashboardElements.metricAccuracy.textContent = `${metrics.reference_self_match_accuracy}%`;
  dashboardElements.metricRunnerUp.textContent = `${metrics.average_runner_up_score}%`;
  dashboardElements.metricGap.textContent = `${metrics.average_top1_gap}%`;
  dashboardElements.metricCoverage.textContent = `${metrics.catalog_coverage}%`;
  dashboardElements.kpiApproach.textContent = model.approach;
  dashboardElements.kpiApproachDesc.textContent = "Primary recognition strategy active in production.";
  dashboardElements.kpiInference.textContent = model.inference_type.includes("SpaceDataset")
    ? "ML + Similarity Hybrid"
    : "Fast Local Retrieval";
  dashboardElements.kpiInferenceDesc.textContent = model.inference_type;
  dashboardElements.avgResolution.textContent = `${dataset.avg_width} x ${dataset.avg_height}`;
  dashboardElements.resolutionRange.textContent =
    `${dataset.min_width} x ${dataset.min_height} to ${dataset.max_width} x ${dataset.max_height}`;
  dashboardElements.lowResCount.textContent = dataset.low_resolution_count;
  dashboardElements.grayscaleCount.textContent = dataset.grayscale_count;
  if (ml_evaluation && ml_evaluation.evaluation) {
    const pcaVariance = ml_evaluation.evaluation.pca_explained_variance;
    dashboardElements.evalPcaComponents.textContent = ml_evaluation.evaluation.pca_components;
    dashboardElements.evalPcaVariance.textContent =
      typeof pcaVariance === "number" ? `${pcaVariance}%` : `${pcaVariance}`;
    dashboardElements.evalTrainSamples.textContent = ml_evaluation.evaluation.train_samples;
    dashboardElements.evalTestSamples.textContent = ml_evaluation.evaluation.test_samples;
    dashboardElements.validationProtocol.textContent = ml_evaluation.evaluation.validation_protocol;
    dashboardElements.evaluationNotes.innerHTML = ml_evaluation.evaluation.notes
      .map((item) => `<li>${item}</li>`)
      .join("");
    dashboardElements.trainingSamples.textContent = ml_evaluation.evaluation.train_samples || "-";
    dashboardElements.trainingSmoteSamples.textContent =
      ml_evaluation.evaluation.train_samples_after_smote || "-";
  }

  if (ml_evaluation && Array.isArray(ml_evaluation.models)) {
    dashboardElements.modelMetricsGrid.innerHTML = ml_evaluation.models
      .map(
        (entry) => `
          <article class="model-card metric-model-card">
            <span class="eyebrow">${entry.label}</span>
            <h4>${entry.accuracy}% Accuracy</h4>
            <div class="model-metric-list">
              <div class="detail-item">
                <span>Macro F1</span>
                <strong>${entry.macro_f1}%</strong>
              </div>
              <div class="detail-item">
                <span>Top-3 Accuracy</span>
                <strong>${entry.top3_accuracy}%</strong>
              </div>
              ${
                entry.fit_seconds !== undefined
                  ? `
              <div class="detail-item">
                <span>Train Time</span>
                <strong>${entry.fit_seconds}s</strong>
              </div>`
                  : ""
              }
              ${
                entry.inference_ms_per_image !== undefined
                  ? `
              <div class="detail-item">
                <span>Inference / image</span>
                <strong>${entry.inference_ms_per_image} ms</strong>
              </div>`
                  : ""
              }
            </div>
          </article>
        `
      )
      .join("");

    renderBarChart(
      dashboardElements.modelComparisonChart,
      ml_evaluation.models,
      "accuracy",
      "%"
    );

    if (ml_evaluation.evaluation.train_samples_after_smote) {
      renderBarChart(dashboardElements.trainingChart, [
        { label: "Train", value: ml_evaluation.evaluation.train_samples },
        {
          label: "After SMOTE",
          value: ml_evaluation.evaluation.train_samples_after_smote,
        },
        { label: "Test", value: ml_evaluation.evaluation.test_samples },
      ], "value");
    }
  }

  dashboardElements.pipelineSteps.innerHTML = model.pipeline_steps
    .map(
      (step, index) => `
        <article class="pipeline-step">
          <span class="pipeline-index">0${index + 1}</span>
          <p>${step}</p>
        </article>
      `
    )
    .join("");

  renderBarChart(
    dashboardElements.categoryChart,
    dataset.category_counts,
    "count"
  );
  renderBarChart(
    dashboardElements.featureChart,
    model.feature_blocks,
    "size"
  );
  renderBarChart(
    dashboardElements.weightsChart,
    model.distance_weights.map((entry) => ({
      ...entry,
      weight_percent: Math.round(entry.weight * 100),
    })),
    "weight_percent",
    "%"
  );

  dashboardElements.thresholdGrid.innerHTML = model.confidence_thresholds
    .map(
      (entry) => `
        <div class="threshold-card">
          <span class="eyebrow">${entry.label}</span>
          <strong>${entry.range}</strong>
        </div>
      `
    )
    .join("");

  dashboardElements.strengthList.innerHTML = model.strengths
    .map((item) => `<li>${item}</li>`)
    .join("");
  dashboardElements.limitationList.innerHTML = model.limitations
    .map((item) => `<li>${item}</li>`)
    .join("");
  dashboardElements.metricNotes.innerHTML = model.metric_notes
    .map((item) => `<li>${item}</li>`)
    .join("");
}

async function loadDashboard() {
  const response = await fetch("/api/dashboard");
  const payload = await response.json();
  renderDashboard(payload);
}

loadDashboard().catch(() => {
  dashboardElements.pipelineSteps.innerHTML =
    "<article class='pipeline-step'><p>Dashboard data could not be loaded.</p></article>";
});

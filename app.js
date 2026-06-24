const state = {
  demo: null,
  selectedSample: null,
  currentFile: null,
  prediction: null,
};

const elements = {
  app: document.querySelector("#app"),
  setupScreen: document.querySelector("#setup-screen"),
  setupMessage: document.querySelector("#setup-message"),
  serviceStatus: document.querySelector("#service-status"),
  modelName: document.querySelector("#model-name"),
  demoReady: document.querySelector("#demo-ready"),
  sidebarClassCount: document.querySelector("#sidebar-class-count"),
  sampleList: document.querySelector("#sample-list"),
  inspectionImage: document.querySelector("#inspection-image"),
  inputSource: document.querySelector("#input-source"),
  imageUpload: document.querySelector("#image-upload"),
  predictUpload: document.querySelector("#predict-upload"),
  resultEmpty: document.querySelector("#result-empty"),
  resultContent: document.querySelector("#result-content"),
  resultLatency: document.querySelector("#result-latency"),
  outcomeIndicator: document.querySelector("#outcome-indicator"),
  outcomeType: document.querySelector("#outcome-type"),
  outcomeLabel: document.querySelector("#outcome-label"),
  matchList: document.querySelector("#match-list"),
  thresholdNote: document.querySelector("#threshold-note"),
  correctionLabel: document.querySelector("#correction-label"),
  submitFeedback: document.querySelector("#submit-feedback"),
  teachForm: document.querySelector("#teach-form"),
  teachLabel: document.querySelector("#teach-label"),
  teachImages: document.querySelector("#teach-images"),
  teachFileCount: document.querySelector("#teach-file-count"),
  classList: document.querySelector("#class-list"),
  metricGrid: document.querySelector("#metric-grid"),
  endpointList: document.querySelector("#endpoint-list"),
  toast: document.querySelector("#toast"),
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail : body;
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return body;
}

function showToast(message, tone = "success") {
  elements.toast.textContent = message;
  elements.toast.className = `toast visible ${tone}`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => (elements.toast.className = "toast"), 3600);
}

function setServiceState(online, label) {
  elements.serviceStatus.className = `service-status ${online ? "online" : "offline"}`;
  elements.serviceStatus.innerHTML = `<i></i> ${label}`;
}

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("active", page.dataset.page === view);
  });
  if (view === "classes") refreshClasses();
  if (view === "signals") refreshMetrics();
  window.location.hash = view;
}

function renderSamples(samples) {
  elements.sampleList.innerHTML = samples
    .map(
      (sample) => `
        <button class="sample-item" data-sample-id="${sample.id}">
          <img src="${sample.image_url}" alt="${sample.title}" />
          <span><strong>${sample.title}</strong><small>${sample.expected}</small></span>
          <i aria-hidden="true"></i>
        </button>`,
    )
    .join("");

  elements.sampleList.querySelectorAll(".sample-item").forEach((button) => {
    button.addEventListener("click", () => selectDemoSample(button.dataset.sampleId));
  });
}

async function selectDemoSample(sampleId, silent = false) {
  const sample = state.demo.samples.find((item) => item.id === sampleId);
  if (!sample) return;
  state.selectedSample = sample;
  state.currentFile = null;
  elements.imageUpload.value = "";
  elements.predictUpload.disabled = true;
  elements.inspectionImage.src = sample.image_url;
  elements.inspectionImage.alt = sample.title;
  elements.inputSource.textContent = "Demo fixture";
  document.querySelectorAll(".sample-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.sampleId === sampleId);
  });
  await predictDemoSample(sampleId, silent);
}

async function predictDemoSample(sampleId, silent) {
  const startedAt = performance.now();
  setResultLoading();
  try {
    const result = await api(`/v1/demo/samples/${sampleId}/predict`, { method: "POST" });
    renderPrediction(result, performance.now() - startedAt);
    if (!silent) showToast("Inspection completed");
  } catch (error) {
    showResultError(error.message);
  }
}

async function predictUploadedImage() {
  if (!state.currentFile) return;
  const form = new FormData();
  form.append("file", state.currentFile);
  const startedAt = performance.now();
  setResultLoading();
  try {
    const result = await api("/v1/predict?top_k=3", { method: "POST", body: form });
    renderPrediction(result, performance.now() - startedAt);
  } catch (error) {
    showResultError(error.message);
  }
}

function setResultLoading() {
  elements.resultEmpty.hidden = false;
  elements.resultContent.hidden = true;
  elements.resultEmpty.innerHTML = '<div class="spinner"></div><strong>Inspecting image</strong><p>Comparing against class prototypes...</p>';
  elements.resultLatency.textContent = "Running";
}

function renderPrediction(result, elapsedMs) {
  state.prediction = result;
  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = false;
  elements.resultLatency.textContent = `${Math.round(elapsedMs)} ms`;
  elements.outcomeType.textContent = result.is_unknown ? "REJECTED AS UNKNOWN" : "KNOWN CLASS";
  elements.outcomeLabel.textContent = result.is_unknown ? "No confident match" : result.label;
  elements.outcomeIndicator.className = `outcome-indicator ${result.is_unknown ? "unknown" : "known"}`;
  elements.matchList.innerHTML = result.matches
    .map((match, index) => {
      const percent = Math.max(0, Math.min(100, match.similarity * 100));
      return `<div class="match-row"><div><span>${index + 1}. ${match.label}</span><strong>${match.similarity.toFixed(3)}</strong></div><div class="match-track"><i style="width:${percent}%"></i></div><small>${match.examples} reference examples</small></div>`;
    })
    .join("");
  elements.thresholdNote.textContent = `Decision threshold: ${result.threshold.toFixed(2)} cosine similarity.`;
}

function showResultError(message) {
  elements.resultContent.hidden = true;
  elements.resultEmpty.hidden = false;
  elements.resultEmpty.innerHTML = `<strong>Inspection unavailable</strong><p>${message}</p>`;
  elements.resultLatency.textContent = "Error";
  showToast(message, "error");
}

async function submitFeedback() {
  const label = elements.correctionLabel.value.trim();
  if (!label) return showToast("Enter the correct class name", "error");
  const form = new FormData();
  if (state.currentFile) {
    form.append("file", state.currentFile);
  } else if (state.selectedSample) {
    const image = await fetch(state.selectedSample.image_url).then((response) => response.blob());
    form.append("file", image, `${state.selectedSample.id}.png`);
  } else {
    return showToast("Select an image first", "error");
  }
  try {
    const result = await api(`/v1/feedback/${encodeURIComponent(label)}`, { method: "POST", body: form });
    elements.correctionLabel.value = "";
    showToast(`${result.label} updated to ${result.total_examples} examples`);
    await refreshClasses();
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function teachClass(event) {
  event.preventDefault();
  const label = elements.teachLabel.value.trim();
  const files = Array.from(elements.teachImages.files);
  if (!label || !files.length) return;
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const button = elements.teachForm.querySelector("button[type=submit]");
  button.disabled = true;
  button.textContent = "Embedding examples...";
  try {
    const result = await api(`/v1/classes/${encodeURIComponent(label)}/examples`, { method: "POST", body: form });
    showToast(`${result.examples_added} examples added to ${result.label}`);
    elements.teachForm.reset();
    elements.teachFileCount.textContent = "No images selected";
    await refreshClasses();
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Add to class memory";
  }
}

async function refreshClasses() {
  try {
    const { classes } = await api("/v1/classes");
    elements.sidebarClassCount.textContent = `${classes.length} ${classes.length === 1 ? "class" : "classes"}`;
    elements.classList.innerHTML = classes.length
      ? classes.map((item) => `<div class="class-row"><span class="class-dot"></span><div><strong>${item.label}</strong><small>Updated ${formatDate(item.updated_at)}</small></div><span>${item.examples} examples</span><button data-delete-class="${item.label}" title="Delete class" aria-label="Delete ${item.label}">&times;</button></div>`).join("")
      : '<div class="empty-state"><strong>No classes yet</strong><p>Add examples to create the first prototype.</p></div>';
    elements.classList.querySelectorAll("[data-delete-class]").forEach((button) => {
      button.addEventListener("click", () => deleteClass(button.dataset.deleteClass));
    });
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function deleteClass(label) {
  if (!window.confirm(`Delete class "${label}" from prototype memory?`)) return;
  try {
    await api(`/v1/classes/${encodeURIComponent(label)}`, { method: "DELETE" });
    showToast(`${label} deleted`);
    await refreshClasses();
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function refreshMetrics() {
  try {
    const metrics = await api("/v1/metrics");
    const cards = [
      ["Classes", metrics.class_count, "Active prototypes"],
      ["Examples", metrics.example_count, "Embedded references"],
      ["Predictions", metrics.observations, `Rolling window of ${metrics.window_size}`],
      ["Unknown rate", `${(metrics.unknown_rate * 100).toFixed(1)}%`, "Traffic rejected by threshold"],
      ["Mean similarity", formatMetric(metrics.mean_top_similarity), "Average nearest match"],
      ["P10 similarity", formatMetric(metrics.p10_top_similarity), "Lower confidence boundary"],
    ];
    elements.metricGrid.innerHTML = cards.map(([label, value, note]) => `<div class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join("");
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderEndpoints() {
  const endpoints = [
    ["POST", "/v1/predict", "Classify an uploaded image"],
    ["POST", "/v1/classes/{label}/examples", "Add reference examples"],
    ["POST", "/v1/feedback/{label}", "Apply a human correction"],
    ["GET", "/v1/classes", "Inspect class memory"],
    ["GET", "/v1/metrics", "Read rolling operational signals"],
    ["GET", "/health", "Check model and demo readiness"],
  ];
  elements.endpointList.innerHTML = endpoints.map(([method, path, description]) => `<div class="endpoint-row"><span class="method ${method.toLowerCase()}">${method}</span><code>${path}</code><p>${description}</p></div>`).join("");
}

function formatMetric(value) {
  return value === null ? "No data" : Number(value).toFixed(3);
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  elements.imageUpload.addEventListener("change", () => {
    const [file] = elements.imageUpload.files;
    state.currentFile = file || null;
    state.selectedSample = null;
    elements.predictUpload.disabled = !file;
    if (file) {
      elements.inspectionImage.src = URL.createObjectURL(file);
      elements.inspectionImage.alt = file.name;
      elements.inputSource.textContent = "Uploaded image";
      document.querySelectorAll(".sample-item").forEach((button) => button.classList.remove("active"));
    }
  });
  elements.predictUpload.addEventListener("click", predictUploadedImage);
  elements.submitFeedback.addEventListener("click", submitFeedback);
  elements.teachForm.addEventListener("submit", teachClass);
  elements.teachImages.addEventListener("change", () => {
    const count = elements.teachImages.files.length;
    elements.teachFileCount.textContent = count ? `${count} image${count === 1 ? "" : "s"} selected` : "No images selected";
  });
  document.querySelector("#refresh-classes").addEventListener("click", refreshClasses);
}

async function initialize() {
  bindEvents();
  renderEndpoints();
  try {
    elements.setupMessage.textContent = "Checking service and demo memory...";
    const health = await api("/health");
    setServiceState(true, "Service online");
    elements.modelName.textContent = health.model;

    state.demo = await api("/v1/demo");
    renderSamples(state.demo.samples);
    if (!state.demo.ready) {
      elements.setupMessage.textContent = "Embedding built-in inspection fixtures. The first setup may take a moment...";
      state.demo = await api("/v1/demo/bootstrap", { method: "POST" });
    }

    elements.demoReady.textContent = "Demo ready";
    elements.demoReady.classList.add("ready");
    await Promise.all([refreshClasses(), refreshMetrics()]);
    elements.setupScreen.classList.add("done");
    elements.app.hidden = false;
    window.setTimeout(() => (elements.setupScreen.hidden = true), 350);
    const requestedView = window.location.hash.replace("#", "");
    switchView(["inspect", "classes", "signals", "api"].includes(requestedView) ? requestedView : "inspect");
    await selectDemoSample("connector-pass", true);
  } catch (error) {
    setServiceState(false, "Service unavailable");
    elements.setupMessage.textContent = error.message;
    elements.setupScreen.classList.add("failed");
  }
}

initialize();

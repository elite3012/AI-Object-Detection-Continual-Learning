const state = {
  model: null,
  examples: [],
  selectedExample: null,
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
  demoWarning: document.querySelector("#demo-warning"),
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
  correctionClass: document.querySelector("#correction-class"),
  reviewNote: document.querySelector("#review-note"),
  submitReview: document.querySelector("#submit-review"),
  classList: document.querySelector("#class-list"),
  metricGrid: document.querySelector("#metric-grid"),
  endpointList: document.querySelector("#endpoint-list"),
  toast: document.querySelector("#toast"),
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
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
  showToast.timer = window.setTimeout(() => (elements.toast.className = "toast"), 3400);
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
  window.location.hash = view;
}

function renderExamples() {
  elements.sampleList.innerHTML = state.examples
    .map(
      (sample) => `
        <button class="sample-item" data-sample-id="${sample.id}">
          <img src="${sample.image_url}" alt="${sample.title}" />
          <span>
            <strong>${sample.title}</strong>
            <small>${sample.subtitle}</small>
          </span>
          <i aria-hidden="true"></i>
        </button>`,
    )
    .join("");
  elements.sampleList.querySelectorAll(".sample-item").forEach((button) => {
    button.addEventListener("click", () => selectExample(button.dataset.sampleId));
  });
}

function renderClasses() {
  const classes = state.model.classes;
  elements.sidebarClassCount.textContent = `${classes.length} classes`;
  elements.correctionClass.innerHTML = [
    '<option value="">Không chắc / bỏ qua</option>',
    ...classes.map(
      (item) =>
        `<option value="${item.ip102_id}">${item.common_name_vi} - ${item.canonical_name}</option>`,
    ),
  ].join("");
  elements.classList.innerHTML = classes
    .map(
      (item) => `
        <div class="class-row pest-class-row">
          <span class="class-dot"></span>
          <div>
            <strong>${item.common_name_vi}</strong>
            <small>${item.canonical_name}</small>
          </div>
          <span>${item.stratum}</span>
          <code>${item.ip102_id}</code>
        </div>`,
    )
    .join("");
}

function renderEvidence() {
  const model = state.model.model;
  const dataset = state.model.dataset || {};
  const thresholds = state.model.thresholds;
  const cards = [
    ["Run", state.model.run_id || "unknown", "Model version"],
    ["Classes", state.model.classes.length, "Reviewed IP102 subset"],
    ["Params", model.parameter_count.toLocaleString("en-US"), model.name],
    ["Image", `${state.model.preprocessing.image_size}px`, "Center crop inference"],
    ["Accept", thresholds.accepted.toFixed(2), "Confidence threshold"],
    ["Manifest", shortHash(dataset.manifest_sha256), "Dataset fingerprint"],
  ];
  elements.metricGrid.innerHTML = cards
    .map(
      ([label, value, note]) =>
        `<div class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`,
    )
    .join("");
}

function renderEndpoints() {
  const endpoints = [
    ["GET", "/api/v1/health/ready", "Readiness includes model loading"],
    ["GET", "/api/v1/model", "Model card, class map, preprocessing"],
    ["GET", "/api/v1/examples", "Licensed sample metadata"],
    ["POST", "/api/v1/predictions", "Classify one uploaded image"],
    ["POST", "/api/v1/reviews", "Store offline human feedback"],
    ["GET", "/docs", "OpenAPI schema"],
  ];
  elements.endpointList.innerHTML = endpoints
    .map(
      ([method, path, description]) => `
        <div class="endpoint-row">
          <span class="method ${method.toLowerCase()}">${method}</span>
          <code>${path}</code>
          <p>${description}</p>
        </div>`,
    )
    .join("");
}

async function selectExample(exampleId, silent = false) {
  const sample = state.examples.find((item) => item.id === exampleId);
  if (!sample) return;
  state.selectedExample = sample;
  state.currentFile = null;
  elements.imageUpload.value = "";
  elements.predictUpload.disabled = true;
  elements.inspectionImage.src = `${sample.image_url}?t=${Date.now()}`;
  elements.inspectionImage.alt = sample.title;
  elements.inputSource.textContent = "Sample";
  document.querySelectorAll(".sample-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.sampleId === exampleId);
  });
  await predictExample(exampleId, silent);
}

async function predictExample(exampleId, silent) {
  setResultLoading();
  try {
    const result = await api(`/api/v1/examples/${exampleId}/predict?top_k=3`, {
      method: "POST",
    });
    renderPrediction(result);
    if (!silent) showToast("Prediction completed");
  } catch (error) {
    showResultError(error.message);
  }
}

async function predictUploadedImage() {
  if (!state.currentFile) return;
  const form = new FormData();
  form.append("file", state.currentFile);
  setResultLoading();
  try {
    const result = await api("/api/v1/predictions?top_k=3", {
      method: "POST",
      body: form,
    });
    renderPrediction(result);
  } catch (error) {
    showResultError(error.message);
  }
}

function setResultLoading() {
  elements.resultEmpty.hidden = false;
  elements.resultContent.hidden = true;
  elements.resultEmpty.innerHTML =
    '<div class="spinner"></div><strong>Đang chạy inference</strong><p>Chuẩn hóa ảnh và tính top-k...</p>';
  elements.resultLatency.textContent = "Running";
}

function renderPrediction(result) {
  state.prediction = result;
  const top = result.top_k[0];
  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = false;
  elements.resultLatency.textContent = `${Math.round(result.latency_ms)} ms`;
  elements.outcomeType.textContent = result.decision.toUpperCase();
  elements.outcomeLabel.textContent =
    result.decision === "accepted" ? top.common_name_vi : decisionLabel(result.decision);
  elements.outcomeIndicator.className = `outcome-indicator ${decisionTone(result.decision)}`;
  elements.matchList.innerHTML = result.top_k
    .map((match, index) => {
      const percent = Math.max(0, Math.min(100, match.confidence * 100));
      return `
        <div class="match-row">
          <div>
            <span>${index + 1}. ${match.common_name_vi}</span>
            <strong>${percent.toFixed(1)}%</strong>
          </div>
          <div class="match-track"><i style="width:${percent}%"></i></div>
          <small>${match.canonical_name} · ${match.stratum}</small>
        </div>`;
    })
    .join("");
  elements.thresholdNote.textContent = result.reason;
}

async function submitReview() {
  if (!state.prediction) return showToast("Chưa có prediction để review", "error");
  const top = state.prediction.top_k[0];
  const corrected = elements.correctionClass.value;
  try {
    const result = await api("/api/v1/reviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_id: state.prediction.prediction_id,
        decision: state.prediction.decision,
        predicted_class_id: top ? top.class_id : null,
        corrected_class_id: corrected ? Number(corrected) : null,
        note: elements.reviewNote.value.trim() || null,
        image_consent: false,
      }),
    });
    elements.reviewNote.value = "";
    elements.correctionClass.value = "";
    showToast(`Review #${result.review_id} saved`);
  } catch (error) {
    showToast(error.message, "error");
  }
}

function showResultError(message) {
  elements.resultContent.hidden = true;
  elements.resultEmpty.hidden = false;
  elements.resultEmpty.innerHTML = `<strong>Không thể predict</strong><p>${message}</p>`;
  elements.resultLatency.textContent = "Error";
  showToast(message, "error");
}

function decisionLabel(decision) {
  if (decision === "uncertain") return "Cần kiểm tra thêm";
  if (decision === "unsupported") return "Ngoài phạm vi hỗ trợ";
  return "Đã nhận diện";
}

function decisionTone(decision) {
  if (decision === "accepted") return "known";
  if (decision === "unsupported") return "unknown";
  return "uncertain";
}

function shortHash(value) {
  if (!value) return "n/a";
  return `${String(value).slice(0, 8)}...`;
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  elements.imageUpload.addEventListener("change", () => {
    const [file] = elements.imageUpload.files;
    state.currentFile = file || null;
    state.selectedExample = null;
    elements.predictUpload.disabled = !file;
    if (file) {
      elements.inspectionImage.src = URL.createObjectURL(file);
      elements.inspectionImage.alt = file.name;
      elements.inputSource.textContent = "Upload";
      document.querySelectorAll(".sample-item").forEach((button) => {
        button.classList.remove("active");
      });
    }
  });
  elements.predictUpload.addEventListener("click", predictUploadedImage);
  elements.submitReview.addEventListener("click", submitReview);
}

async function initialize() {
  bindEvents();
  renderEndpoints();
  try {
    elements.setupMessage.textContent = "Loading model bundle...";
    const ready = await api("/api/v1/health/ready");
    state.model = await api("/api/v1/model");
    const { examples } = await api("/api/v1/examples");
    state.examples = examples;

    setServiceState(true, "Ready");
    elements.modelName.textContent = ready.model_version;
    elements.demoReady.textContent = "Ready";
    elements.demoReady.classList.add("ready");
    if (state.model.demo_model) {
      elements.demoWarning.hidden = false;
      elements.demoWarning.textContent =
        "Demo model đang bật để smoke test UI/API. Train PestNet-S và mount bundle thật trước khi báo cáo metric.";
    }

    renderExamples();
    renderClasses();
    renderEvidence();
    elements.setupScreen.classList.add("done");
    elements.app.hidden = false;
    window.setTimeout(() => (elements.setupScreen.hidden = true), 350);
    const requestedView = window.location.hash.replace("#", "");
    switchView(["inspect", "classes", "evidence", "api"].includes(requestedView) ? requestedView : "inspect");
    if (state.examples.length) await selectExample(state.examples[0].id, true);
  } catch (error) {
    setServiceState(false, "Unavailable");
    elements.setupMessage.textContent = error.message;
    elements.setupScreen.classList.add("failed");
  }
}

initialize();

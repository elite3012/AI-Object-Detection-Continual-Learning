function initialLayerIndex() {
  const requested = new URLSearchParams(window.location.search).get("layer");
  const layerMap = {
    input: 0,
    stem: 1,
    res32: 2,
    res64: 3,
    res128: 4,
    attention: 5,
    pool: 6,
    gate: 7,
  };
  return layerMap[requested] ?? 0;
}

const state = {
  model: null,
  experiment: null,
  experimentError: null,
  examples: [],
  selectedExample: null,
  currentFile: null,
  prediction: null,
  stemFeatures: null,
  stemFeaturesError: null,
  residual32Features: null,
  residual32FeaturesError: null,
  residual64Features: null,
  residual64FeaturesError: null,
  residual128Features: null,
  residual128FeaturesError: null,
  attentionFeatures: null,
  attentionFeaturesError: null,
  poolFeatures: null,
  poolFeaturesError: null,
  gateFeatures: null,
  gateFeaturesError: null,
  activeLayerIndex: initialLayerIndex(),
  activeReproStep: "config",
  activeTrustLens: "use",
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
  runAudit: document.querySelector("#run-audit"),
  sampleAudit: document.querySelector("#sample-audit"),
  gateCurrent: document.querySelector("#gate-current"),
  gateCurrentMarker: document.querySelector("#gate-current-marker"),
  gateUncertainMarker: document.querySelector("#gate-uncertain-marker"),
  gateAcceptedMarker: document.querySelector("#gate-accepted-marker"),
  gateUncertainLabel: document.querySelector("#gate-uncertain-label"),
  gateAcceptedLabel: document.querySelector("#gate-accepted-label"),
  copyReport: document.querySelector("#copy-report"),
  downloadReport: document.querySelector("#download-report"),
  classList: document.querySelector("#class-list"),
  layerFlow: document.querySelector("#layer-flow"),
  perClassMetrics: document.querySelector("#per-class-metrics"),
  optimizationLab: document.querySelector("#optimization-lab"),
  trustSnapshot: document.querySelector("#trust-snapshot"),
  experimentEvidence: document.querySelector("#experiment-evidence"),
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

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return entities[char];
  });
}

function commonName(item) {
  return item.common_name_en || item.expected_name || item.title || item.dataset_label || "Unknown pest";
}

function displayCommonName(item) {
  return titleCase(commonName(item));
}

function imageKindLabel(item) {
  if (item?.image_kind === "photo") {
    return "Photo";
  }
  if (item?.image_kind === "dataset") {
    return "IP102 sample";
  }
  return "Reference";
}

function scientificName(item) {
  return item.canonical_name || item.subtitle || "Scientific name unavailable";
}

function layerSampleExample() {
  return (
    state.examples.find((item) => Number(item.class_id) === 87) ||
    state.examples.find((item) => commonName(item).toLowerCase() === "tobacco cutworm") ||
    state.examples.find((item) => item.image_kind === "dataset") ||
    state.examples.find((item) => item.image_kind === "photo") ||
    state.examples[0]
  );
}

function titleCase(value) {
  return String(value || "unknown")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function percent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(1)}%`;
}

function numberLabel(value) {
  if (value === null || value === undefined || value === "") return "n/a";
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  return number.toLocaleString("en-US");
}

function decimalLabel(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "n/a";
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  return number.toFixed(digits);
}

function dateLabel(value) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function metricTone(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "unknown";
  if (value < 0.4) return "critical";
  if (value < 0.55) return "watch";
  return "stable";
}

function classByIndexMap() {
  return new Map(state.model.classes.map((item) => [Number(item.index), item]));
}

function showToast(message, tone = "success") {
  elements.toast.textContent = message;
  elements.toast.className = `toast visible ${tone}`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => (elements.toast.className = "toast"), 3400);
}

function setServiceState(online, label) {
  elements.serviceStatus.className = `service-status ${online ? "online" : "offline"}`;
  elements.serviceStatus.innerHTML = `<i></i> ${escapeHtml(label)}`;
}

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("active", page.dataset.page === view);
  });
  window.location.hash = view;
  updateParallax();
}

function updateParallax() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const layers = document.querySelectorAll("[data-parallax-depth]");
  const y = window.scrollY || 0;
  layers.forEach((layer) => {
    const depth = Number(layer.dataset.parallaxDepth || 0);
    layer.style.transform = `translate3d(0, ${Math.round(y * depth)}px, 0)`;
  });
}

function renderExamples() {
  elements.sampleList.innerHTML = state.examples
    .map((sample) => {
      const name = displayCommonName(sample);
      return `
        <button class="sample-item" data-sample-id="${escapeHtml(sample.id)}">
          <img src="${escapeHtml(`${sample.image_url}?v=image-scale`)}" alt="${escapeHtml(name)}" />
          <span>
            <strong>${escapeHtml(name)}</strong>
            <small>${escapeHtml(scientificName(sample))}</small>
          </span>
          <em>${escapeHtml(imageKindLabel(sample))}</em>
          <i aria-hidden="true"></i>
        </button>`;
    })
    .join("");
  elements.sampleList.querySelectorAll(".sample-item").forEach((button) => {
    button.addEventListener("click", () => selectExample(button.dataset.sampleId));
  });
}

function renderClasses() {
  const classes = state.model.classes;
  const examplesByClass = new Map(state.examples.map((example) => [example.class_id, example]));
  const best = state.model.metrics?.best_validation || {};
  const decision = state.model.calibration?.decision_summary || {};
  const metricsByIndex = new Map((best.per_class || []).map((row) => [Number(row.index), row]));
  elements.sidebarClassCount.textContent = `${classes.length} classes`;
  elements.correctionClass.innerHTML = [
    '<option value="">Not sure / skip</option>',
    ...classes.map(
      (item) =>
        `<option value="${item.ip102_id}">${escapeHtml(displayCommonName(item))} - ${escapeHtml(scientificName(item))}</option>`,
    ),
  ].join("");
  elements.classList.innerHTML = classes
    .map(
      (item) => {
        const example = examplesByClass.get(Number(item.ip102_id));
        const imageUrl = example ? `${example.image_url}?v=image-scale` : "";
        const metric = metricsByIndex.get(Number(item.index));
        const tone = metricTone(metric?.f1);
        return `
        <div class="class-card pest-class-row ${tone}">
          <img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(displayCommonName(item))}" loading="lazy" />
          <em>${escapeHtml(imageKindLabel(example))}</em>
          <span class="class-dot"></span>
          <div>
            <strong>${escapeHtml(displayCommonName(item))}</strong>
            <small>${escapeHtml(scientificName(item))}</small>
          </div>
          <span>${escapeHtml(titleCase(item.stratum))}</span>
          <code>${item.ip102_id}</code>
          <div class="class-score">
            <span>F1</span>
            <strong>${percent(metric?.f1)}</strong>
            <small>P ${percent(metric?.precision)} / R ${percent(metric?.recall)}</small>
          </div>
        </div>`;
      },
    )
    .join("");
  const metricRows = renderPerClassMetrics(best.per_class || []);
  renderOptimizationPanel(metricRows, best, decision);
}

function renderModelLab() {
  renderLayerFlow();
}

function trustLensDefinitions() {
  return {
    use: {
      title: "Use with field review",
      note: "Accepted predictions inside this bundle are the app's strongest signal.",
      signals: ["Accepted", "Supported class", "Human check"],
    },
    review: {
      title: "Send to review",
      note: "Uncertain or messy images should be treated as a queue, not a final answer.",
      signals: ["Uncertain", "Blurred crop", "Low confidence"],
    },
    reject: {
      title: "Outside model scope",
      note: "Unsupported images can still get scores, so the gate matters.",
      signals: ["Unsupported", "Non-scope pest", "Do not act"],
    },
  };
}

function trustLensCardMarkup(activeLens) {
  return `
    <article class="trust-lens-card">
      <div>
        <strong>${escapeHtml(activeLens.title)}</strong>
        <small>${escapeHtml(activeLens.note)}</small>
      </div>
      <div class="trust-signal-list">
        ${activeLens.signals.map((signal) => `<span>${escapeHtml(signal)}</span>`).join("")}
      </div>
    </article>`;
}

function trustOverviewMarkup() {
  const model = state.model.model || {};
  const dataset = state.model.dataset || {};
  const best = state.model.metrics?.best_validation || {};
  const decision = state.model.calibration?.decision_summary || {};
  const selection = state.model.calibration?.selection || {};
  const training = state.model.training || {};
  const thresholds = state.model.thresholds || {};
  const classCount = state.model.classes.length;
  const lenses = trustLensDefinitions();
  const activeLens = lenses[state.activeTrustLens] || lenses.use;
  const acceptedPct = Math.max(0, Math.min(100, Number(thresholds.accepted || 0) * 100));
  const uncertainPct = Math.max(0, Math.min(acceptedPct, Number(thresholds.uncertain || 0) * 100));
  return `
    <section class="evidence-section trust-merged-section">
      <div class="evidence-section-head">
        <span>TRUST CHECK</span>
        <h3>Field-use checklist</h3>
      </div>
      <div class="trust-quick-grid">
        <article class="trust-quick-card">
          <span>Scope</span>
          <strong>${classCount}</strong>
          <small>classes</small>
        </article>
        <article class="trust-quick-card">
          <span>Macro-F1</span>
          <strong>${percent(best.macro_f1)}</strong>
          <small>validation</small>
        </article>
        <article class="trust-quick-card">
          <span>Gate precision</span>
          <strong>${percent(decision.id_accepted_precision)}</strong>
          <small>${percent(decision.id_coverage)} coverage</small>
        </article>
        <article class="trust-quick-card">
          <span>Run</span>
          <strong>${Number(training.epochs_run || 0)}</strong>
          <small>${escapeHtml(titleCase(training.class_strategy || "weighted_loss"))}</small>
        </article>
      </div>
      <div class="trust-lens-workbench">
        <div class="trust-lens-buttons" role="tablist" aria-label="Trust decision lens">
          ${Object.entries(lenses)
            .map(
              ([id, item]) => `
                <button
                  class="trust-lens-button ${id === state.activeTrustLens ? "active" : ""}"
                  type="button"
                  role="tab"
                  aria-selected="${id === state.activeTrustLens ? "true" : "false"}"
                  data-trust-lens="${escapeHtml(id)}"
                >
                  ${escapeHtml(titleCase(id))}
                </button>`,
            )
            .join("")}
        </div>
        ${trustLensCardMarkup(activeLens)}
        <div class="trust-gate-visual" style="--accepted:${acceptedPct}%; --uncertain:${uncertainPct}%">
          <div class="trust-gate-head">
            <span>Decision gate</span>
            <strong>${Number(thresholds.accepted || 0).toFixed(3)}</strong>
          </div>
          <div class="trust-gate-track">
            <i></i>
            <b class="uncertain"></b>
            <b class="accepted"></b>
          </div>
          <div class="trust-gate-labels">
            <span>Unsupported</span>
            <span>Uncertain</span>
            <span>Accepted</span>
          </div>
        </div>
      </div>
      <div class="trust-data-strip">
        <span>Manifest ${escapeHtml(shortHash(dataset.manifest_sha256))}</span>
        <span>${Number(dataset.train_records || 0).toLocaleString("en-US")} train</span>
        <span>${Number(dataset.val_records || 0).toLocaleString("en-US")} val</span>
        <span>Target ${percent(selection.target_accept_precision)}</span>
      </div>
    </section>`;
}

function renderTrustSnapshot() {
  if (elements.trustSnapshot) {
    elements.trustSnapshot.innerHTML = trustOverviewMarkup();
  }
}

function curveDomain(rows, keys, fallbackMin = 0, fallbackMax = 1) {
  const values = rows.flatMap((row) =>
    keys.map((key) => Number(row[key])).filter((value) => Number.isFinite(value)),
  );
  if (!values.length) return [fallbackMin, fallbackMax];
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 0.05;
    max += 0.05;
  }
  const pad = (max - min) * 0.08;
  return [Math.max(0, min - pad), max + pad];
}

function curvePath(rows, key, domain, width, height, pad) {
  const values = rows.map((row) => Number(row[key]));
  const [min, max] = domain;
  const span = max - min || 1;
  return values
    .map((value, index) => {
      const x = pad + (rows.length <= 1 ? 0 : (index / (rows.length - 1)) * (width - pad * 2));
      const y = pad + (1 - (Number(value) - min) / span) * (height - pad * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function renderCurveChart(rows, title, subtitle, series, domain, valueFormatter) {
  if (!rows.length) {
    return `<article class="curve-card empty"><strong>${escapeHtml(title)}</strong><p>No training history is recorded for this bundle.</p></article>`;
  }
  const width = 640;
  const height = 230;
  const pad = 28;
  const startEpoch = rows[0]?.epoch ?? 1;
  const endEpoch = rows[rows.length - 1]?.epoch ?? rows.length;
  const latest = rows[rows.length - 1] || {};
  return `
    <article class="curve-card">
      <div class="curve-head">
        <div>
          <span>${escapeHtml(subtitle)}</span>
          <strong>${escapeHtml(title)}</strong>
        </div>
        <small>Epoch ${escapeHtml(startEpoch)}-${escapeHtml(endEpoch)}</small>
      </div>
      <svg class="curve-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${escapeHtml(title)} curve">
        <path class="curve-grid" d="M ${pad} ${pad} H ${width - pad} M ${pad} ${height / 2} H ${width - pad} M ${pad} ${height - pad} H ${width - pad}" />
        ${series
          .map(
            (item) =>
              `<path class="curve-line ${escapeHtml(item.tone)}" pathLength="1" d="${curvePath(rows, item.key, domain, width, height, pad)}" />`,
          )
          .join("")}
      </svg>
      <div class="curve-legend">
        ${series
          .map((item) => {
            const value = latest[item.key];
            return `<span class="${escapeHtml(item.tone)}"><i></i>${escapeHtml(item.label)} <strong>${escapeHtml(valueFormatter(value))}</strong></span>`;
          })
          .join("")}
      </div>
    </article>`;
}

function experimentCard(label, value, note, tone = "") {
  return `
    <article class="experiment-stat ${escapeHtml(tone)}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </article>`;
}

function renderExperimentSummary(experiment) {
  const run = experiment.run || {};
  const split = experiment.split || {};
  const best = experiment.best_validation || {};
  const training = experiment.training || {};
  return `
    <div class="experiment-stats">
      ${experimentCard("Checkpoint", run.checkpoint_sha256_short || "n/a", `${run.checkpoint_file || "model.pt"} / ${run.run_id || "unknown run"}`, "checkpoint")}
      ${experimentCard("Split", `${numberLabel(split.train_records)} train`, `${numberLabel(split.val_records)} validation samples`, "split")}
      ${experimentCard("Best validation", percent(best.macro_f1), `Top-1 ${percent(best.top1_accuracy)} / Top-3 ${percent(best.top3_accuracy)}`, "metric")}
      ${experimentCard("Training run", `${numberLabel(training.epochs_run)} epochs`, `${titleCase(training.class_strategy || "weighted_loss")} on ${training.device || "unknown device"}`, "run")}
    </div>`;
}

function renderTrainingCurves(experiment) {
  const rows = experiment.curves || [];
  const scoreDomain = [0, 1];
  const lossDomain = curveDomain(rows, ["train_loss", "val_loss"], 0, 2);
  return `
    <section class="evidence-section curves-section">
      <div class="evidence-section-head">
        <span>TRAINING CURVES</span>
        <h3>Learning progress</h3>
      </div>
      <div class="curve-grid">
        ${renderCurveChart(
          rows,
          "Macro-F1",
          "class-balanced signal",
          [
            { key: "train_macro_f1", label: "Train", tone: "train" },
            { key: "val_macro_f1", label: "Validation", tone: "validation" },
          ],
          scoreDomain,
          percent,
        )}
        ${renderCurveChart(
          rows,
          "Cross-entropy loss",
          "optimization trace",
          [
            { key: "train_loss", label: "Train", tone: "train" },
            { key: "val_loss", label: "Validation", tone: "validation" },
          ],
          lossDomain,
          (value) => decimalLabel(value, 3),
        )}
      </div>
    </section>`;
}

function renderClassDistribution(experiment) {
  const rows = [...(experiment.class_distribution || [])].sort((a, b) => Number(b.total || 0) - Number(a.total || 0));
  if (!rows.length) {
    return `<section class="evidence-section"><div class="empty-evidence">Class distribution is not available for this bundle.</div></section>`;
  }
  return `
    <section class="evidence-section distribution-section">
      <div class="evidence-section-head">
        <span>SPLIT STRATEGY</span>
        <h3>Selected classes and split counts</h3>
      </div>
      <div class="split-note">
        <strong>${escapeHtml(experiment.split?.manifest_sha256_short || "manifest")}</strong>
        <span>${escapeHtml(experiment.split?.strategy || "Official split loaded from the bundle manifest.")}</span>
      </div>
      <div class="distribution-list">
        ${rows
          .map((row, index) => {
            const total = Number(row.total || 0) || 1;
            const trainPct = (Number(row.train || 0) / total) * 100;
            const valPct = (Number(row.val || 0) / total) * 100;
            const testPct = (Number(row.test || 0) / total) * 100;
            const valEndPct = trainPct + valPct;
            return `
              <div class="distribution-row" style="--delay:${index * 34}ms; --train:${trainPct}%; --val:${valPct}%; --test:${testPct}%; --val-end:${valEndPct}%">
                <div class="distribution-name">
                  <strong>${escapeHtml(titleCase(row.common_name_en || row.dataset_label || "Unknown class"))}</strong>
                  <small>${escapeHtml(row.scientific_name || row.dataset_label || "")}</small>
                </div>
                <div class="distribution-track" aria-label="Train ${numberLabel(row.train)}, validation ${numberLabel(row.val)}, test ${numberLabel(row.test)}">
                  <i class="train"></i><i class="val"></i><i class="test"></i>
                </div>
                <div class="distribution-counts">
                  <span>${numberLabel(row.train)} train</span>
                  <span>${numberLabel(row.val)} val</span>
                  <span>${numberLabel(row.test)} test</span>
                </div>
              </div>`;
          })
          .join("")}
      </div>
    </section>`;
}

function renderConfusionMatrix(experiment) {
  const confusion = experiment.confusion || {};
  const labels = confusion.labels || [];
  const matrix = confusion.matrix || [];
  if (!matrix.length) {
    return `<section class="evidence-section"><div class="empty-evidence">Confusion matrix is not recorded for this bundle.</div></section>`;
  }
  const maxValue = Math.max(...matrix.flat().map((value) => Number(value) || 0), 1);
  const cells = matrix
    .flatMap((row, actualIndex) =>
      row.map((count, predictedIndex) => {
        const value = Number(count) || 0;
        const actual = labels[actualIndex]?.name || `Class ${actualIndex}`;
        const predicted = labels[predictedIndex]?.name || `Class ${predictedIndex}`;
        const heat = Math.max(0.06, value / maxValue);
        const delay = Math.min(520, (actualIndex * matrix.length + predictedIndex) * 8);
        return `<span class="matrix-cell ${actualIndex === predictedIndex ? "diagonal" : "off-diagonal"}" style="--heat:${heat}; --delay:${delay}ms" title="Actual ${escapeHtml(titleCase(actual))}; predicted ${escapeHtml(titleCase(predicted))}: ${value}"></span>`;
      }),
    )
    .join("");
  return `
    <section class="evidence-section confusion-section">
      <div class="evidence-section-head">
        <span>ERROR ANALYSIS</span>
        <h3>Confusion matrix</h3>
      </div>
      <div class="matrix-layout">
        <div class="matrix-canvas-wrap">
          <div class="matrix-axis top">Predicted class</div>
          <div class="matrix-axis left">Actual class</div>
          <div class="matrix-canvas" style="--matrix-size:${matrix.length}">${cells}</div>
        </div>
        <div class="matrix-notes">
          <div class="matrix-legend">
            <span><i class="diag"></i>Correct</span>
            <span><i class="miss"></i>Confused</span>
          </div>
          <strong>Largest misses</strong>
          ${(confusion.top_pairs || [])
            .map(
              (pair) => `
                <p>
                  <span>${escapeHtml(titleCase(pair.actual))}</span>
                  <b>${escapeHtml(titleCase(pair.predicted))}</b>
                  <em>${numberLabel(pair.count)}</em>
                </p>`,
            )
            .join("")}
        </div>
      </div>
    </section>`;
}

function renderFailureAnalysis(experiment) {
  const failure = experiment.failure_analysis || {};
  const weakest = failure.weakest_classes || [];
  const drivers = failure.confusion_drivers || [];
  const hardCases = failure.hard_cases || [];
  const causes = failure.root_causes || [];
  const steps = failure.improvement_steps || [];
  if (!weakest.length && !drivers.length && !hardCases.length) {
    return `<section class="evidence-section failure-section"><div class="empty-evidence">Failure analysis is not available for this bundle.</div></section>`;
  }
  return `
    <section class="evidence-section failure-section">
      <div class="evidence-section-head">
        <span>FAILURE ANALYSIS</span>
        <h3>Where the model fails and what to fix</h3>
      </div>
      <div class="failure-summary-strip">
        <article>
          <span>Weakest class</span>
          <strong>${escapeHtml(failure.summary?.primary_failure || "n/a")}</strong>
        </article>
        <article>
          <span>Largest confusion</span>
          <strong>${escapeHtml(failure.summary?.largest_confusion || "n/a")}</strong>
        </article>
        <article>
          <span>Image-level cases</span>
          <strong>${numberLabel(failure.summary?.hard_case_count || hardCases.length)}</strong>
        </article>
      </div>
      <div class="failure-grid">
        <div class="failure-column">
          <div class="failure-block-head"><span>WHERE IT BREAKS</span><strong>Weak classes</strong></div>
          <div class="weak-class-list">
            ${weakest
              .map(
                (item, index) => `
                  <article class="weak-class-card" style="--delay:${index * 70}ms">
                    <div>
                      <strong>${escapeHtml(titleCase(item.name))}</strong>
                      <small>${escapeHtml(item.scientific_name || "")}</small>
                    </div>
                    <span class="weak-score">${percent(item.f1)} F1</span>
                    <div class="weak-bars">
                      <span><i style="--value:${Math.max(0, Math.min(1, Number(item.precision || 0)))}"></i><b>P ${percent(item.precision)}</b></span>
                      <span><i style="--value:${Math.max(0, Math.min(1, Number(item.recall || 0)))}"></i><b>R ${percent(item.recall)}</b></span>
                    </div>
                    <p>${escapeHtml(item.why)}</p>
                    <em>${item.most_confused_with ? `Often confused with ${escapeHtml(titleCase(item.most_confused_with))} (${numberLabel(item.confused_count)} cases)` : "No dominant confusion pair"}</em>
                  </article>`,
              )
              .join("")}
          </div>
        </div>
        <div class="failure-column">
          <div class="failure-block-head"><span>WHY IT FAILS</span><strong>Top confusion drivers</strong></div>
          <div class="confusion-driver-list">
            ${drivers
              .slice(0, 5)
              .map(
                (item, index) => `
                  <article class="confusion-driver" style="--delay:${index * 65}ms">
                    <div>
                      <strong>${escapeHtml(titleCase(item.actual))}</strong>
                      <span>to</span>
                      <strong>${escapeHtml(titleCase(item.predicted))}</strong>
                    </div>
                    <em>${numberLabel(item.count)} images / ${percent(item.share_of_actual)}</em>
                    <p>${escapeHtml(item.why)}</p>
                  </article>`,
              )
              .join("")}
          </div>
        </div>
      </div>
      <div class="hard-case-section">
        <div class="failure-block-head"><span>HARD IMAGES</span><strong>External cases the model ranked incorrectly</strong></div>
        <div class="hard-case-grid">
          ${hardCases.length
            ? hardCases
                .map(
                  (item, index) => `
                    <article class="hard-case-card" style="--delay:${index * 80}ms">
                      <img src="${escapeHtml(`${item.image_url}?v=failure-analysis`)}" alt="${escapeHtml(titleCase(item.actual))}" loading="lazy" />
                      <div>
                        <span>${escapeHtml(titleCase(item.decision || "review"))} / ${percent(item.confidence)}</span>
                        <strong>${escapeHtml(titleCase(item.actual))}</strong>
                        <small>Predicted ${escapeHtml(titleCase(item.predicted))}</small>
                      </div>
                      <p>${escapeHtml(item.why)}</p>
                      <em>${escapeHtml(item.provider || "External benchmark")} · ${escapeHtml(item.license || "license recorded")}</em>
                    </article>`,
                )
                .join("")
            : `<div class="empty-evidence">No external hard-case images are available locally. Aggregate confusion still shows the failure patterns above.</div>`}
        </div>
      </div>
      <div class="failure-actions compact">
        <details class="failure-disclosure">
          <summary>
            <span>ROOT CAUSES</span>
            <strong>Why it breaks</strong>
            <em>${numberLabel(causes.length)} notes</em>
          </summary>
          <div class="failure-disclosure-body">
            ${causes
              .map(
                (item) => `
                  <article>
                    <strong>${escapeHtml(item.title)}</strong>
                    <p>${escapeHtml(item.evidence)}</p>
                    <em>${escapeHtml(item.action)}</em>
                  </article>`,
              )
              .join("")}
          </div>
        </details>
        <details class="failure-disclosure">
          <summary>
            <span>NEXT FIXES</span>
            <strong>Improvement plan</strong>
            <em>${numberLabel(steps.length)} steps</em>
          </summary>
          <div class="failure-disclosure-body">
            <ol>
              ${steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}
            </ol>
            <small>${escapeHtml(failure.summary?.sample_level_note || "")}</small>
          </div>
        </details>
      </div>
    </section>`;
}

function reproSteps(repro) {
  const summary = repro.config_summary || {};
  const artifacts = repro.artifacts || [];
  const artifactFor = (labels) => artifacts.filter((item) => labels.includes(item.label));
  return [
    {
      id: "config",
      index: "01",
      label: "Config",
      title: "Lock the run",
      note: "One YAML, fixed seed, selected classes.",
      command: "",
      action: "Ready",
      facts: [
        ["Seed", numberLabel(repro.seed)],
        ["Config", repro.config_path || "n/a"],
        ["Classes", numberLabel(summary.selected_classes)],
        ["Image", summary.image_size ? `${numberLabel(summary.image_size)} px` : "n/a"],
      ],
      artifacts: artifactFor(["Training config", "Dataset manifest"]),
    },
    {
      id: "train",
      index: "02",
      label: "Train",
      title: "Create the bundle",
      note: "Train CNN and write output.",
      command: repro.commands?.train || "",
      action: "Copy train",
      facts: [
        ["Epochs", numberLabel(summary.epochs)],
        ["Batch", numberLabel(summary.batch_size)],
        ["LR", decimalLabel(summary.learning_rate, 4)],
        ["Strategy", titleCase(summary.class_strategy || "n/a")],
      ],
      artifacts: artifactFor(["Run folder", "History CSV", "Model bundle", "Checkpoint weights", "Bundle metadata"]),
    },
    {
      id: "evaluate",
      index: "03",
      label: "Evaluate",
      title: "Score validation",
      note: "Validate and update thresholds.",
      command: repro.commands?.evaluate || "",
      action: "Copy eval",
      facts: [
        ["Split", "Val"],
        ["Output", "Eval JSON"],
        ["Artifacts", numberLabel(artifacts.length)],
        ["Status", artifacts.every((item) => item.exists) ? "Complete" : "Check files"],
      ],
      artifacts: artifactFor(["Validation metrics", "Evaluation report", "Bundle metadata"]),
    },
  ];
}

function reproActiveCardMarkup(active, repro) {
  return `
    <article class="repro-active-card ${escapeHtml(active.id)}">
      <div class="repro-active-head">
        <div>
          <span>${escapeHtml(active.index)} / 03</span>
          <strong>${escapeHtml(active.title)}</strong>
        </div>
        ${
          active.command
            ? `<button class="button secondary compact copy-repro-command" type="button" data-command="${escapeHtml(active.command)}">${escapeHtml(active.action)}</button>`
            : `<span class="repro-status-chip">${repro.config_exists ? "Config found" : "Config missing"}</span>`
        }
      </div>
      <div class="repro-facts">
        ${active.facts
          .map(
            ([label, value]) => `
              <article>
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(value)}</strong>
              </article>`,
          )
          .join("")}
      </div>
      ${
        active.command
          ? `<code class="repro-command-line">${escapeHtml(active.command)}</code>`
          : `<div class="repro-config-visual" aria-hidden="true">
              <span>YAML</span><i></i><span>Seed</span><i></i><span>Manifest</span><i></i><span>Classes</span>
            </div>`
      }
      <div class="artifact-checklist compact">
        <div class="failure-block-head"><span>FILES</span><strong>Click a file to inspect</strong></div>
        ${active.artifacts
          .map(
            (item, index) => `
              <button class="artifact-row ${item.exists ? "exists" : "missing"}" type="button" style="--delay:${index * 35}ms">
                <i aria-hidden="true"></i>
                <div>
                  <strong>${escapeHtml(item.label)}</strong>
                  <code>${escapeHtml(item.path)}</code>
                  <small>${escapeHtml(item.description || "")}</small>
                </div>
                <em>${item.exists ? "Found" : "Missing"}</em>
              </button>`,
          )
          .join("")}
      </div>
    </article>`;
}

function renderReproducibility(experiment) {
  const repro = experiment.reproducibility || {};
  const artifacts = repro.artifacts || [];
  if (!repro.commands && !artifacts.length) {
    return `<section class="evidence-section reproducibility-section"><div class="empty-evidence">Reproducibility details are not recorded for this bundle.</div></section>`;
  }
  const steps = reproSteps(repro);
  const active = steps.find((step) => step.id === state.activeReproStep) || steps[0];
  return `
    <section class="evidence-section reproducibility-section">
      <div class="evidence-section-head">
        <span>REPRODUCIBILITY</span>
        <h3>Run it again</h3>
      </div>
      <div class="repro-workbench">
        <div class="repro-stepper" role="tablist" aria-label="Reproducibility workflow">
          ${steps
            .map(
              (step) => `
                <button
                  class="repro-step ${step.id === active.id ? "active" : ""}"
                  type="button"
                  role="tab"
                  aria-selected="${step.id === active.id ? "true" : "false"}"
                  data-repro-step="${escapeHtml(step.id)}"
                >
                  <span>${escapeHtml(step.index)}</span>
                  <strong>${escapeHtml(step.label)}</strong>
                  <small>${escapeHtml(step.note)}</small>
                </button>`,
            )
            .join("")}
        </div>
        ${reproActiveCardMarkup(active, repro)}
      </div>
    </section>`;
}

function updateTrustLens(lensId) {
  const lenses = trustLensDefinitions();
  state.activeTrustLens = lenses[lensId] ? lensId : "use";
  elements.experimentEvidence.querySelectorAll(".trust-lens-button").forEach((button) => {
    const active = button.dataset.trustLens === state.activeTrustLens;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  const card = elements.experimentEvidence.querySelector(".trust-lens-card");
  if (card) {
    card.outerHTML = trustLensCardMarkup(lenses[state.activeTrustLens]);
  }
}

function updateReproStep(stepId) {
  const repro = state.experiment?.reproducibility || {};
  const steps = reproSteps(repro);
  const activeStep = steps.some((step) => step.id === stepId) ? stepId : "config";
  state.activeReproStep = activeStep;
  elements.experimentEvidence.querySelectorAll(".repro-step").forEach((button) => {
    const active = button.dataset.reproStep === state.activeReproStep;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  const active = steps.find((step) => step.id === state.activeReproStep) || steps[0];
  const card = elements.experimentEvidence.querySelector(".repro-active-card");
  if (card && active) {
    card.outerHTML = reproActiveCardMarkup(active, repro);
  }
}

async function copyReproCommand(button) {
  const command = button.dataset.command || "";
  if (!command) return;
  try {
    await navigator.clipboard.writeText(command);
    showToast(`${button.closest(".repro-active-card")?.classList.contains("train") ? "Train" : "Eval"} command copied`);
  } catch {
    showToast("Clipboard is unavailable in this browser", "error");
  }
}

function bindReproducibilityActions() {
  if (!elements.experimentEvidence || elements.experimentEvidence.dataset.actionsBound === "true") return;
  elements.experimentEvidence.dataset.actionsBound = "true";
  elements.experimentEvidence.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (!target) return;
    const copyButton = target.closest(".copy-repro-command");
    if (copyButton && elements.experimentEvidence.contains(copyButton)) {
      void copyReproCommand(copyButton);
      return;
    }
    const trustButton = target.closest(".trust-lens-button");
    if (trustButton && elements.experimentEvidence.contains(trustButton)) {
      updateTrustLens(trustButton.dataset.trustLens || "use");
      return;
    }
    const stepButton = target.closest(".repro-step");
    if (stepButton && elements.experimentEvidence.contains(stepButton)) {
      updateReproStep(stepButton.dataset.reproStep || "config");
      return;
    }
    const artifactButton = target.closest(".artifact-row");
    if (artifactButton && elements.experimentEvidence.contains(artifactButton)) {
      artifactButton.classList.toggle("open");
    }
  });
}

function kv(label, value, note = "") {
  return `<div class="evidence-kv"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>${note ? `<small>${escapeHtml(note)}</small>` : ""}</div>`;
}

function renderExperimentConfig(experiment) {
  const training = experiment.training || {};
  const augmentation = experiment.augmentation || {};
  const run = experiment.run || {};
  const split = experiment.split || {};
  const full = split.full_dataset || {};
  const cropScale = Array.isArray(augmentation.crop_scale) ? augmentation.crop_scale.join(" - ") : augmentation.crop_scale;
  return `
    <section class="evidence-section config-section">
      <div class="evidence-section-head">
        <span>RUN CONFIG</span>
        <h3>Hyperparameters and checkpoint</h3>
      </div>
      <div class="config-grid">
        <dl>
          ${kv("Batch size", numberLabel(training.batch_size))}
          ${kv("Learning rate", decimalLabel(training.learning_rate, 4))}
          ${kv("Weight decay", decimalLabel(training.weight_decay, 4))}
          ${kv("Loss", titleCase(training.loss || "cross_entropy"))}
          ${kv("Scheduler", titleCase(training.scheduler || "none"))}
          ${kv("Seed", numberLabel(training.seed))}
        </dl>
        <dl>
          ${kv("Random crop scale", String(cropScale || "n/a"))}
          ${kv("Horizontal flip", percent(augmentation.hflip_probability))}
          ${kv("Rotation", `${decimalLabel(augmentation.rotation_degrees, 1)} deg`)}
          ${kv("Color jitter", decimalLabel(augmentation.color_jitter, 2))}
          ${kv("Augmentation source", augmentation.source || "bundle metadata")}
        </dl>
        <dl>
          ${kv("Run created", dateLabel(run.created_at))}
          ${kv("Git commit", run.git_commit_short || "n/a", run.git_dirty ? "working tree had local changes" : "")}
          ${kv("Metrics file", run.metrics_file || "n/a")}
          ${kv("Full IP102 audit", `${numberLabel(full.records)} images / ${numberLabel(full.class_count)} classes`)}
          ${kv("Near split duplicates", numberLabel(full.near_cross_split_pairs))}
        </dl>
      </div>
    </section>`;
}

function renderExperimentEvidence() {
  if (!elements.experimentEvidence) return;
  if (state.experimentError) {
    elements.experimentEvidence.innerHTML = `<div class="empty-evidence error"><strong>Experiment evidence unavailable</strong><p>${escapeHtml(state.experimentError)}</p></div>`;
    return;
  }
  const experiment = state.experiment;
  if (!experiment) {
    elements.experimentEvidence.innerHTML = '<div class="empty-evidence">Loading experiment evidence...</div>';
    return;
  }
  elements.experimentEvidence.innerHTML = `
    ${trustOverviewMarkup()}
    ${renderExperimentSummary(experiment)}
    ${renderTrainingCurves(experiment)}
    <div class="evidence-two-column">
      ${renderConfusionMatrix(experiment)}
      ${renderClassDistribution(experiment)}
    </div>
    ${renderFailureAnalysis(experiment)}
    ${renderReproducibility(experiment)}
  `;
  bindReproducibilityActions();
}

async function loadStemFeatures() {
  const sample = layerSampleExample();
  if (!sample) return;
  try {
    state.stemFeatures = await api(`/api/v1/examples/${sample.id}/stem-activations?channel_count=4`);
    state.stemFeaturesError = null;
  } catch (error) {
    state.stemFeatures = null;
    state.stemFeaturesError = error.message;
  }
}

async function loadResidual32Features() {
  const sample = layerSampleExample();
  if (!sample) return;
  try {
    state.residual32Features = await api(`/api/v1/examples/${sample.id}/residual32-activations?channel_count=3`);
    state.residual32FeaturesError = null;
  } catch (error) {
    state.residual32Features = null;
    state.residual32FeaturesError = error.message;
  }
}

async function loadResidual64Features() {
  const sample = layerSampleExample();
  if (!sample) return;
  try {
    state.residual64Features = await api(`/api/v1/examples/${sample.id}/residual64-activations?channel_count=3`);
    state.residual64FeaturesError = null;
  } catch (error) {
    state.residual64Features = null;
    state.residual64FeaturesError = error.message;
  }
}

async function loadResidual128Features() {
  const sample = layerSampleExample();
  if (!sample) return;
  try {
    state.residual128Features = await api(`/api/v1/examples/${sample.id}/residual128-activations?channel_count=3`);
    state.residual128FeaturesError = null;
  } catch (error) {
    state.residual128Features = null;
    state.residual128FeaturesError = error.message;
  }
}

async function loadAttentionFeatures() {
  const sample = layerSampleExample();
  if (!sample) return;
  try {
    state.attentionFeatures = await api(`/api/v1/examples/${sample.id}/attention-activations?channel_count=4`);
    state.attentionFeaturesError = null;
  } catch (error) {
    state.attentionFeatures = null;
    state.attentionFeaturesError = error.message;
  }
}

async function loadPoolFeatures() {
  const sample = layerSampleExample();
  if (!sample) return;
  try {
    state.poolFeatures = await api(`/api/v1/examples/${sample.id}/global-pool-activations?channel_count=5`);
    state.poolFeaturesError = null;
  } catch (error) {
    state.poolFeatures = null;
    state.poolFeaturesError = error.message;
  }
}

async function loadGateFeatures() {
  const sample = layerSampleExample();
  if (!sample) return;
  try {
    state.gateFeatures = await api(`/api/v1/examples/${sample.id}/decision-gate?top_k=5`);
    state.gateFeaturesError = null;
  } catch (error) {
    state.gateFeatures = null;
    state.gateFeaturesError = error.message;
  }
}

function layerVisualMarkup(visual) {
  if (visual === "input") {
    const sample = layerSampleExample();
    const sampleImage = sample ? `${sample.image_url}?v=input-workflow` : "";
    const sampleName = sample ? displayCommonName(sample) : "Field Sample";
    const imageTag = sampleImage
      ? `<img src="${escapeHtml(sampleImage)}" alt="" loading="lazy" />`
      : '<span class="input-real-fallback"></span>';
    return `
    <span class="input-workflow">
      <span class="input-panel input-real-panel">
        <b>Field photo</b>
        <span class="input-real-image">
          ${imageTag}
          <i class="input-crop-frame"></i>
          <em>${escapeHtml(sampleName)}</em>
        </span>
      </span>
      <span class="input-stage-arrow"><strong>crop</strong><i></i></span>
      <span class="input-panel input-crop-panel">
        <b>224x224 crop</b>
        <span class="input-square-image">
          ${imageTag}
          <i></i>
        </span>
        <em>whole insect kept</em>
      </span>
      <span class="input-stage-arrow"><strong>RGB</strong><i></i></span>
      <span class="input-panel input-normalize-panel">
        <b>Normalize tensor</b>
        <span class="input-channel channel-r"><em>R</em><i></i><strong>&mu; 0.485</strong><small>&sigma; 0.229</small></span>
        <span class="input-channel channel-g"><em>G</em><i></i><strong>&mu; 0.456</strong><small>&sigma; 0.224</small></span>
        <span class="input-channel channel-b"><em>B</em><i></i><strong>&mu; 0.406</strong><small>&sigma; 0.225</small></span>
        <span class="input-tensor-facts"><strong>3 x 224 x 224</strong><small>(pixel - mean) / std</small></span>
      </span>
    </span>`;
  }
  if (visual === "stem") {
    const sample = layerSampleExample();
    const sampleImage = sample ? `${sample.image_url}?v=stem-workflow` : "";
    const imageTag = sampleImage
      ? `<img src="${escapeHtml(sampleImage)}" alt="" loading="lazy" />`
      : '<span class="input-real-fallback"></span>';
    const features = state.stemFeatures;
    const outputShape = features?.output_shape?.join(" x ") || "32 x 112 x 112";
    const channelCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="activation-card">
                <img src="${escapeHtml(channel.image)}" alt="" loading="lazy" />
                <b>filter ${String(channel.index).padStart(2, "0")}</b>
                <small>response ${Number(channel.energy).toFixed(3)}</small>
              </span>`,
          )
          .join("")
      : `<span class="activation-placeholder">${escapeHtml(
          state.stemFeaturesError || "Loading real stem activations",
        )}</span>`;
    return `
      <span class="stem-workflow">
        <span class="stem-panel stem-input-panel">
          <b>Local patch</b>
          <span class="stem-photo">
            ${imageTag}
            <i class="stem-filter-window"></i>
            <i class="stem-scan-sweep"></i>
            <span class="stem-patch-chip">3x3x3 patch</span>
          </span>
          <em>RGB patch sampled by the first convolution</em>
        </span>
        <span class="stem-stage-arrow"><strong>conv</strong><i></i></span>
        <span class="stem-panel stem-kernel-panel">
          <b>32 learned filters</b>
          <span class="stem-kernel-grid" aria-hidden="true">
            <i data-weight="w1"></i><i data-weight="w2"></i><i data-weight="w3"></i>
            <i data-weight="w4"></i><i data-weight="w5"></i><i data-weight="w6"></i>
            <i data-weight="w7"></i><i data-weight="w8"></i><i data-weight="w9"></i>
          </span>
          <em>weights are learned, then normalized by BN and SiLU</em>
        </span>
        <span class="stem-stage-arrow"><strong>32 maps</strong><i></i></span>
        <span class="stem-panel stem-map-panel">
          <b>Feature map</b>
          <span class="stem-edge-map stem-activation-view">
            <span class="activation-title">
              <strong>Real stem activations</strong>
              <small>${escapeHtml(outputShape)}</small>
            </span>
            <span class="activation-legend">
              <span class="activation-scale"><span class="activation-gradient"></span></span>
              <span class="activation-scale-labels">
                <small>weak response</small>
                <strong>pattern found</strong>
              </span>
            </span>
            <span class="activation-grid">
              ${channelCards}
            </span>
          </span>
          <em>actual tensors returned by Conv-BN-SiLU for this sample</em>
        </span>
      </span>`;
  }
  if (visual === "res32") {
    const features = state.residual32Features;
    const inputShape = features?.input_shape?.join(" x ") || "32 x 112 x 112";
    const outputShape = features?.output_shape?.join(" x ") || "32 x 112 x 112";
    const beforeCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="res32-map-card">
                <img src="${escapeHtml(channel.before_image)}" alt="" loading="lazy" />
                <strong>filter ${String(channel.index).padStart(2, "0")}</strong>
              </span>`,
          )
          .join("")
      : `<span class="res32-placeholder">${escapeHtml(
          state.residual32FeaturesError || "Loading residual maps",
        )}</span>`;
    const afterCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="res32-map-card">
                <img src="${escapeHtml(channel.after_image)}" alt="" loading="lazy" />
                <strong>change ${Number(channel.change).toFixed(3)}</strong>
              </span>`,
          )
          .join("")
      : `<span class="res32-placeholder">${escapeHtml(
          state.residual32FeaturesError || "Loading residual maps",
        )}</span>`;
    return `
      <span class="res32-workflow">
        <span class="res32-panel res32-before">
          <b>Before residual block</b>
          <em>${escapeHtml(inputShape)}</em>
          <span class="res32-map-grid">${beforeCards}</span>
        </span>
        <span class="res32-branch">
          <b>Residual branch</b>
          <small>identity shortcut keeps the original tensor unchanged</small>
          <span>Conv 3x3</span>
          <span>BN + SiLU</span>
          <span>Conv 3x3</span>
          <span>BN</span>
          <em>learns a correction, then adds it to the shortcut</em>
        </span>
        <span class="res32-plus" aria-hidden="true">+</span>
        <span class="res32-panel res32-after">
          <b>After add + SiLU</b>
          <em>${escapeHtml(outputShape)}</em>
          <span class="res32-map-grid">${afterCards}</span>
        </span>
      </span>`;
  }
  if (visual === "res64") {
    const features = state.residual64Features;
    const inputShape = features?.input_shape?.join(" x ") || "32 x 112 x 112";
    const outputShape = features?.output_shape?.join(" x ") || "64 x 56 x 56";
    const branchCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="res64-map-card">
                <img src="${escapeHtml(channel.branch_image)}" alt="" loading="lazy" />
                <strong>filter ${String(channel.index).padStart(2, "0")}</strong>
              </span>`,
          )
          .join("")
      : `<span class="res64-placeholder">${escapeHtml(
          state.residual64FeaturesError || "Loading residual maps",
        )}</span>`;
    const shortcutCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="res64-map-card">
                <img src="${escapeHtml(channel.shortcut_image)}" alt="" loading="lazy" />
                <strong>projected</strong>
              </span>`,
          )
          .join("")
      : `<span class="res64-placeholder">${escapeHtml(
          state.residual64FeaturesError || "Loading residual maps",
        )}</span>`;
    const outputCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="res64-map-card">
                <img src="${escapeHtml(channel.output_image)}" alt="" loading="lazy" />
                <strong>energy ${Number(channel.energy).toFixed(3)}</strong>
              </span>`,
          )
          .join("")
      : `<span class="res64-placeholder">${escapeHtml(
          state.residual64FeaturesError || "Loading residual maps",
        )}</span>`;
    return `
      <span class="res64-workflow">
        <span class="res64-shape">
          <strong>${escapeHtml(inputShape)}</strong>
          <span>stride 2</span>
          <strong>${escapeHtml(outputShape)}</strong>
        </span>
        <span class="res64-panel">
          <b>Residual branch</b>
          <em>3x3 stride-2 conv learns denser texture maps</em>
          <span class="res64-map-grid">${branchCards}</span>
        </span>
        <span class="res64-combine">
          <span>+</span>
          <small>same shape before add</small>
        </span>
        <span class="res64-panel">
          <b>Projection shortcut</b>
          <em>1x1 stride-2 keeps old signal while resizing channels</em>
          <span class="res64-map-grid">${shortcutCards}</span>
        </span>
        <span class="res64-output">
          <b>After add + SiLU</b>
          <em>${escapeHtml(outputShape)} maps sent deeper</em>
          <span class="res64-map-grid">${outputCards}</span>
        </span>
      </span>`;
  }
  if (visual === "res128") {
    const features = state.residual128Features;
    const inputShape = features?.input_shape?.join(" x ") || "64 x 56 x 56";
    const outputShape = features?.output_shape?.join(" x ") || "128 x 28 x 28";
    const branchCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="res128-map-card">
                <img src="${escapeHtml(channel.branch_image)}" alt="" loading="lazy" />
                <strong>shape ${String(channel.index).padStart(2, "0")}</strong>
              </span>`,
          )
          .join("")
      : `<span class="res128-placeholder">${escapeHtml(
          state.residual128FeaturesError || "Loading residual maps",
        )}</span>`;
    const shortcutCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="res128-strip">
                <img src="${escapeHtml(channel.shortcut_image)}" alt="" loading="lazy" />
                <strong>old signal</strong>
              </span>`,
          )
          .join("")
      : `<span class="res128-placeholder">${escapeHtml(
          state.residual128FeaturesError || "Loading residual maps",
        )}</span>`;
    const outputCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="res128-map-card">
                <img src="${escapeHtml(channel.output_image)}" alt="" loading="lazy" />
                <strong>energy ${Number(channel.energy).toFixed(3)}</strong>
              </span>`,
          )
          .join("")
      : `<span class="res128-placeholder">${escapeHtml(
          state.residual128FeaturesError || "Loading residual maps",
        )}</span>`;
    return `
      <span class="res128-workflow">
        <span class="res128-stage-note">
          <strong>${escapeHtml(inputShape)} to ${escapeHtml(outputShape)}</strong>
          <small>resolution drops again, channels double, shape cues become more compact</small>
        </span>
        <span class="res128-panel res128-branch-panel">
          <b>Residual branch</b>
          <em>deeper 3x3 filters respond to body outline and posture</em>
          <span class="res128-map-grid">${branchCards}</span>
        </span>
        <span class="res128-shortcut-panel">
          <b>Projection shortcut</b>
          <em>1x1 stride-2 carries earlier evidence into the new 128-channel space</em>
          <span class="res128-strip-grid">${shortcutCards}</span>
        </span>
        <span class="res128-output-panel">
          <b>After add + SiLU</b>
          <em>compact shape maps prepared for attention and pooling</em>
          <span class="res128-map-grid">${outputCards}</span>
        </span>
      </span>`;
  }
  if (visual === "attention") {
    const features = state.attentionFeatures;
    const inputShape = features?.input_shape?.join(" x ") || "128 x 28 x 28";
    const branchShape = features?.branch_shape?.join(" x ") || "256 x 14 x 14";
    const gateShape = features?.gate_shape?.join(" x ") || "256 x 1 x 1";
    const outputShape = features?.output_shape?.join(" x ") || "256 x 14 x 14";
    const summary = features?.gate_summary;
    const gateRows = features?.channels?.length
      ? features.channels
          .map((channel) => {
            const gate = Number(channel.gate);
            return `
              <span class="attention-gate-row" style="--gate:${Math.max(0, Math.min(1, gate))}">
                <strong>ch ${String(channel.index).padStart(3, "0")}</strong>
                <span><span></span></span>
                <em>${(gate * 100).toFixed(1)}%</em>
              </span>`;
          })
          .join("")
      : `<span class="attention-placeholder">${escapeHtml(
          state.attentionFeaturesError || "Loading attention gates",
        )}</span>`;
    const mapCards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="attention-map-card">
                <strong>channel ${String(channel.index).padStart(3, "0")}</strong>
                <span>
                  <img src="${escapeHtml(channel.before_image)}" alt="" loading="lazy" />
                  <img src="${escapeHtml(channel.after_image)}" alt="" loading="lazy" />
                </span>
                <small>before gate -> after gate</small>
              </span>`,
          )
          .join("")
      : `<span class="attention-placeholder">${escapeHtml(
          state.attentionFeaturesError || "Loading attention maps",
        )}</span>`;
    return `
      <span class="attention-workflow">
        <span class="attention-note">
          <strong>${escapeHtml(inputShape)} to ${escapeHtml(outputShape)}</strong>
          <small>SE attention squeezes each channel to a score, then reweights feature maps</small>
        </span>
        <span class="attention-squeeze">
          <b>Squeeze</b>
          <em>global average pool</em>
          <span>${escapeHtml(branchShape)}</span>
          <strong>${escapeHtml(gateShape)}</strong>
        </span>
        <span class="attention-gates">
          <b>Excite gates</b>
          <em>sigmoid channel weights</em>
          <span class="attention-summary">
            <small>min ${summary ? (summary.min * 100).toFixed(1) : "--"}%</small>
            <small>mean ${summary ? (summary.mean * 100).toFixed(1) : "--"}%</small>
            <small>max ${summary ? (summary.max * 100).toFixed(1) : "--"}%</small>
          </span>
          <span class="attention-gate-list">${gateRows}</span>
        </span>
        <span class="attention-maps">
          <b>Reweighted maps</b>
          <em>bright channels are kept stronger</em>
          <span class="attention-map-grid">${mapCards}</span>
        </span>
      </span>`;
  }
  if (visual === "pool") {
    const features = state.poolFeatures;
    const inputShape = features?.input_shape?.join(" x ") || "256 x 14 x 14";
    const outputShape = features?.output_shape?.join(" x ") || "256";
    const cards = features?.channels?.length
      ? features.channels
          .map(
            (channel) => `
              <span class="pool-map-card">
                <img src="${escapeHtml(channel.image)}" alt="" loading="lazy" />
                <strong>ch ${String(channel.index).padStart(3, "0")}</strong>
                <small>max ${Number(channel.max).toFixed(3)}</small>
              </span>`,
          )
          .join("")
      : `<span class="pool-placeholder">${escapeHtml(
          state.poolFeaturesError || "Loading pooled features",
        )}</span>`;
    const vectorRows = features?.channels?.length
      ? features.channels
          .map((channel) => {
            const magnitude = Math.min(1, Math.abs(Number(channel.pooled_value)));
            return `
              <span class="pool-vector-row" style="--value:${magnitude}">
                <strong>ch ${String(channel.index).padStart(3, "0")}</strong>
                <span><span></span></span>
                <em>${Number(channel.pooled_value).toFixed(3)}</em>
              </span>`;
          })
          .join("")
      : `<span class="pool-placeholder">${escapeHtml(
          state.poolFeaturesError || "Loading pooled vector",
        )}</span>`;
    return `
      <span class="pool-workflow">
        <span class="pool-note">
          <strong>${escapeHtml(inputShape)} to ${escapeHtml(outputShape)} vector</strong>
          <small>each map is averaged across all spatial cells before the classifier reads it</small>
        </span>
        <span class="pool-panel pool-maps">
          <b>Last feature maps</b>
          <em>strongest channels before pooling</em>
          <span class="pool-map-grid">${cards}</span>
        </span>
        <span class="pool-operator">
          <strong>mean</strong>
          <small>H x W</small>
        </span>
        <span class="pool-panel pool-vector">
          <b>Pooled vector</b>
          <em>one value per channel</em>
          <span class="pool-vector-list">${vectorRows}</span>
        </span>
      </span>`;
  }
  if (visual === "gate") {
    const features = state.gateFeatures;
    const thresholds = features?.thresholds || { uncertain: 0.25, accepted: 0.55 };
    const confidence = Number(features?.confidence || 0);
    const decision = features?.decision || "loading";
    const vectorShape = features?.vector_shape?.join(" x ") || "256";
    const logitsShape = features?.logits_shape?.join(" x ") || "12";
    const topRows = features?.top_k?.length
      ? features.top_k
          .map((item, index) => {
            const score = Number(item.confidence);
            return `
              <span class="gate-class-row ${index === 0 ? "winner" : ""}" style="--score:${Math.max(0, Math.min(1, score))}">
                <strong>${index + 1}. ${escapeHtml(item.common_name_en || item.dataset_label)}</strong>
                <span><span></span></span>
                <em>${(score * 100).toFixed(1)}%</em>
                <small>logit ${Number(item.logit).toFixed(3)}</small>
              </span>`;
          })
          .join("")
      : `<span class="gate-placeholder">${escapeHtml(
          state.gateFeaturesError || "Loading decision scores",
        )}</span>`;
    return `
      <span class="gate-workflow">
        <span class="gate-note">
          <strong>${escapeHtml(vectorShape)} vector to ${escapeHtml(logitsShape)} logits</strong>
          <small>linear classifier ranks classes, softmax normalizes scores, thresholds decide whether to trust the result</small>
        </span>
        <span class="gate-panel gate-transform">
          <b>Classifier head</b>
          <em>Linear weights convert pooled evidence into class logits</em>
          <span class="gate-shape-flow">
            <strong>${escapeHtml(vectorShape)}</strong>
            <span>linear</span>
            <strong>${escapeHtml(logitsShape)}</strong>
            <span>softmax</span>
          </span>
          <small>margin ${(Number(features?.margin || 0) * 100).toFixed(1)} percentage points</small>
        </span>
        <span class="gate-panel gate-softmax">
          <b>Top-k probabilities</b>
          <em>highest softmax scores from the active sample</em>
          <span class="gate-class-list">${topRows}</span>
        </span>
        <span class="gate-panel gate-thresholds">
          <b>Decision threshold</b>
          <em>${escapeHtml(features?.reason || "Waiting for model decision")}</em>
          <span
            class="gate-decision-track"
            style="--confidence:${Math.max(0, Math.min(1, confidence))}; --uncertain:${thresholds.uncertain}; --accepted:${thresholds.accepted}"
          >
            <span class="gate-zone unsupported"></span>
            <span class="gate-zone uncertain"></span>
            <span class="gate-zone accepted"></span>
            <strong></strong>
          </span>
          <span class="gate-threshold-labels">
            <small>uncertain >= ${(thresholds.uncertain * 100).toFixed(0)}%</small>
            <strong>${escapeHtml(decision)} ${(confidence * 100).toFixed(1)}%</strong>
            <small>accepted >= ${(thresholds.accepted * 100).toFixed(0)}%</small>
          </span>
        </span>
      </span>`;
  }
  return "<i></i><i></i><i></i><i></i>";
}

function layerDefinitions() {
  return [
    {
      label: "Input",
      note: "Crop, resize, normalize",
      visual: "input",
      tasks: ["Preserve whole insect", "Standardize RGB scale"],
    },
    {
      label: "Stem conv",
      note: "Conv-BN-SiLU, stride 2",
      visual: "stem",
      tasks: ["Learn 32 local filters", "Downsample to 112x112 maps"],
    },
    {
      label: "Residual 32",
      note: "Identity shortcut + learned correction",
      visual: "res32",
      tasks: ["Preserve 32x112x112 maps", "Refine texture without losing input"],
    },
    {
      label: "Residual 64",
      note: "Downsample + projection shortcut",
      visual: "res64",
      tasks: ["Compress 112x112 to 56x56", "Expand 32 channels to 64"],
    },
    {
      label: "Residual 128",
      note: "Compact shape maps",
      visual: "res128",
      tasks: ["Compress 56x56 to 28x28", "Prepare 128-channel evidence"],
    },
    {
      label: "SE attention",
      note: "Squeeze-excite channel gates",
      visual: "attention",
      tasks: ["Pool each channel to one score", "Reweight feature maps with sigmoid gates"],
    },
    {
      label: "Global pool",
      note: "Adaptive average pooling",
      visual: "pool",
      tasks: ["Average each channel over H x W", "Convert feature maps into classifier vector"],
    },
    {
      label: "Decision gate",
      note: "Classifier logits + thresholds",
      visual: "gate",
      tasks: ["Convert logits into softmax probabilities", "Apply accepted and uncertain thresholds"],
    },
  ];
}

function renderLayerFlow() {
  const layers = layerDefinitions();
  const activeIndex = Math.min(Math.max(state.activeLayerIndex, 0), layers.length - 1);
  const active = layers[activeIndex];
  elements.layerFlow.innerHTML = `
      <div class="layer-workbench">
        <article class="layer-focus layer-${escapeHtml(active.visual)}" style="--step:${activeIndex}">
          <div class="layer-focus-head">
            <span>${String(activeIndex + 1).padStart(2, "0")} / ${layers.length}</span>
            <div>
              <strong>${escapeHtml(active.label)}</strong>
              <small>${escapeHtml(active.note)}</small>
            </div>
          </div>
          <div class="layer-visual layer-focus-visual ${active.visual === "input" ? "input-visual" : ""}" aria-hidden="true">
            ${layerVisualMarkup(active.visual)}
          </div>
          <div class="layer-focus-tasks">
            ${active.tasks.map((task) => `<span>${escapeHtml(task)}</span>`).join("")}
          </div>
        </article>
        <div class="layer-stepper" role="tablist" aria-label="CNN stages">
          ${layers
            .map(
              (layer, index) => `
                <button
                  class="layer-step ${index === activeIndex ? "active" : ""}"
                  type="button"
                  role="tab"
                  aria-selected="${index === activeIndex ? "true" : "false"}"
                  data-layer-index="${index}"
                >
                  <span>${String(index + 1).padStart(2, "0")}</span>
                  <strong>${escapeHtml(layer.label)}</strong>
                </button>`,
            )
            .join("")}
        </div>
      </div>`;
  elements.layerFlow.querySelectorAll(".layer-step").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeLayerIndex = Number(button.dataset.layerIndex);
      renderLayerFlow();
    });
  });
}

function renderPerClassMetrics(perClass) {
  const classesByIndex = classByIndexMap();
  const rows = perClass
    .map((row) => ({ ...row, classInfo: classesByIndex.get(Number(row.index)) }))
    .filter((row) => row.classInfo)
    .sort((a, b) => Number(b.f1 || 0) - Number(a.f1 || 0));
  elements.perClassMetrics.innerHTML = rows
    .map(
      (row) => `
        <div class="metric-row ${metricTone(row.f1)}">
          <div>
            <strong>${escapeHtml(displayCommonName(row.classInfo))}</strong>
            <small>${escapeHtml(scientificName(row.classInfo))} - n=${Number(row.support || 0)}</small>
          </div>
          <span>${percent(row.precision)}</span>
          <span>${percent(row.recall)}</span>
          <span>${percent(row.f1)}</span>
        </div>`,
    )
    .join("");
  elements.perClassMetrics.insertAdjacentHTML(
    "afterbegin",
    '<div class="metric-row header"><div>Class</div><span>Precision</span><span>Recall</span><span>F1</span></div>',
  );
  return rows;
}

function explainWeakness(row) {
  if (Number(row.recall || 0) < 0.4) return "Recall is low: many true samples are missed.";
  if (Number(row.precision || 0) < 0.4) return "Precision is low: the class is over-predicted.";
  return "Borderline F1: keep it in the next ablation watchlist.";
}

function topConfusions(best, classesByIndex) {
  const matrix = best.confusion_matrix || [];
  const pairs = [];
  matrix.forEach((row, actualIndex) => {
    row.forEach((count, predictedIndex) => {
      if (actualIndex === predictedIndex || Number(count) <= 0) return;
      const actual = classesByIndex.get(actualIndex);
      const predicted = classesByIndex.get(predictedIndex);
      if (!actual || !predicted) return;
      pairs.push({
        count: Number(count),
        actual: displayCommonName(actual),
        predicted: displayCommonName(predicted),
      });
    });
  });
  return pairs.sort((a, b) => b.count - a.count).slice(0, 4);
}

function renderOptimizationPanel(rows, best, decision) {
  const classesByIndex = classByIndexMap();
  const weakest = [...rows].sort((a, b) => Number(a.f1 || 0) - Number(b.f1 || 0))[0];
  const confusion = topConfusions(best, classesByIndex)[0];
  const currentTraining = state.model.training || {};
  const activeLoss = currentTraining.loss || "cross_entropy";
  elements.optimizationLab.innerHTML = `
    <div class="simple-plan-grid">
      <article>
        <span>Model health</span>
        <strong>${percent(best.macro_f1)} macro-F1</strong>
        <small>${percent(decision.id_accepted_precision)} precision when the gate accepts ${percent(decision.id_coverage)} of validation samples.</small>
      </article>
      <article>
        <span>Fix first</span>
        <strong>${weakest ? escapeHtml(displayCommonName(weakest.classInfo)) : "No weak class"}</strong>
        <small>${weakest ? `${percent(weakest.f1)} F1. ${escapeHtml(explainWeakness(weakest))}` : "Metrics are not available in this bundle."}</small>
      </article>
      <article>
        <span>Most confused pair</span>
        <strong>${confusion ? `${escapeHtml(confusion.actual)} -> ${escapeHtml(confusion.predicted)}` : "No confusion data"}</strong>
        <small>${confusion ? `${confusion.count} validation samples. Add harder examples or class-balanced augmentation here.` : "Confusion matrix is not available."}</small>
      </article>
      <article>
        <span>Next run</span>
        <strong>${escapeHtml(titleCase(activeLoss))}</strong>
        <small>Try the optimized training config, then compare this Species page again after calibration.</small>
      </article>
    </div>
  `;
}

async function selectExample(exampleId, silent = false) {
  const sample = state.examples.find((item) => item.id === exampleId);
  if (!sample) return;
  state.selectedExample = sample;
  state.currentFile = null;
  elements.imageUpload.value = "";
  elements.predictUpload.disabled = true;
  elements.inspectionImage.src = `${sample.image_url}?t=${Date.now()}`;
  elements.inspectionImage.alt = displayCommonName(sample);
  elements.inputSource.textContent = `Sample / ${imageKindLabel(sample)}`;
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
    '<div class="spinner"></div><strong>Running inference</strong><p>Preparing the image and scoring classes...</p>';
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
    result.decision === "accepted" ? displayCommonName(top) : decisionLabel(result.decision);
  elements.outcomeIndicator.className = `outcome-indicator ${decisionTone(result.decision)}`;
  elements.matchList.innerHTML = result.top_k
    .map((match, index) => {
      const percent = Math.max(0, Math.min(100, match.confidence * 100));
      return `
        <div class="match-row">
          <div>
            <span>${index + 1}. ${escapeHtml(displayCommonName(match))}</span>
            <strong>${percent.toFixed(1)}%</strong>
          </div>
          <div class="match-track"><i style="width:${percent}%"></i></div>
          <small>${escapeHtml(scientificName(match))} / ${escapeHtml(titleCase(match.stratum))}</small>
        </div>`;
    })
    .join("");
  elements.thresholdNote.textContent = result.reason;
  renderConfidenceGate(result);
}

function renderConfidenceGate(result) {
  const thresholds = state.model?.thresholds || { accepted: 0.55, uncertain: 0.25 };
  const confidence = Math.max(0, Math.min(1, result.confidence || 0));
  const accepted = Math.max(0, Math.min(1, thresholds.accepted));
  const uncertain = Math.max(0, Math.min(accepted, thresholds.uncertain));
  elements.gateCurrent.textContent = `${(confidence * 100).toFixed(1)}%`;
  elements.gateCurrentMarker.style.left = `${confidence * 100}%`;
  elements.gateCurrentMarker.className = `gate-current-marker ${decisionTone(result.decision)}`;
  elements.gateAcceptedMarker.style.left = `${accepted * 100}%`;
  elements.gateUncertainMarker.style.left = `${uncertain * 100}%`;
  elements.gateAcceptedLabel.textContent = `Accepted >= ${(accepted * 100).toFixed(1)}%`;
  elements.gateUncertainLabel.textContent = `Uncertain >= ${(uncertain * 100).toFixed(1)}%`;
}

function predictionReport() {
  if (!state.prediction) return null;
  const source = state.currentFile?.name || state.selectedExample?.id || "unknown";
  const top = state.prediction.top_k[0];
  return {
    app: "PestScope",
    model_version: state.prediction.model_version,
    source,
    decision: state.prediction.decision,
    confidence: state.prediction.confidence,
    top_prediction: {
      class_id: top.class_id,
      common_name: displayCommonName(top),
      scientific_name: scientificName(top),
    },
    top_k: state.prediction.top_k.map((item) => ({
      class_id: item.class_id,
      common_name: displayCommonName(item),
      scientific_name: scientificName(item),
      confidence: item.confidence,
    })),
    thresholds: state.model?.thresholds,
    latency_ms: state.prediction.latency_ms,
  };
}

async function copyScoutNote() {
  const report = predictionReport();
  if (!report) return showToast("Run a prediction before copying a result note", "error");
  const lines = [
    `PestScope result note`,
    `Model: ${report.model_version}`,
    `Source: ${report.source}`,
    `Decision: ${report.decision}`,
    `Top match: ${report.top_prediction.common_name} (${report.top_prediction.scientific_name})`,
    `Confidence: ${(report.confidence * 100).toFixed(1)}%`,
    `Alternatives: ${report.top_k
      .slice(1)
      .map((item) => `${item.common_name} ${(item.confidence * 100).toFixed(1)}%`)
      .join(", ")}`,
  ];
  try {
    await navigator.clipboard.writeText(lines.join("\n"));
    showToast("Result note copied");
  } catch {
    showToast("Clipboard is unavailable in this browser", "error");
  }
}

function downloadPredictionJson() {
  const report = predictionReport();
  if (!report) return showToast("Run a prediction before downloading JSON", "error");
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `pestscope-${report.decision}-${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function runSampleAudit() {
  if (!state.examples.length) return;
  elements.runAudit.disabled = true;
  elements.runAudit.textContent = "Auditing...";
  elements.sampleAudit.hidden = false;
  elements.sampleAudit.innerHTML = '<div class="audit-loading">Running sample audit</div>';
  const rows = [];
  for (const sample of state.examples) {
    const result = await api(`/api/v1/examples/${sample.id}/predict?top_k=3`, { method: "POST" });
    const topIds = result.top_k.map((item) => item.class_id);
    rows.push({
      sample,
      result,
      top1: topIds[0] === sample.class_id,
      top3: topIds.includes(sample.class_id),
    });
  }
  renderSampleAudit(rows);
  elements.runAudit.disabled = false;
  elements.runAudit.textContent = "Run audit";
}

function renderSampleAudit(rows) {
  const top1 = rows.filter((row) => row.top1).length;
  const top3 = rows.filter((row) => row.top3).length;
  const accepted = rows.filter((row) => row.result.decision === "accepted").length;
  elements.sampleAudit.innerHTML = `
    <div class="audit-summary">
      <span><strong>${top1}/${rows.length}</strong> top-1</span>
      <span><strong>${top3}/${rows.length}</strong> top-3</span>
      <span><strong>${accepted}</strong> accepted</span>
    </div>
    <div class="audit-rows">
      ${rows
        .map((row) => {
          const top = row.result.top_k[0];
          return `
            <div class="audit-row ${row.top1 ? "pass" : row.top3 ? "partial" : "fail"}">
              <span>${escapeHtml(displayCommonName(row.sample))}</span>
              <strong>${escapeHtml(displayCommonName(top))}</strong>
              <code>${row.result.decision}</code>
            </div>`;
        })
        .join("")}
    </div>`;
}

async function submitReview() {
  if (!state.prediction) return showToast("No prediction is available to review", "error");
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
  elements.resultEmpty.innerHTML = `<strong>Prediction failed</strong><p>${escapeHtml(message)}</p>`;
  elements.resultLatency.textContent = "Error";
  showToast(message, "error");
}

function decisionLabel(decision) {
  if (decision === "uncertain") return "Needs field review";
  if (decision === "unsupported") return "Outside supported scope";
  return "Accepted match";
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
  elements.runAudit.addEventListener("click", runSampleAudit);
  elements.copyReport.addEventListener("click", copyScoutNote);
  elements.downloadReport.addEventListener("click", downloadPredictionJson);
  window.addEventListener("scroll", () => window.requestAnimationFrame(updateParallax), {
    passive: true,
  });
  window.addEventListener("resize", updateParallax);
}

async function loadLayerFeaturesInBackground() {
  await Promise.allSettled([
    loadStemFeatures(),
    loadResidual32Features(),
    loadResidual64Features(),
    loadResidual128Features(),
    loadAttentionFeatures(),
    loadPoolFeatures(),
    loadGateFeatures(),
  ]);
  renderModelLab();
}

async function initialize() {
  bindEvents();
  try {
    elements.setupMessage.textContent = "Loading model bundle...";
    const ready = await api("/api/v1/health/ready");
    state.model = await api("/api/v1/model");
    try {
      state.experiment = await api("/api/v1/experiments/current");
      state.experimentError = null;
    } catch (error) {
      state.experiment = null;
      state.experimentError = error.message;
    }
    const { examples } = await api("/api/v1/examples");
    state.examples = examples;

    setServiceState(true, "Ready");
    elements.modelName.textContent = ready.model_version;
    elements.demoReady.textContent = "Model ready";
    elements.demoReady.classList.add("ready");
    if (state.model.demo_model) {
      elements.demoWarning.hidden = false;
      elements.demoWarning.textContent =
        "Demo fallback is active for UI/API smoke tests. Train and mount PestNet-S before reporting model metrics.";
    }

    renderExamples();
    renderClasses();
    renderTrustSnapshot();
    renderExperimentEvidence();
    renderModelLab();
    elements.setupScreen.classList.add("done");
    elements.app.hidden = false;
    window.setTimeout(() => (elements.setupScreen.hidden = true), 350);
    const requestedView = window.location.hash.replace("#", "");
    const normalizedView = requestedView === "trust" ? "experiments" : requestedView;
    switchView(["inspect", "classes", "experiments", "lab"].includes(normalizedView) ? normalizedView : "inspect");
    updateParallax();
    loadLayerFeaturesInBackground();
    if (state.examples.length) await selectExample(state.examples[0].id, true);
  } catch (error) {
    setServiceState(false, "Unavailable");
    elements.setupMessage.textContent = error.message;
    elements.setupScreen.classList.add("failed");
  }
}

initialize();

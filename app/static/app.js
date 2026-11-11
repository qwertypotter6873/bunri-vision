(() => {
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));

  function activateTab(selectedTab) {
    for (const tab of tabs) {
      const isSelected = tab === selectedTab;
      tab.setAttribute("aria-selected", String(isSelected));
      const panel = document.getElementById(tab.dataset.panel);
      if (panel) {
        if (isSelected) {
          panel.removeAttribute("hidden");
          panel.setAttribute("tabindex", "0");
        } else {
          panel.setAttribute("hidden", "until-found");
          panel.removeAttribute("tabindex");
        }
      }
    }
  }

  for (const tab of tabs) {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", (e) => {
      const idx = tabs.indexOf(tab);
      if (e.key === "ArrowRight") {
        const next = tabs[(idx + 1) % tabs.length];
        next.focus();
        activateTab(next);
      } else if (e.key === "ArrowLeft") {
        const prev = tabs[(idx - 1 + tabs.length) % tabs.length];
        prev.focus();
        activateTab(prev);
      }
    });
  }

  for (const panel of panels) {
    panel.addEventListener("beforematch", () => {
      const matchingTab = tabs.find((t) => t.dataset.panel === panel.id);
      if (matchingTab) activateTab(matchingTab);
    });
  }

  const runtimeText = document.getElementById("runtime-model-text");
  const sampleGrid = document.getElementById("sample-grid");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const latencyTag = document.getElementById("latency-tag");
  const errorBanner = document.getElementById("error-banner");
  const errorCodeBadge = document.getElementById("error-code-badge");
  const errorMessageText = document.getElementById("error-message-text");
  const emptyState = document.getElementById("empty-state");
  const predictionContent = document.getElementById("prediction-content");

  function showError(code, msg) {
    errorBanner.hidden = false;
    errorCodeBadge.textContent = code;
    errorMessageText.textContent = msg;
    latencyTag.textContent = "예외 차단됨";
  }

  function clearError() {
    errorBanner.hidden = true;
  }

  function renderPrediction(data) {
    clearError();
    emptyState.hidden = true;
    predictionContent.hidden = false;

    latencyTag.textContent = `${data.model_checkpoint} · ${data.inference_ms} ms`;

    const pill = document.getElementById("pred-status-pill");
    pill.textContent = data.rule.recyclable_status;
    pill.dataset.badge = data.rule.bin_badge_type;

    document.getElementById("pred-ko-title").textContent = `${data.predicted_ko_name} (${data.predicted_class})`;
    document.getElementById("pred-bin-target").textContent = `배출함: ${data.rule.bin_category}`;
    document.getElementById("pred-confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById("pred-entropy").textContent = `정규화 엔트로피 ${data.normalized_entropy.toFixed(2)}`;

    const gtBar = document.getElementById("ground-truth-bar");
    if (data.sample_ground_truth) {
      const matched = data.sample_ground_truth.ground_truth_en === data.predicted_class;
      gtBar.hidden = false;
      gtBar.textContent = `홀드아웃 테스트셋 실제 정답: ${data.sample_ground_truth.ground_truth_ko} (${data.sample_ground_truth.ground_truth_en}) → ${matched ? "정답 일치 ✓" : "경계 오차"}`;
    } else {
      gtBar.hidden = true;
    }

    const alertsStack = document.getElementById("alerts-stack");
    alertsStack.replaceChildren();
    for (const al of data.alerts || []) {
      const div = document.createElement("div");
      div.className = "alert-card";
      div.dataset.level = al.level;
      const st = document.createElement("strong");
      st.textContent = al.title;
      const span = document.createElement("span");
      span.textContent = al.message;
      div.append(st, span);
      alertsStack.appendChild(div);
    }

    document.getElementById("img-raw").src = data.visuals.raw_image_url;
    document.getElementById("img-preproc").src = data.visuals.preprocessed_image_url;
    document.getElementById("img-cam").src = data.visuals.cam_overlay_url;

    const qr = data.quality_report;
    document.getElementById("stat-before").textContent = `대비 ${qr.contrast_std} · 색편향 ${qr.color_cast_index}`;
    document.getElementById("stat-after").textContent = `대비 ${qr.after_contrast_std} · 색편향 ${qr.after_color_cast_index}`;

    const stepsList = document.getElementById("steps-list");
    stepsList.replaceChildren();
    for (const step of data.rule.steps) {
      const li = document.createElement("li");
      li.textContent = step;
      stepsList.appendChild(li);
    }
    document.getElementById("penalty-trap-text").textContent = data.rule.penalty_trap;

    const probBars = document.getElementById("prob-bars");
    probBars.replaceChildren();
    data.top_predictions.forEach((item, idx) => {
      const pct = (item.probability * 100).toFixed(1);
      const li = document.createElement("li");
      li.className = idx === 0 ? "prob-row" : "prob-row secondary";

      const nameEl = document.createElement("span");
      nameEl.className = "prob-name";
      nameEl.textContent = item.class_ko;

      const track = document.createElement("div");
      track.className = "prob-track";
      const fill = document.createElement("div");
      fill.className = "prob-fill";
      fill.style.setProperty("--prob-pct", pct);
      track.appendChild(fill);

      const pctEl = document.createElement("span");
      pctEl.className = "prob-pct-text";
      pctEl.textContent = `${pct}%`;

      li.append(nameEl, track, pctEl);
      probBars.appendChild(li);
    });
  }

  async function uploadFile(file) {
    if (!file) return;
    latencyTag.textContent = "AI 모델 추론 중...";
    document.querySelectorAll(".sample-btn").forEach((b) => b.classList.remove("active"));

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const payload = await res.json();
      if (!res.ok || payload.error) {
        showError(payload.error_code || "UPLOAD_ERROR", payload.message || "파일 검증 실패");
        return;
      }
      renderPrediction(payload);
    } catch (err) {
      showError("NETWORK_ERROR", "서버와 통신 중 오류가 발생했습니다.");
    }
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      uploadFile(fileInput.files[0]);
      fileInput.value = "";
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-active");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-active"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-active");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  document.querySelectorAll("[data-exception]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const caseType = btn.dataset.exception;
      latencyTag.textContent = "예외 검증 중...";
      const res = await fetch(`/api/test-exception/${caseType}`, { method: "POST" });
      const payload = await res.json();
      if (payload.error) {
        showError(payload.error_code, `[테스트 파일: ${payload.tested_file}] ${payload.message}`);
      }
    });
  });

  async function initApp() {
    try {
      const health = await (await fetch("/api/health")).json();
      const f1 = health.metadata && health.metadata.test_macro_f1 ? ` · Test F1 ${health.metadata.test_macro_f1}` : "";
      runtimeText.textContent = `자체 학습 가중치 로드됨 (${health.loaded_checkpoint}${f1})`;
    } catch {
      runtimeText.textContent = "오프라인 상태";
    }

    try {
      const { samples } = await (await fetch("/api/samples")).json();
      sampleGrid.replaceChildren();
      for (const s of samples) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sample-btn";
        const img = document.createElement("img");
        img.src = s.image_url;
        img.alt = `${s.ground_truth_ko} 테스트 샘플`;
        img.width = 120;
        img.height = 120;
        img.loading = "lazy";
        const label = document.createElement("span");
        label.textContent = s.ground_truth_ko;
        btn.append(img, label);
        btn.addEventListener("click", async () => {
          document.querySelectorAll(".sample-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          latencyTag.textContent = "샘플 추론 중...";
          const res = await fetch(`/api/predict-sample/${s.id}`, { method: "POST" });
          const payload = await res.json();
          renderPrediction(payload);
        });
        sampleGrid.appendChild(btn);
      }
    } catch {}

    try {
      const bench = await (await fetch("/api/benchmark")).json();
      const tbody = document.getElementById("bench-tbody");
      const order = [
        "m1_scratch_cnn",
        "m2_mobilenetv3",
        "m3_resnet50",
        "m4_ablation_raw",
        "m4_proposed_convnext",
      ];
      tbody.replaceChildren();
      for (const key of order) {
        const m = bench.models && bench.models[key];
        if (!m) continue;
        const tr = document.createElement("tr");
        if (key === bench.selected_final_model) tr.className = "highlight-row";
        tr.innerHTML = `
          <th>${m.display_name}</th>
          <td>${m.training_mode}</td>
          <td>${m.param_millions}M</td>
          <td>${m.test_macro_f1.toFixed(4)}</td>
          <td>${(m.test_accuracy * 100).toFixed(2)}%</td>
          <td>${(m.test_top2_accuracy * 100).toFixed(2)}%</td>
          <td>${m.test_macro_auroc.toFixed(4)}</td>
          <td>${m.test_ece.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
      }

      const finalModel = bench.models && bench.models[bench.selected_final_model];
      const perClassBody = document.getElementById("per-class-tbody");
      if (finalModel && finalModel.per_class) {
        perClassBody.replaceChildren();
        for (const [enName, stats] of Object.entries(finalModel.per_class)) {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <th>${stats.ko_name} (${enName})</th>
            <td>${stats.precision.toFixed(4)}</td>
            <td>${stats.recall.toFixed(4)}</td>
            <td><strong>${stats.f1.toFixed(4)}</strong></td>
            <td>${stats.support}장</td>
          `;
          perClassBody.appendChild(tr);
        }
      }
    } catch {}

    try {
      const { classes } = await (await fetch("/api/rules")).json();
      const acc = document.getElementById("rules-accordion");
      acc.replaceChildren();
      classes.forEach((r, idx) => {
        const det = document.createElement("details");
        det.name = "waste-dict";
        if (idx === 0) det.open = true;
        const sum = document.createElement("summary");
        sum.textContent = `${r.ko_name} (${r.en_name}) — ${r.bin_category}`;
        const body = document.createElement("div");
        body.className = "accordion-body";
        body.innerHTML = `
          <p><strong>대표 품목:</strong> ${r.examples}</p>
          <p><strong>배출 핵심:</strong> ${r.steps.join(" / ")}</p>
          <p><strong>주의사항:</strong> ${r.penalty_trap}</p>
        `;
        det.append(sum, body);
        acc.appendChild(det);
      });
    } catch {}
  }

  initApp();
})();

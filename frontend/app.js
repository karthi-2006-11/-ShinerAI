/**
 * ShinerAI: Dashboard Frontend Controller (Phase 5)
 * Pure JavaScript client integrating with Flask REST API:
 *   GET /health
 *   GET /model-info
 *   POST /predict
 *   POST /explain
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // Configuration & State
    // -------------------------------------------------------------------------
    const API_BASE = ""; // Relative path to current origin
    const OPERATIONAL_DO_THRESHOLD = 3.0; // mg/L provisional research threshold

    // Predefined Evaluated Test Scenarios from Phase 3 Holdout Set
    const DEMO_SCENARIOS = {
        safe: {
            pond_id: "ara2_0677080b",
            prediction_timestamp: "2026-01-26T10:15:00",
            current_do: 5.40,
            do_t_minus_15: 5.15,
            do_t_minus_30: 4.87,
            do_t_minus_45: 4.36,
            do_t_minus_60: 4.21,
            do_t_minus_75: 3.89,
            do_t_minus_90: 3.72,
            do_t_minus_105: 3.23,
            do_t_minus_120: 2.92
        },
        at_risk: {
            pond_id: "ara2_0677080b",
            prediction_timestamp: "2026-01-26T03:30:00",
            current_do: 3.84,
            do_t_minus_15: 4.05,
            do_t_minus_30: 4.49,
            do_t_minus_45: 4.67,
            do_t_minus_60: 4.69,
            do_t_minus_75: 4.73,
            do_t_minus_90: 4.85,
            do_t_minus_105: 5.00,
            do_t_minus_120: 5.02
        }
    };

    // Human-readable labels for model predictors
    const FEATURE_LABELS = {
        "current_do": "Current Dissolved Oxygen (T)",
        "do_t_minus_15": "DO 15 min ago (T - 15m)",
        "do_t_minus_30": "DO 30 min ago (T - 30m)",
        "do_t_minus_45": "DO 45 min ago (T - 45m)",
        "do_t_minus_60": "DO 60 min ago (T - 60m)",
        "do_t_minus_75": "DO 75 min ago (T - 75m)",
        "do_t_minus_90": "DO 90 min ago (T - 90m)",
        "do_t_minus_105": "DO 105 min ago (T - 105m)",
        "do_t_minus_120": "DO 120 min ago (T - 120m)",
        "hour_of_day": "Hour of Day (Diurnal Cycle)",
        "minute_of_day": "Minute of Day (Time of Day)"
    };

    // Frozen Global Feature Importance for XGBoost Config C
    const GLOBAL_IMPORTANCE = [
        { feature: "current_do", name: "Current Dissolved Oxygen (T)", mean_shap: 1.3896 },
        { feature: "minute_of_day", name: "Minute of Day (Diurnal Phase)", mean_shap: 0.5388 },
        { feature: "hour_of_day", name: "Hour of Day (Diurnal Cycle)", mean_shap: 0.2549 },
        { feature: "do_t_minus_30", name: "DO 30 min ago (T - 30m)", mean_shap: 0.2388 },
        { feature: "do_t_minus_15", name: "DO 15 min ago (T - 15m)", mean_shap: 0.2385 },
        { feature: "do_t_minus_120", name: "DO 120 min ago (T - 120m)", mean_shap: 0.1562 },
        { feature: "do_t_minus_45", name: "DO 45 min ago (T - 45m)", mean_shap: 0.1555 },
        { feature: "do_t_minus_90", name: "DO 90 min ago (T - 90m)", mean_shap: 0.1237 },
        { feature: "do_t_minus_105", name: "DO 105 min ago (T - 105m)", mean_shap: 0.1117 },
        { feature: "do_t_minus_60", name: "DO 60 min ago (T - 60m)", mean_shap: 0.1058 },
        { feature: "do_t_minus_75", name: "DO 75 min ago (T - 75m)", mean_shap: 0.0986 }
    ];

    // DOM Elements
    const apiStatusBadge = document.getElementById("apiStatusBadge");
    const apiStatusText = document.getElementById("apiStatusText");
    const activeModelBadge = document.getElementById("activeModelBadge");
    const apiErrorBanner = document.getElementById("apiErrorBanner");
    const btnRetryConnection = document.getElementById("btnRetryConnection");

    const btnDemoSafe = document.getElementById("btnDemoSafe");
    const btnDemoAtRisk = document.getElementById("btnDemoAtRisk");
    const btnClearInputs = document.getElementById("btnClearInputs");

    const pondInputForm = document.getElementById("pondInputForm");
    const pondIdInput = document.getElementById("pondId");
    const predictionTimestampInput = document.getElementById("predictionTimestamp");
    const currentDoInput = document.getElementById("currentDo");
    const validationAlert = document.getElementById("validationAlert");
    const validationAlertMessage = document.getElementById("validationAlertMessage");
    const btnAnalyze = document.getElementById("btnAnalyze");

    const doChartSvg = document.getElementById("doChart");

    const loadingPrediction = document.getElementById("loadingPrediction");
    const placeholderPrediction = document.getElementById("placeholderPrediction");
    const resultContent = document.getElementById("resultContent");
    const riskBanner = document.getElementById("riskBanner");
    const riskIcon = document.getElementById("riskIcon");
    const riskLabel = document.getElementById("riskLabel");
    const riskProbabilityValue = document.getElementById("riskProbabilityValue");
    const riskMeterFill = document.getElementById("riskMeterFill");
    const resCurrentDo = document.getElementById("resCurrentDo");
    const resTimestamp = document.getElementById("resTimestamp");
    const resModelUsed = document.getElementById("resModelUsed");
    const resConfig = document.getElementById("resConfig");
    const alertMessageBox = document.getElementById("alertMessageBox");
    const alertMessageText = document.getElementById("alertMessageText");

    const loadingExplanation = document.getElementById("loadingExplanation");
    const placeholderExplanation = document.getElementById("placeholderExplanation");
    const explanationContent = document.getElementById("explanationContent");
    const shapTableBody = document.getElementById("shapTableBody");
    const topRiskDriversList = document.getElementById("topRiskDriversList");
    const topSafeDriversList = document.getElementById("topSafeDriversList");

    const globalImportanceChart = document.getElementById("globalImportanceChart");
    const infoModelName = document.getElementById("infoModelName");

    // Lag input IDs in chronological sequence from past to present
    const lagInputIds = [
        "do_t_minus_120",
        "do_t_minus_105",
        "do_t_minus_90",
        "do_t_minus_75",
        "do_t_minus_60",
        "do_t_minus_45",
        "do_t_minus_30",
        "do_t_minus_15"
    ];

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    function init() {
        renderGlobalFeatureImportance();
        checkApiHealth();
        renderDoChart();
        setupEventListeners();
    }

    // -------------------------------------------------------------------------
    // API Health & Model Info
    // -------------------------------------------------------------------------
    async function checkApiHealth() {
        try {
            apiStatusBadge.className = "status-badge status-checking";
            apiStatusText.textContent = "Checking API...";

            const response = await fetch(`${API_BASE}/health`, {
                headers: { "Accept": "application/json" }
            });

            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const data = await response.json();
            if (data.status === "healthy" && data.model_loaded) {
                apiStatusBadge.className = "status-badge status-connected";
                apiStatusText.textContent = "● API Connected";
                activeModelBadge.textContent = `Model: ${data.model_family} (${data.model_artifact})`;
                apiErrorBanner.classList.add("hidden");
                fetchModelInfo();
            } else {
                throw new Error("Model not fully loaded or unhealthy");
            }
        } catch (err) {
            apiStatusBadge.className = "status-badge status-offline";
            apiStatusText.textContent = "● API Offline";
            activeModelBadge.textContent = "Model: Unavailable";
            apiErrorBanner.classList.remove("hidden");
        }
    }

    async function fetchModelInfo() {
        try {
            const response = await fetch(`${API_BASE}/model-info`, {
                headers: { "Accept": "application/json" }
            });
            if (response.ok) {
                const info = await response.json();
                if (infoModelName && info.model_family) {
                    infoModelName.textContent = `${info.model_family} (${info.model_artifact})`;
                }
            }
        } catch (err) {
            console.warn("Could not retrieve model specifications:", err);
        }
    }

    // -------------------------------------------------------------------------
    // Global Feature Importance Rendering
    // -------------------------------------------------------------------------
    function renderGlobalFeatureImportance() {
        if (!globalImportanceChart) return;
        const maxVal = GLOBAL_IMPORTANCE[0].mean_shap;

        globalImportanceChart.innerHTML = GLOBAL_IMPORTANCE.map((item, idx) => {
            const pct = ((item.mean_shap / maxVal) * 100).toFixed(1);
            return `
                <div class="global-bar-row">
                    <span class="global-bar-label" title="${item.name}">${idx + 1}. ${item.feature}</span>
                    <div class="global-bar-track">
                        <div class="global-bar-fill" style="width: ${pct}%;"></div>
                    </div>
                    <span class="global-bar-val">${item.mean_shap.toFixed(4)}</span>
                </div>
            `;
        }).join("");
    }

    // -------------------------------------------------------------------------
    // Event Listeners
    // -------------------------------------------------------------------------
    function setupEventListeners() {
        if (btnRetryConnection) {
            btnRetryConnection.addEventListener("click", checkApiHealth);
        }

        if (btnDemoSafe) {
            btnDemoSafe.addEventListener("click", () => loadScenario(DEMO_SCENARIOS.safe));
        }

        if (btnDemoAtRisk) {
            btnDemoAtRisk.addEventListener("click", () => loadScenario(DEMO_SCENARIOS.at_risk));
        }

        if (btnClearInputs) {
            btnClearInputs.addEventListener("click", resetForm);
        }

        // Live chart updates when inputs change
        [currentDoInput, ...lagInputIds.map(id => document.getElementById(id))].forEach(input => {
            if (input) {
                input.addEventListener("input", () => {
                    hideValidationAlert();
                    renderDoChart();
                });
            }
        });

        if (pondInputForm) {
            pondInputForm.addEventListener("submit", handleFormSubmit);
        }
    }

    // -------------------------------------------------------------------------
    // Scenario Loading & Form Reset
    // -------------------------------------------------------------------------
    function loadScenario(scenario) {
        hideValidationAlert();
        pondIdInput.value = scenario.pond_id;
        predictionTimestampInput.value = scenario.prediction_timestamp;
        currentDoInput.value = scenario.current_do.toFixed(2);

        lagInputIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && scenario[id] !== undefined) {
                el.value = Number(scenario[id]).toFixed(2);
            }
        });

        renderDoChart();
    }

    function resetForm() {
        pondInputForm.reset();
        hideValidationAlert();
        hideResults();
        renderDoChart();
    }

    function hideValidationAlert() {
        if (validationAlert) {
            validationAlert.classList.add("hidden");
        }
    }

    function showValidationAlert(message) {
        if (validationAlert && validationAlertMessage) {
            validationAlertMessage.textContent = message;
            validationAlert.classList.remove("hidden");
        }
    }

    function hideResults() {
        if (placeholderPrediction) placeholderPrediction.classList.remove("hidden");
        if (resultContent) resultContent.classList.add("hidden");
        if (placeholderExplanation) placeholderExplanation.classList.remove("hidden");
        if (explanationContent) explanationContent.classList.add("hidden");
    }

    // -------------------------------------------------------------------------
    // 2-Hour DO Trajectory Chart (Pure SVG)
    // -------------------------------------------------------------------------
    function renderDoChart() {
        if (!doChartSvg) return;

        // Collect 9 readings: T-120 to T
        const trajectoryPoints = [
            { label: "T-120", time: -120, val: parseFloat(document.getElementById("do_t_minus_120")?.value) },
            { label: "T-105", time: -105, val: parseFloat(document.getElementById("do_t_minus_105")?.value) },
            { label: "T-90",  time: -90,  val: parseFloat(document.getElementById("do_t_minus_90")?.value) },
            { label: "T-75",  time: -75,  val: parseFloat(document.getElementById("do_t_minus_75")?.value) },
            { label: "T-60",  time: -60,  val: parseFloat(document.getElementById("do_t_minus_60")?.value) },
            { label: "T-45",  time: -45,  val: parseFloat(document.getElementById("do_t_minus_45")?.value) },
            { label: "T-30",  time: -30,  val: parseFloat(document.getElementById("do_t_minus_30")?.value) },
            { label: "T-15",  time: -15,  val: parseFloat(document.getElementById("do_t_minus_15")?.value) },
            { label: "T (Now)", time: 0,  val: parseFloat(currentDoInput?.value) }
        ];

        // Chart dimensions
        const width = 620;
        const height = 260;
        const padLeft = 45;
        const padRight = 35;
        const padTop = 25;
        const padBottom = 40;

        const plotWidth = width - padLeft - padRight;
        const plotHeight = height - padTop - padBottom;

        // Determine Y bounds
        const validVals = trajectoryPoints.map(p => isNaN(p.val) ? 0 : p.val);
        const minVal = Math.min(...validVals, 0);
        const maxVal = Math.max(...validVals, 6.0);
        const yMax = Math.ceil(maxVal + 1.0);
        const yMin = Math.max(0, Math.floor(minVal - 0.5));

        const scaleX = (idx) => padLeft + (idx / (trajectoryPoints.length - 1)) * plotWidth;
        const scaleY = (val) => padTop + plotHeight - ((val - yMin) / (yMax - yMin)) * plotHeight;

        let svgHtml = "";

        // Background grid lines and Y-axis labels
        const yStep = yMax - yMin > 8 ? 2 : 1;
        for (let y = yMin; y <= yMax; y += yStep) {
            const py = scaleY(y);
            svgHtml += `
                <line x1="${padLeft}" y1="${py}" x2="${width - padRight}" y2="${py}" stroke="#e2e8f0" stroke-width="1" />
                <text x="${padLeft - 8}" y="${py + 4}" fill="#64748b" font-size="11" text-anchor="end" font-family="sans-serif">${y.toFixed(1)}</text>
            `;
        }

        // Horizontal Reference Line for 3.0 mg/L Threshold
        const refY = scaleY(OPERATIONAL_DO_THRESHOLD);
        svgHtml += `
            <line x1="${padLeft}" y1="${refY}" x2="${width - padRight}" y2="${refY}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4" />
            <rect x="${width - padRight - 165}" y="${refY - 18}" width="165" height="16" fill="#fee2e2" rx="3" stroke="#fca5a5" stroke-width="1" />
            <text x="${width - padRight - 8}" y="${refY - 6}" fill="#991b1b" font-size="9.5" font-weight="700" text-anchor="end" font-family="sans-serif">3.0 mg/L (Provisional Threshold)</text>
        `;

        // X-axis baseline and tick labels
        svgHtml += `<line x1="${padLeft}" y1="${scaleY(yMin)}" x2="${width - padRight}" y2="${scaleY(yMin)}" stroke="#94a3b8" stroke-width="1.5" />`;

        trajectoryPoints.forEach((pt, idx) => {
            const px = scaleX(idx);
            svgHtml += `
                <line x1="${px}" y1="${scaleY(yMin)}" x2="${px}" y2="${scaleY(yMin) + 5}" stroke="#94a3b8" stroke-width="1.5" />
                <text x="${px}" y="${scaleY(yMin) + 18}" fill="#475569" font-size="10.5" font-weight="500" text-anchor="middle" font-family="sans-serif">${pt.label}</text>
            `;
        });

        // Trajectory line and point dots
        const validCoords = trajectoryPoints
            .filter(p => !isNaN(p.val))
            .map((p, idx) => ({ x: scaleX(idx), y: scaleY(p.val), val: p.val, isCurrent: idx === trajectoryPoints.length - 1 }));

        if (validCoords.length > 1) {
            const pathD = validCoords.reduce((acc, pt, i) => `${acc} ${i === 0 ? "M" : "L"} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`, "");
            const currentVal = validCoords[validCoords.length - 1].val;
            const lineColor = currentVal < OPERATIONAL_DO_THRESHOLD ? "#dc2626" : "#0284c7";

            svgHtml += `<path d="${pathD}" fill="none" stroke="${lineColor}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />`;

            // Draw data points with values
            validCoords.forEach(pt => {
                const dotColor = pt.val < OPERATIONAL_DO_THRESHOLD ? "#dc2626" : (pt.isCurrent ? "#0284c7" : "#0ea5e9");
                const r = pt.isCurrent ? 5.5 : 4;
                svgHtml += `
                    <circle cx="${pt.x.toFixed(1)}" cy="${pt.y.toFixed(1)}" r="${r}" fill="${dotColor}" stroke="#ffffff" stroke-width="1.5" />
                    <text x="${pt.x.toFixed(1)}" y="${(pt.y - 8).toFixed(1)}" fill="#0f172a" font-size="10" font-weight="700" text-anchor="middle" font-family="sans-serif">${pt.val.toFixed(2)}</text>
                `;
            });
        }

        // Y-axis title
        svgHtml += `
            <text x="14" y="${padTop + plotHeight / 2}" fill="#475569" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90, 14, ${padTop + plotHeight / 2})" font-family="sans-serif">
                Dissolved Oxygen (mg/L)
            </text>
        `;

        doChartSvg.innerHTML = svgHtml;
    }

    // -------------------------------------------------------------------------
    // Form Validation & Submission
    // -------------------------------------------------------------------------
    async function handleFormSubmit(e) {
        e.preventDefault();
        hideValidationAlert();

        const pondId = pondIdInput.value.trim();
        const predictionTimestamp = predictionTimestampInput.value.trim();
        const currentDo = parseFloat(currentDoInput.value);

        // 1. Check required metadata
        if (!pondId) {
            showValidationAlert("Please enter a valid Pond Identifier (e.g., ara2_0677080b).");
            pondIdInput.focus();
            return;
        }

        if (!predictionTimestamp) {
            showValidationAlert("Please enter an ISO prediction timestamp (e.g., 2026-01-26T10:15:00).");
            predictionTimestampInput.focus();
            return;
        }

        // 2. Validate current DO
        if (isNaN(currentDo)) {
            showValidationAlert("Current DO must be a valid numeric value.");
            currentDoInput.focus();
            return;
        }

        if (currentDo < 0.0) {
            showValidationAlert("Current DO cannot be negative.");
            currentDoInput.focus();
            return;
        }

        // 3. Operational boundary check: current DO must be >= 3.0 mg/L
        if (currentDo < OPERATIONAL_DO_THRESHOLD) {
            showValidationAlert(
                `Operational Boundary Condition Violated: Current DO is already ${currentDo.toFixed(2)} mg/L (below the ${OPERATIONAL_DO_THRESHOLD.toFixed(1)} mg/L threshold). ` +
                `The pond is already experiencing hypoxia. ShinerAI operates only when current DO is at or above ${OPERATIONAL_DO_THRESHOLD.toFixed(1)} mg/L ` +
                `to predict impending low-DO risk within the next 2 hours.`
            );
            currentDoInput.focus();
            return;
        }

        // 4. Validate all 8 lag inputs
        const lags = {};
        for (const id of lagInputIds) {
            const el = document.getElementById(id);
            const val = parseFloat(el?.value);
            if (isNaN(val)) {
                showValidationAlert(`Please provide a valid numeric DO reading for ${id.replace(/_/g, " ")}.`);
                if (el) el.focus();
                return;
            }
            if (val < 0.0) {
                showValidationAlert(`Historical reading for ${id.replace(/_/g, " ")} cannot be negative.`);
                if (el) el.focus();
                return;
            }
            lags[id] = val;
        }

        // 5. Construct payload adhering strictly to Flask API schema
        const payload = {
            pond_id: pondId,
            prediction_timestamp: predictionTimestamp,
            current_do: currentDo,
            ...lags
        };

        // Execute prediction workflow
        await executePredictionFlow(payload);
    }

    // -------------------------------------------------------------------------
    // Prediction & Explanation Workflow
    // -------------------------------------------------------------------------
    async function executePredictionFlow(payload) {
        // UI Loading State for Prediction
        btnAnalyze.disabled = true;
        if (placeholderPrediction) placeholderPrediction.classList.add("hidden");
        if (resultContent) resultContent.classList.add("hidden");
        if (loadingPrediction) loadingPrediction.classList.remove("hidden");

        // UI Loading State for Explanation
        if (placeholderExplanation) placeholderExplanation.classList.add("hidden");
        if (explanationContent) explanationContent.classList.add("hidden");
        if (loadingExplanation) loadingExplanation.classList.remove("hidden");

        let predictionResult = null;

        // Step 1: POST /predict
        try {
            const predResponse = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const predData = await predResponse.json();

            if (!predResponse.ok) {
                const errorMsg = predData.message || "An error occurred while computing the prediction.";
                showValidationAlert(`API Error (${predResponse.status}): ${errorMsg}`);
                hideResults();
                btnAnalyze.disabled = false;
                loadingPrediction.classList.add("hidden");
                loadingExplanation.classList.add("hidden");
                return;
            }

            predictionResult = predData;
            renderPredictionResult(predictionResult);
        } catch (err) {
            showValidationAlert("Unable to connect to ShinerAI backend. Please ensure the Flask server is running.");
            hideResults();
            btnAnalyze.disabled = false;
            loadingPrediction.classList.add("hidden");
            loadingExplanation.classList.add("hidden");
            return;
        } finally {
            loadingPrediction.classList.add("hidden");
            btnAnalyze.disabled = false;
        }

        // Step 2: POST /explain
        try {
            const explainResponse = await fetch(`${API_BASE}/explain`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const explainData = await explainResponse.json();

            if (!explainResponse.ok) {
                console.warn("Could not retrieve SHAP explanations:", explainData);
                if (placeholderExplanation) {
                    placeholderExplanation.innerHTML = `<p class="placeholder-sub text-muted">Explanations unavailable: ${explainData.message || "Endpoint error"}</p>`;
                    placeholderExplanation.classList.remove("hidden");
                }
            } else {
                renderExplanation(explainData);
            }
        } catch (err) {
            console.warn("Explanation network failure:", err);
            if (placeholderExplanation) {
                placeholderExplanation.innerHTML = `<p class="placeholder-sub text-muted">Unable to reach explanation service.</p>`;
                placeholderExplanation.classList.remove("hidden");
            }
        } finally {
            loadingExplanation.classList.add("hidden");
        }
    }

    // -------------------------------------------------------------------------
    // Render Prediction Result
    // -------------------------------------------------------------------------
    function renderPredictionResult(res) {
        if (!resultContent) return;

        const isAtRisk = res.predicted_label === "AT_RISK" || res.binary_prediction === 1;
        const probPct = (res.risk_probability * 100).toFixed(1);

        // Visual Treatment
        if (isAtRisk) {
            riskBanner.className = "risk-banner banner-risk";
            riskIcon.textContent = "⚠️";
            riskLabel.textContent = "AT_RISK";
            riskMeterFill.className = "meter-bar-fill meter-risk-fill";
            alertMessageBox.className = "alert-message-box alert-box-risk";
            alertMessageText.textContent = res.alert_message || `WARNING: Risk of low DO (< 3.0 mg/L) within 2 hours is elevated (${probPct}%). Immediate inspection or aeration is recommended.`;
        } else {
            riskBanner.className = "risk-banner banner-safe";
            riskIcon.textContent = "✓";
            riskLabel.textContent = "SAFE";
            riskMeterFill.className = "meter-bar-fill meter-safe-fill";
            alertMessageBox.className = "alert-message-box alert-box-safe";
            alertMessageText.textContent = res.alert_message || `NORMAL: Pond expected to remain safe (>= 3.0 mg/L DO) over the next 2 hours (estimated risk: ${probPct}%).`;
        }

        riskProbabilityValue.textContent = `${probPct}%`;
        riskMeterFill.style.width = `${Math.min(100, Math.max(0, probPct))}%`;

        resCurrentDo.textContent = `${Number(res.current_do).toFixed(2)} mg/L`;
        resTimestamp.textContent = res.prediction_timestamp || "--";
        resModelUsed.textContent = res.model_used || "XGBoost";
        resConfig.textContent = "Config C — DO History Only";

        resultContent.classList.remove("hidden");
    }

    // -------------------------------------------------------------------------
    // Render SHAP Local Explanation
    // -------------------------------------------------------------------------
    function renderExplanation(data) {
        if (!explanationContent) return;

        const contributions = data.feature_contributions || [];

        // Build feature table rows
        shapTableBody.innerHTML = contributions.map(item => {
            const isPushingRisk = item.shap_value > 0;
            const dirBadgeClass = isPushingRisk ? "shap-direction-badge dir-risk" : "shap-direction-badge dir-safe";
            const dirText = isPushingRisk ? "↑ Toward AT_RISK" : "↓ Toward SAFE";
            const humanName = FEATURE_LABELS[item.feature] || item.feature;
            const sign = item.shap_value > 0 ? "+" : "";

            return `
                <tr>
                    <td>
                        <span class="feat-name-main">${humanName}</span>
                        <span class="feat-name-tech">${item.feature}</span>
                    </td>
                    <td>${Number(item.value).toFixed(2)}</td>
                    <td><strong>${sign}${Number(item.shap_value).toFixed(4)}</strong></td>
                    <td><span class="${dirBadgeClass}">${dirText}</span></td>
                </tr>
            `;
        }).join("");

        // Top Risk Drivers
        const riskDrivers = data.top_risk_drivers || [];
        if (topRiskDriversList) {
            if (riskDrivers.length === 0) {
                topRiskDriversList.innerHTML = "<li>None (No features shifted risk positively)</li>";
            } else {
                topRiskDriversList.innerHTML = riskDrivers.map(d => {
                    const name = FEATURE_LABELS[d.feature] || d.feature;
                    return `<li><strong>${name}</strong>: +${Number(d.shap_value).toFixed(4)}</li>`;
                }).join("");
            }
        }

        // Top Safe Drivers
        const safeDrivers = data.top_safe_drivers || [];
        if (topSafeDriversList) {
            if (safeDrivers.length === 0) {
                topSafeDriversList.innerHTML = "<li>None (No features shifted risk negatively)</li>";
            } else {
                topSafeDriversList.innerHTML = safeDrivers.map(d => {
                    const name = FEATURE_LABELS[d.feature] || d.feature;
                    return `<li><strong>${name}</strong>: ${Number(d.shap_value).toFixed(4)}</li>`;
                }).join("");
            }
        }

        explanationContent.classList.remove("hidden");
    }

    // Run initialization
    init();
});

# Phase 5 Beginner's Guide: Understanding the ShinerAI Dashboard & Frontend Integration

Welcome to the **Phase 5 Beginner's Guide**! If this is your first AI/ML or web engineering project, this guide will walk you step-by-step through how the dashboard interface connects to our machine learning model running on the Flask server.

---

## 1. Why Doesn't the Browser Load the ML Model Directly?

In machine learning web applications, you might wonder:  
*Why can't we just load `xgboost_config_c.joblib` directly inside the browser using JavaScript?*

There are several vital scientific and engineering reasons:

1. **Model Format & Runtime Compatibility:**  
   Our machine learning models are trained using Python libraries (`scikit-learn` and `xgboost`) and saved as serialized binary `.joblib` files. Web browsers execute JavaScript, which cannot natively execute compiled C++/Python gradient boosting trees.
2. **Computational Footprint:**  
   Calculating local SHAP values requires generating hundreds of tree traversals with `shap.TreeExplainer`. Running this directly on low-power client devices (like a farmer's smartphone or tablet) would drain battery and freeze the browser.
3. **Security & Intellectual Property:**  
   Serving predictions via an API keeps the model artifacts, training parameters, and proprietary weights securely protected on the server rather than downloading them to the client.
4. **Separation of Concerns:**  
   The browser handles user interaction, rendering, and accessibility. The backend handles data validation, inference, and explainability.

---

## 2. How Does the Frontend Talk to Flask?

The browser and the Flask backend communicate using the **HTTP/REST protocol** through asynchronous JavaScript `fetch()` calls.

```
+------------------------------------+                    +------------------------------------+
|            WEB BROWSER             |                    |            FLASK SERVER            |
|       (HTML, CSS, JavaScript)      |                    |              (Python)              |
+------------------------------------+                    +------------------------------------+
|                                    |                    |                                    |
|  1. User enters DO readings        |                    |                                    |
|                                    |                    |                                    |
|  2. Client validates inputs        |                    |                                    |
|     (e.g., current_do >= 3.0)      |                    |                                    |
|                                    |                    |                                    |
|  3. Sends HTTP POST with JSON      |  --- POST JSON --> |  4. Flask validates payload        |
|     to /predict                    |                    |     derives hour & minute of day   |
|                                    |                    |                                    |
|                                    |                    |  5. Model computes risk score      |
|                                    |                    |                                    |
|  7. Receives risk & displays state |  <-- 200 OK ------ |  6. Returns prediction JSON        |
|     (SAFE or AT_RISK)              |                    |                                    |
|                                    |                    |                                    |
|  8. Sends HTTP POST with JSON      |  --- POST JSON --> |  9. TreeExplainer computes SHAP    |
|     to /explain                    |                    |                                    |
|                                    |                    |                                    |
| 11. Formats feature contribution   |  <-- 200 OK ------ | 10. Returns explanation JSON       |
|     table & direction badges       |                    |                                    |
+------------------------------------+                    +------------------------------------+
```

---

## 3. What JSON Request Is Sent?

When you click **"Analyze Risk"**, the frontend bundles the form values into a clean, structured JSON object:

```json
{
  "pond_id": "ara2_0677080b",
  "prediction_timestamp": "2026-01-26T10:15:00",
  "current_do": 5.40,
  "do_t_minus_15": 5.15,
  "do_t_minus_30": 4.87,
  "do_t_minus_45": 4.36,
  "do_t_minus_60": 4.21,
  "do_t_minus_75": 3.89,
  "do_t_minus_90": 3.72,
  "do_t_minus_105": 3.23,
  "do_t_minus_120": 2.92
}
```

### Notice What Is NOT in the Payload:
The user is **not** asked for `hour_of_day` or `minute_of_day`. Flask's validation module automatically parses the ISO timestamp (`2026-01-26T10:15:00`) and extracts:
- `hour_of_day = 10.0`
- `minute_of_day = 10 * 60 + 15 = 615.0`

This prevents user error and ensures consistency with the Phase 2 training pipeline.

---

## 4. How the Prediction Is Displayed

When Flask responds to `POST /predict`, it returns:
```json
{
  "predicted_label": "SAFE",
  "risk_probability": 0.0667,
  "binary_prediction": 0,
  "warning_issued": false,
  "alert_message": "NORMAL: Pond 'ara2_0677080b' is expected to remain safe (>= 3.0 mg/L DO) over the next 2 hours..."
}
```

The browser converts this response into clear visual indicators:
1. **Status Banner:**  
   - If `predicted_label === "SAFE"`: Green background, checkmark icon (`✓`), label `SAFE`.
   - If `predicted_label === "AT_RISK"`: Crimson background, warning icon (`⚠️`), label `AT_RISK`.
2. **Probability Percentage:**  
   The raw float `0.0667` is multiplied by 100 and formatted as `6.7%`.
3. **Decision Meter Bar:**  
   A visual bar from 0% to 100% shows where the estimated risk falls relative to the fixed **50.0% Decision Threshold**.
4. **Operator Advisory Box:**  
   Presents clear operational guidance without requiring the user to interpret raw probability numbers.

---

## 5. How the SHAP Explanation Is Displayed

After the prediction finishes, the frontend immediately calls `POST /explain` with the same inputs.

Flask uses `shap.TreeExplainer` to calculate how much each feature increased or decreased the risk score:
```json
{
  "base_value": -0.0039,
  "feature_contributions": [
    {
      "feature": "minute_of_day",
      "value": 615.0,
      "shap_value": -1.1596,
      "direction": "toward_SAFE"
    },
    {
      "feature": "current_do",
      "value": 5.40,
      "shap_value": -0.6279,
      "direction": "toward_SAFE"
    }
  ],
  "top_safe_drivers": [...],
  "top_risk_drivers": [...]
}
```

The frontend renders this data into an intuitive table:
- **Feature Name:** Technical names like `minute_of_day` are mapped to human-readable names like `Minute of Day (Time of Day)`.
- **Direction Badges:**  
  - If $\text{SHAP} > 0$: Displayed as `↑ Toward AT_RISK` in a soft red badge.
  - If $\text{SHAP} < 0$: Displayed as `↓ Toward SAFE` in a soft emerald badge.
- **Top Drivers Summary Cards:**  
  Highlights the top 3 features pushing risk upward and top 3 features pulling risk downward.

---

## 6. How the 2-Hour DO Trajectory Chart Works

Rather than importing a massive charting library like Chart.js or D3.js (which would add dozens of files and megabytes of code), ShinerAI uses a **lightweight, pure SVG chart generator**:

1. **Data Collection:**  
   Reads the 9 historical readings ($T-120\text{m}, T-105\text{m}, \dots, T$) from the input inputs.
2. **Coordinate Scaling:**  
   Maps time $[-120\text{m} \dots 0\text{m}]$ to horizontal pixel coordinates $[45\text{px} \dots 585\text{px}]$, and dissolved oxygen $[0.0 \dots 7.0\text{ mg/L}]$ to vertical pixel coordinates.
3. **Path Construction:**  
   Generates an SVG `<path d="M x1 y1 L x2 y2 ...">` that smoothly traces the pond's recent history.
4. **Provisional Reference Line:**  
   Draws a prominent horizontal dashed red line at exactly $3.0\text{ mg/L}$ with a badge reading:
   `3.0 mg/L (Provisional Threshold)`.
5. **Dynamic Updates:**  
   Whenever you edit a number or click a demo scenario button, the SVG recalculates and re-renders instantly!

---

## 7. Operational Boundary Checks: Why Current DO < 3.0 Is Blocked

In aquaculture, dissolved oxygen below $3.0\text{ mg/L}$ is already considered a hypoxic emergency.

If a pond's current DO is $2.6\text{ mg/L}$, asking the model:  
*"Will this pond experience hypoxia in the next 2 hours?"*  
is biologically and logically redundant—it is **already** hypoxic!

Therefore, ShinerAI enforces a strict boundary rule:
- **Client-Side:** If `current_do < 3.0`, the frontend blocks the submission and displays:
  > *"Operational Boundary Condition Violated: Current DO is already below 3.0 mg/L. ShinerAI operates only when current DO is at or above 3.0 mg/L to predict low-DO risk within the next 2 hours."*
- **Server-Side:** If an external script bypasses the client and sends `current_do < 3.0`, Flask immediately rejects the payload with `400 Bad Request` and `status: ALREADY_LOW_DO`.

---

## 8. How the Interactive Aquarium Background Works

To make ShinerAI feel deeply connected to aquaculture while maintaining strict scientific readability, the dashboard includes a custom procedural Canvas 2D simulation (`frontend/aquarium.js`):

1. **Why Canvas 2D Instead of Video or WebGL?**  
   - Large video loops consume 50–100 MB of bandwidth, repeat visibly, and cannot interact with the mouse.
   - Heavy 3D engines (like Three.js) would add megabytes of JavaScript dependencies.
   - Canvas 2D provides smooth 60fps organic motion, zero external dependencies, and instant load time (< 41 KB of pure JavaScript).

2. **How Do the Goldfish Move?**  
   - **Anatomical Spine Undulation:** Each fish computes a traveling sinusoidal wave along 5 spine segments:
     $$\text{offset}_i = \text{amplitude} \cdot \sin(\text{phase} - i \cdot 0.5)$$
     This creates realistic body bends and causes the translucent tail fin to ripple gently behind it.
   - **Schooling Flocking Dynamics:** Fish are grouped into cohorts. Each fish computes steering forces toward its group centroid (cohesion), away from immediate neighbors (separation), and in the general group heading (alignment).
   - **Interactive Group Escape Physics:** The canvas tracks cursor movement. When the mouse approaches within 180–260 pixels, nearby fish sense the disturbance, accelerate smoothly away with organic scatter, and gradually decelerate back to calm slow swimming once the cursor departs.

3. **Ensuring 100% Text Readability (WCAG Compliance):**  
   - Background canvas has `pointer-events: none` and `aria-hidden="true"` so it never blocks clicks or interferes with screen readers.
   - All dashboard cards use **genuine glassmorphism** with a refined translucent gradient (`rgba(255, 255, 255, 0.22)` to `0.15`), `backdrop-filter: blur(22px) saturate(115%)`, subtle readability veil (`rgba(235, 248, 255, 0.08)`), and white inner borders (`rgba(255, 255, 255, 0.30)`).
   - Deep ink primary text (`#071522`) and secondary text (`#1f3b50`) with soft white micro-shadows (`text-shadow: 0 1px 1px rgba(255, 255, 255, 0.45)`) guarantee crisp, high-contrast readability across all cards, allowing goldfish, swaying plants, and rising bubbles to remain clearly visible *through* the cards without visual clutter.
   - Automatic CPU throttling: when you switch browser tabs, the Page Visibility API (`document.hidden`) pauses the render loop, reducing battery and CPU usage to zero!

---

## 9. Summary of What You Learned in Phase 5

1. **API-First Architecture:** Clean decoupling between frontend presentation and backend ML inference.
2. **Input Normalization:** Translating raw user inputs into canonical feature vectors expected by the model.
3. **Operational Guardrails:** Preventing nonsensical model predictions when the boundary condition is already breached.
4. **Interpretable AI:** Rendering both global model importance and single-instance local SHAP attributions so end users understand why a prediction was made.
5. **Scientific Responsibility:** Distinguishing water quality risk from fish mortality and presenting machine-learning outputs with proper caveats.
6. **Accessible Aesthetics:** Adding a rich, interactive procedural aquarium theme that enhances domain engagement while guaranteeing strict WCAG text contrast and zero CPU drain when idle.

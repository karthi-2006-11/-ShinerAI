# ShinerAI: Phase 3 Beginner's Guide to Machine Learning

Welcome to Phase 3 of **ShinerAI**! If you are new to Artificial Intelligence and Machine Learning (AI/ML), this guide breaks down the core concepts used to train and evaluate our early-warning system in clear, intuitive terms.

---

## 1. What is X and what is y?

In machine learning, mathematical notation is commonly used to describe data:
- **$X$ (The Feature Matrix / Inputs):** This is the information the machine learning algorithm is given to learn from or make a prediction about. In ShinerAI, $X$ is a table containing recent sensor measurements: the pond's current Dissolved Oxygen (DO), pH, temperature, historical readings over the past 2 hours ($t-15\text{m}, t-30\text{m}, \dots, t-120\text{m}$), and the time of day.
- **$y$ (The Target / Label / Output):** This is the outcome we want the model to predict. In ShinerAI, $y$ is binary ($0$ or $1$):
  - $y = 0$ (`SAFE`): Dissolved oxygen remains healthy ($\ge 3.0\text{ mg/L}$) throughout the entire next 2 hours.
  - $y = 1$ (`AT_RISK`): Dissolved oxygen drops below $3.0\text{ mg/L}$ at any point in the next 2 hours.

*Note: The target label captures water oxygen depletion ($\text{DO} < 3.0\text{ mg/L}$); it does not directly predict fish disease or fish mortality.*

---

## 2. What is a Feature? What is a Target?

- **Feature:** An individual measurable property or characteristic of a pond at prediction time $T$. For example, `current_do = 4.2 mg/L` and `do_t_minus_15 = 4.8 mg/L` are features.
- **Target:** The ground-truth event that occurs in the future window $(T, T + 2\text{ hours}]$. In supervised learning, we give the model pairs of $(X, y)$ so it learns the statistical relationship connecting current and recent conditions to future outcomes.

---

## 3. What is Classification?

**Classification** is a category of machine learning where the computer's job is to assign an input example into one of a discrete set of categories (classes).
- Because ShinerAI has exactly two categories (`SAFE` and `AT_RISK`), this is a **binary classification** task.
- If we were trying to predict the exact numerical DO concentration 2 hours later (e.g., $2.4\text{ mg/L}$), that would be **regression**. We chose classification because fish farmers need an actionable operational decision rule: *"Should I turn on the paddlewheel aerators now, yes or no?"*

---

## 4. What does Training mean? What does Testing mean?

- **Training:** The learning phase. The machine learning algorithm inspects thousands of historical pond examples where both the inputs ($X$) and the true future outcome ($y$) are known. It adjusts its internal mathematical parameters (coefficients or decision split thresholds) to minimize prediction errors.
- **Testing:** The evaluation phase. We take a completely separate set of examples that the model was **never allowed to see during training**. We ask the model to predict the future risk based only on $X$, and then we compare its predictions against the true $y$ to measure real-world performance.

---

## 5. What is a Baseline?

A **baseline** is a simple, intuitive benchmark model used to set the minimum standard of performance.
- Any sophisticated machine learning model (like Random Forest or XGBoost) is only useful if it proves to be significantly better than a simple common-sense rule.
- In ShinerAI, we evaluated two distinct baselines:
  1. **Majority Baseline:** A "naive" model that always predicts `SAFE (0)` for every pond, because 87.5% of the dataset is safe.
  2. **Current-DO Baselines:**
     - **Current-DO Threshold Baseline (DO <= 4.2 mg/L):** A direct Boolean decision rule derived strictly from training data optimization (threshold 4.2 mg/L maximized training F1-score; zero test data used). On the holdout test set, it achieves Recall = 70.73%, Precision = 56.10%, Specificity = 92.87%, and F1 = 0.6257 (catching 667 low-DO events with 522 false alarms).
     - **Current-DO Ranking Baseline (1-D Logistic):** A continuous 1-D model evaluating the rank-ordering capability of instantaneous DO across all thresholds (PR-AUC = 0.6149, ROC-AUC = 0.9024). At its default balanced threshold ($p=0.50$, alerting whenever DO <= 5.93 mg/L), it achieves Recall = 89.61%, Precision = 30.19%, Specificity = 73.30%, and F1 = 0.4516 (catching 845 events with 1,954 false alarms).

---

## 6. Why Can't We Randomly Split Time-Series Data?

In standard tabular machine learning (like predicting house prices), textbooks often teach you to shuffle rows randomly using `train_test_split()`.
**In environmental time series, randomly shuffling rows is a serious methodological error.**
Here is why:
- Pond sensors take readings every 15 minutes.
- If row 100 (at 2:00 AM) is randomly put into the training set, and row 101 (at 2:15 AM) is put into the test set, row 100 and row 101 share almost identical water temperatures, pH, and historical readings.
- The model would simply memorize the water condition at 2:00 AM to "predict" 2:15 AM.
- This creates artificially inflated test scores that fail when deployed on tomorrow's unobserved data.

---

## 7. What is Data Leakage? What is a Purge Gap?

- **Data Leakage:** When information from the future or from the test set accidentally leaks into the training process.
- **The Purge Gap (Embargo):** Because our target label evaluates what happens over the **next 2 hours**, consider a training example recorded 15 minutes before the test period begins. Its target label looks 2 hours into the future—meaning it reads data that belongs to the test period!
- To prevent this leakage, ShinerAI enforces a **2-hour purge gap**: we discard all training records within 2 hours of the split boundary (108 rows). That way, the training labels strictly complete before the test set begins.

---

## 8. Why is Accuracy Misleading for Our Class Distribution?

Our dataset has an imbalanced class distribution:
- **SAFE (0):** 87.46% (36,101 rows)
- **AT_RISK (1):** 12.54% (5,176 rows)

If a lazy model simply outputs `SAFE` for 100% of all queries, its **Accuracy is 88.58%**!
To a non-technical manager, an "88% accurate model" sounds impressive. But in reality:
- It caught **0 out of 943 low-DO events** (0% Recall).
- Every single impending low-DO event passed without warning.
This is why **Accuracy is the wrong metric** for early-warning systems.

---

## 9. What are Precision, Recall, and F1-Score?

For an early-warning system:
- **True Positive (TP):** The model warned of low DO, and low DO actually occurred. (Success: early warning issued before hypoxia).
- **False Positive (FP):** The model warned of low DO, but the pond stayed healthy. (False alarm: alert triggered when water remained safe, leading to unnecessary inspection or intervention).
- **False Negative (FN):** The model said safe, but the pond crashed below 3.0 mg/L. (Missed event: impending low-DO occurred without alert).
- **True Negative (TN):** The model said safe, and the pond stayed safe. (Normal operation).

From these definitions:
- **Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$. What percentage of all actual low-DO events did our model catch? (High recall = few missed events).
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$. When the alarm sounds, what percentage of the time is it real? (High precision = few false alarms).
- **Specificity:** $\frac{\text{TN}}{\text{TN} + \text{FP}}$. What percentage of safe situations were correctly identified as safe? (High specificity = low false alarm rate).
- **F1-Score:** The harmonic mean of Precision and Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

---

## 10. What is PR-AUC? Why is it Our Primary Metric?

- **ROC-AUC:** Measures the trade-off between True Positive Rate and False Positive Rate across all possible classification probability thresholds. While standard, ROC-AUC can look deceptively high on imbalanced datasets because the large number of True Negatives dilutes the False Positive Rate.
- **PR-AUC (Precision-Recall Area Under Curve):** Plots Precision against Recall for all possible thresholds. In imbalanced early-warning problems, **PR-AUC is the primary benchmark metric** because it focuses entirely on the minority class (`AT_RISK`).
  - A no-skill baseline achieves a PR-AUC equal to the test positive prevalence: **0.1142**.
  - ShinerAI's trained ML models achieve PR-AUC of **0.735 to 0.757**—nearly **7 times higher than random chance**!

---

## 11. What is a Confusion Matrix?

A confusion matrix is a simple $2 \times 2$ grid that displays the exact counts of True Negatives, False Positives, False Negatives, and True Positives.
For our temporal test set (8,261 total examples, with 943 low-DO events):
- **Current-DO Threshold Baseline (DO <= 4.2 mg/L):** Caught **667 low-DO events** (70.7% recall) with 522 false alarms.
- **Current-DO Ranking Baseline (1-D Logistic at $p=0.50$):** Caught **845 low-DO events** (89.6% recall) with 1,954 false alarms.
- **Random Forest (Config B):** Caught **718 low-DO events** (76.1% recall) with 643 false alarms.
- **Random Forest (Config C):** Caught **713 low-DO events** (75.6% recall) with only 504 false alarms (lowest among ML models).
- **XGBoost (Config B):** Caught **748 low-DO events** (79.3% recall) with 835 false alarms.
- **XGBoost (Config C):** Caught **752 low-DO events** (79.8% recall) with 698 false alarms.
- **Logistic Regression (Config B):** Caught **848 low-DO events** (89.9% recall) but produced 2,158 false alarms (overly sensitive alert).

---

## 12. Understanding Model Selection & Trade-offs

We evaluated three classical model families across three feature sets. No single model is universally "best"; each represents a distinct operational trade-off:
1. **XGBoost (Config C - DO History Only):**
   - **Highest overall PR-AUC (0.7574).**
   - Excellent all-around ranking quality, catching 79.8% of low-DO events with 698 false alarms.
2. **Random Forest (Config C - DO History Only):**
   - **Highest F1-score (0.6602) and Specificity (93.11%).**
   - Strongest at minimizing false alarms (504 false positives) at the standard $p=0.50$ decision threshold, producing fewer false alarms which could reduce unnecessary interventions in a deployment where alerts trigger aeration, while still catching 75.6% of events.
3. **Logistic Regression (Config B - Full History):**
   - **High-Recall alternative (89.93% recall).**
   - Useful when missed events carry severe operational consequences and operators are willing to tolerate frequent false alarms (2,158 false alarms).

**Takeaway:** The recent temporal trajectory of DO provides additional predictive information beyond current measurements alone. A consistent improvement was observed across the three tested model families when historical telemetry was included.

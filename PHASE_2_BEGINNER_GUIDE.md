# Beginner's Guide to Phase 2: Data Cleaning & Label Generation
**Project:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Target Audience:** Beginners in AI/ML and Environmental Data Science  

---

## Welcome to Phase 2!

In Phase 1, we explored the raw dataset and confirmed that low dissolved oxygen ($\text{DO} < 3.0\text{ mg/L}$) happens frequently across all 17 ponds.

In **Phase 2**, we transform that raw sensor stream into a format that a machine learning model can actually learn from. 

If you are new to machine learning and time-series forecasting, this guide explains each core concept using simple analogies.

---

## 1. What is Data Cleaning?

Imagine a digital thermometer that usually shows $26^\circ\text{C}$, but when its battery briefly unseats during a storm, it flashes `0.0` for a second before turning back on. 

If you fed that `0.0` into a machine learning model without cleaning, the model might conclude that the tropical fish pond suddenly froze into a block of ice!

**Data Cleaning** is the disciplined process of:
1. Detecting abnormal sensor glitches (like exact `0.0` readings when power resets).
2. Removing ambiguous records (like two conflicting readings at the exact same minute).
3. Handling gaps (when a probe was uninstalled for a week during fish harvest).
4. Formatting timestamps so time flows naturally.

Data cleaning **never** means deleting data you dislike or fabricating numbers out of thin air. In this project, we kept every raw file untouched and clearly tagged every issue in [`QC_CLEANING_POLICY.md`](file:///D:/FISH/QC_CLEANING_POLICY.md).

---

## 2. What is a Time Series?

In standard machine learning (like predicting house prices), each row is independent: House A's price doesn't care whether House B was sold 15 minutes earlier.

In **Time-Series Data**, every measurement is connected in time:
- At 02:00 AM, DO is $5.0\text{ mg/L}$.
- At 02:15 AM, DO is $4.6\text{ mg/L}$.
- At 02:30 AM, DO is $4.1\text{ mg/L}$.

You can see a **downward trajectory**. The order matters completely! If you shuffle the rows randomly like a deck of cards, the story is lost.

---

## 3. What is a Feature?

A **Feature** is an input clue that the machine learning model looks at when trying to make a guess. 

In our project, at prediction time $T$, the features include:
- What is the dissolved oxygen right now (`current_do` or `do_t`)?
- What was the oxygen 15 minutes ago (`do_t_minus_15`)?
- What was the oxygen 30, 45, 60, ..., 120 minutes ago?
- What is the water temperature and pH right now and over the last 2 hours?
- What time of day is it (`hour_of_day`: e.g. 03:00 AM vs 03:00 PM)?

These features describe the **current trajectory** of the pond.

---

## 4. What is a Label (Target)?

A **Label** is the correct answer to the question we want the AI to answer. During training, the AI looks at the features, makes a guess, and compares its guess against the label to learn from its mistakes.

In our project, the question is:
> *"Will this pond drop into hypoxia ($\text{DO} < 3.0\text{ mg/L}$) during the next 2 hours?"*

- If YES: **`target = 1` (`AT_RISK`)** — An actual drop below $3.0\text{ mg/L}$ was observed within the next 2 hours.
- If NO: **`target = 0` (`SAFE`)** — The sensor stayed online and actively monitored the pond for the **complete 2-hour period** (~120 minutes, $\ge 8$ readings), confirming DO never dropped below $3.0\text{ mg/L}$.
- If the sensor stopped recording early without seeing a drop, we **cannot guarantee** the pond stayed safe during the unmonitored minutes. We classify it as `insufficient_future_coverage` and exclude it rather than guessing.

---

## 5. What is a Prediction Window?

A **Prediction Window** is the future time interval we want our forecast to cover.

```text
[  Past 2 Hours History  ] -> [ Current Time T ] -> [  Future 2-Hour Prediction Window  ]
  (Features: t-120 to t)         (Make prediction)              (Look for DO < 3.0 mg/L)
```

In our system, when it is 03:00 AM:
- We use data from **01:00 AM to 03:00 AM** as the **features**.
- We look at what actually happened between **03:00 AM and 05:00 AM** to create the **target label**.

---

## 6. Why Do We Use the Past to Predict the Future?

In real life on a fish farm, a manager standing beside a pond at 03:00 AM does not know what will happen at 04:30 AM. They can only see what has happened up to 03:00 AM!

Therefore, an AI model deployed in the real world will only have access to **past measurements**. If we allowed the model to see future measurements while training, it would fail completely when deployed in reality.

---

## 7. What is Data Leakage? (And Why is it the Deadliest ML Bug?)

**Data Leakage** happens when information from the future "leaks" into the model's training inputs.

Imagine taking a high-school math exam, but someone accidentally printed the answers on the back of the test paper. You would get an A+ on the exam! But the moment you walk into the real world without the answer sheet, you wouldn't know how to solve the problems.

In our project, we enforced **zero leakage**:
- Future readings from the next 2 hours are used **only** to determine whether the label is 0 or 1.
- No future readings are ever included in the feature columns (`do_t_minus_15`, etc.).
- Automated test scripts ([`tests/test_no_data_leakage.py`](file:///D:/FISH/tests/test_no_data_leakage.py)) verify that every single feature is strictly $\le$ the prediction timestamp.

---

## 8. Why Can't We Label a Row That is Already Below 3.0 mg/L as an Early Warning?

Suppose it is 04:00 AM, and a pond's dissolved oxygen is already **$1.8\text{ mg/L}$** (severe crisis).

If our early warning system beeped at 04:00 AM and said: *"Warning: DO will drop below 3.0 mg/L in the next 2 hours!"*, the fish farmer would say:
> *"The fish are already gasping at the surface! The crisis is already here! You didn't give me an early warning—you gave me a late notification."*

An **Early Warning System** must satisfy:
$$\text{Current DO} \ge 3.0\text{ mg/L} \quad \text{AND} \quad \text{Future DO} < 3.0\text{ mg/L}$$

This ensures the alarm sounds **before** the pond becomes dangerous, giving the farmer 1 to 2 hours of advance notice to turn on mechanical aerators.

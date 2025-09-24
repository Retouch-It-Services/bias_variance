# 📊 In-Depth Guide to Classification Evaluation Metrics in Machine Learning

Classification metrics help you understand how well your model performs beyond simple accuracy. This guide covers **20+ metrics** with explanations, use cases, formulas, and Python implementation examples.

---

## 🔍 Core Concepts Before Metrics
- **Binary Classification**: Two classes (e.g., spam/not spam)
- **Multi-class Classification**: ≥3 classes (e.g., cat/dog/bird)
- **Imbalanced Data**: When one class dominates (e.g., 99% normal transactions, 1% fraud)
- **Probabilistic vs. Hard Predictions**: Some metrics use class probabilities (e.g., log loss), others use hard class labels

---

## 📋 1. Confusion Matrix (The Foundation)
A table showing true vs. predicted labels:

|                | **Predicted Positive** | **Predicted Negative** |
|----------------|------------------------|------------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### 📌 Key Terms:
- **TP**: Correctly identified positives (e.g., fraud detected)
- **FP**: False alarms (e.g., legitimate transaction flagged as fraud)
- **FN**: Missed positives (e.g., fraud missed by model)
- **TN**: Correctly identified negatives (e.g., legitimate transaction)

> 💡 **Why it matters**: All other metrics derive from these 4 values. Always start here!

---

## 🎯 2. Accuracy
**Definition**: Ratio of correct predictions to total predictions  
**Formula**:  
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### ✅ Pros:
- Simple to understand
- Good for balanced datasets

### ❌ Cons:
- **Misleading for imbalanced data** (e.g., 99% negative class → 99% accuracy even if model always predicts negative)
- Doesn't distinguish between types of errors

### 🐍 Python Example:
```python
from sklearn.metrics import accuracy_score
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
print(accuracy_score(y_true, y_pred))  # Output: 0.8
```

---

## 🔍 3. Precision (Positive Predictive Value)
**Definition**: How many selected items are relevant?  
**Formula**:  
$$\text{Precision} = \frac{TP}{TP + FP}$$

### 💡 When to use:
- When **false positives are costly** (e.g., spam detection: false positives annoy users)
- In medical tests where false alarms cause unnecessary procedures

### 📌 Example:
- 100 emails flagged as spam → only 80 are actually spam → Precision = 80%

---

## 📈 4. Recall (Sensitivity, True Positive Rate)
**Definition**: How many relevant items were selected?  
**Formula**:  
$$\text{Recall} = \frac{TP}{TP + FN}$$

### 💡 When to use:
- When **false negatives are costly** (e.g., cancer detection: missing a tumor)
- In fraud detection: missing real fraud is worse than false alarms

### 📌 Example:
- 100 actual spam emails → model catches 70 → Recall = 70%

---

## ⚖️ 5. F1-Score
**Definition**: Harmonic mean of Precision and Recall  
**Formula**:  
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 💡 Why harmonic mean?
- Punishes extreme values (e.g., precision=100%, recall=0% → F1=0)
- Better than arithmetic mean for imbalanced data

### 📌 Use Cases:
- When you need balance between precision and recall
- Common in information retrieval, medical diagnosis

### 🐍 Python Example:
```python
from sklearn.metrics import f1_score
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
print(f1_score(y_true, y_pred))  # Output: 0.666...
```

---

## 📈 6. ROC Curve & AUC (Receiver Operating Characteristic)
### 📊 What is ROC Curve?
- Plots **True Positive Rate (Recall)** vs. **False Positive Rate** at different thresholds
- FPR = $\frac{FP}{FP + TN}$

### 🌟 AUC (Area Under Curve)
- Measures the entire 2D area under ROC curve
- **Range**: 0.5 (random guess) to 1.0 (perfect classifier)

### 💡 When to use:
- **Binary classification with balanced classes**
- When you care about **ranking** predictions (e.g., "this is more likely to be positive than that")

### 📌 Key Insight:
- AUC is **threshold-independent** (unlike accuracy/F1)
- Good for comparing models across different operating points

### 🐍 Python Example:
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_true = [0, 1, 1, 0, 1]
y_probs = [0.1, 0.8, 0.4, 0.2, 0.9]  # Probabilities

fpr, tpr, _ = roc_curve(y_true, y_probs)
auc = roc_auc_score(y_true, y_probs)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

![ROC Curve](https://i.imgur.com/8ZbJf3L.png)

---

## 📉 7. Precision-Recall Curve & AUC
### 📊 What is PR Curve?
- Plots **Precision** vs. **Recall** at different thresholds

### 💡 When to use:
- **Highly imbalanced datasets** (e.g., fraud detection, rare disease diagnosis)
- When you care more about **positive class performance**

### 📌 Why better than ROC for imbalanced data?
- ROC can be misleading when negative class dominates
- PR curve focuses only on positive class behavior

### 🐍 Python Example:
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_true, y_probs)
avg_precision = average_precision_score(y_true, y_probs)

plt.plot(recall, precision, label=f"AP = {avg_precision:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
```

![Precision-Recall Curve](https://i.imgur.com/5mXcR3k.png)

---

## 📉 8. Log Loss (Cross-Entropy Loss)
**Definition**: Measures the **quality of probabilistic predictions**  
**Formula**:  
$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

### 💡 Why it matters:
- Penalizes **overconfident wrong predictions** more severely
- Used in logistic regression, neural networks
- Lower is better (0 = perfect prediction)

### 📌 Key Insight:
- **Not threshold-dependent** (uses raw probabilities)
- More informative than accuracy for probabilistic models

### 🐍 Python Example:
```python
from sklearn.metrics import log_loss

y_true = [0, 1, 1, 0, 1]
y_probs = [0.1, 0.8, 0.4, 0.2, 0.9]

print(log_loss(y_true, y_probs))  # Output: 0.425
```

---

## 🎯 9. Cohen's Kappa
**Definition**: Measures agreement between predictions and actual labels, **correcting for chance**  
**Formula**:  
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
- $p_o$ = observed accuracy
- $p_e$ = expected accuracy by chance

### 💡 When to use:
- **Imbalanced datasets** where accuracy is misleading
- Inter-rater reliability (e.g., medical diagnosis)

### 📌 Example:
- 95% accuracy on imbalanced data → Kappa might be only 0.3 (low agreement)

---

## 📐 10. Matthews Correlation Coefficient (MCC)
**Definition**: Balanced measure for binary classification  
**Formula**:  
$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 💡 Why it's special:
- **Range**: -1 (worst) to +1 (best)
- Works for **imbalanced data**
- Considers all 4 confusion matrix values

### 📌 Use Case:
- Bioinformatics, medical diagnosis where class imbalance is common

---

## 🌐 11. Multi-Class Metrics
For ≥3 classes, metrics extend in three ways:

| Type | Description | When to Use |
|------|-------------|-------------|
| **Micro** | Global calculation across all classes | When classes are equally important |
| **Macro** | Average per-class metrics | When classes are equally important |
| **Weighted** | Weighted average by class support | When classes are imbalanced |

### 🐍 Python Example (F1-Score for 3 classes):
```python
from sklearn.metrics import f1_score

y_true = [0, 1, 2, 2, 2]
y_pred = [0, 1, 1, 2, 2]

# Micro F1
print(f1_score(y_true, y_pred, average='micro'))  # 0.8

# Macro F1
print(f1_score(y_true, y_pred, average='macro'))  # 0.777...

# Weighted F1
print(f1_score(y_true, y_pred, average='weighted'))  # 0.76
```

---

## 📊 12. Specificity (True Negative Rate)
**Definition**: Ability to correctly identify negatives  
**Formula**:  
$$\text{Specificity} = \frac{TN}{TN + FP}$$

### 💡 When to use:
- Medical testing where false positives are critical (e.g., HIV test)
- Security systems where false alarms are costly

---

## 📈 13. Gini Coefficient
**Definition**: Related to AUC: $\text{Gini} = 2 \times \text{AUC} - 1$  
- **Range**: -1 to 1 (but typically 0 to 1)

### 💡 Why it matters:
- Common in finance/risk modeling
- Higher Gini = better model discrimination

---

## 📌 14. Balanced Accuracy
**Definition**: Average of recall for each class  
$$\text{Balanced Accuracy} = \frac{\text{Recall}_1 + \text{Recall}_2 + \dots}{\text{number of classes}}$$

### 💡 When to use:
- Highly imbalanced datasets (e.g., 99% negative class)
- More reliable than overall accuracy

---

## 📊 15. Brier Score
**Definition**: Mean squared difference between predicted probability and actual outcome  
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$

### 💡 When to use:
- Calibration of probabilistic models
- Lower score = better calibration

---

## 📈 16. Hamming Loss
**Definition**: Fraction of incorrect labels (for multi-label classification)  
$$\text{Hamming Loss} = \frac{1}{N \times L} \sum_{i=1}^N \sum_{j=1}^L \mathbb{1}(y_{ij} \neq \hat{y}_{ij})$$

### 💡 When to use:
- Multi-label problems (e.g., image tagging: "cat, dog, car")
- Lower is better (0 = perfect)

---

## 📌 17. Zero-One Loss
**Definition**: Fraction of misclassified instances (same as 1 - accuracy)

### 💡 When to use:
- Simple classification tasks
- Rarely used alone (better to use accuracy)

---

## 📈 18. Hinge Loss
**Definition**: Used in SVMs for classification  
$$\text{Hinge Loss} = \max(0, 1 - y \times \hat{y})$$

### 💡 When to use:
- Support Vector Machines
- When you want margin-based classification

---

## 📊 19. Jaccard Index (Intersection over Union)
**Definition**: Similarity between predicted and actual labels  
$$\text{Jaccard} = \frac{|A \cap B|}{|A \cup B|}$$

### 💡 When to use:
- Semantic segmentation (e.g., image masks)
- Multi-label classification

---

## 📌 20. Kullback-Leibler Divergence
**Definition**: Measures difference between predicted and actual probability distributions  
$$D_{KL}(P \| Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$$

### 💡 When to use:
- Model calibration
- When comparing probability distributions

---

## 🧠 How to Choose the Right Metric

| Scenario | Recommended Metrics |
|----------|---------------------|
| **Balanced dataset** | Accuracy, ROC-AUC |
| **Imbalanced dataset** | Precision-Recall AUC, MCC, F1-Score |
| **Cost of false positives is high** | Precision, Specificity |
| **Cost of false negatives is high** | Recall, Sensitivity |
| **Probabilistic predictions** | Log Loss, Brier Score |
| **Multi-class classification** | Macro/Micro F1, Confusion Matrix |
| **Medical diagnosis** | Sensitivity, Specificity, MCC |
| **Fraud detection** | Precision-Recall AUC, F1-Score |
| **Image segmentation** | Jaccard Index, Dice Coefficient |

---

## 🐍 Python Implementation Cheat Sheet
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Binary classification
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_probs))
print("Log Loss:", log_loss(y_true, y_probs))
print("MCC:", matthews_corrcoef(y_true, y_pred))

# Multi-class
print(classification_report(y_true, y_pred, target_names=classes))

# Confusion Matrix
import seaborn as sns
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Never use accuracy alone for imbalanced data**  
   → Use precision-recall curves or MCC instead

2. **Don't optimize for a single metric**  
   → Use multiple complementary metrics (e.g., F1 + MCC + Log Loss)

3. **Threshold tuning matters**  
   → Adjust decision threshold based on business needs (e.g., higher recall for medical diagnosis)

4. **Use cross-validation for metrics**  
   → Metrics can vary across folds; report mean ± std

5. **Compare to baseline**  
   → Compare to "always predict majority class" or random guessing

6. **Visualize everything**  
   → Confusion matrix, ROC curve, PR curve are more informative than single numbers

---

## 🌟 Real-World Example: Medical Diagnosis

| Metric | Value | Interpretation |
|--------|-------|--------------|
| Accuracy | 95% | Looks good, but... |
| Recall (Sensitivity) | 70% | Missed 30% of actual cancer cases → **dangerous!** |
| Precision | 85% | 15% false positives → unnecessary procedures |
| F1-Score | 77% | Balance between sensitivity and precision |
| MCC | 0.65 | Strong agreement (better than accuracy) |

> 💡 **Conclusion**: For cancer screening, **recall is critical** even if precision is lower. A model with 90% recall (even with 50% precision) is better than one with 95% accuracy but 70% recall.

---

## 🔥 Pro Tips
- **For imbalanced data**: Use **precision-recall curves** instead of ROC curves
- **For probabilistic models**: Always check **log loss** and **Brier score**
- **For multi-class**: Use **macro F1** for equal importance, **weighted F1** for imbalanced classes
- **For medical applications**: Prioritize **sensitivity (recall)** over precision
- **For fraud detection**: Prioritize **precision** to reduce false alarms

> ✅ **Golden Rule**:  
> *"Choose metrics based on business impact, not mathematical convenience."*  
> A metric that aligns with real-world consequences is always better than "standard" metrics.

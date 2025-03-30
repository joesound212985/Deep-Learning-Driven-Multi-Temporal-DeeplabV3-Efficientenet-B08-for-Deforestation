# -*- coding: utf-8 -*-



import math
import matplotlib.pyplot as plt

def fraction_text(numer, denom):
    """
    Return a LaTeX-style fraction string using MathText.
    (Removed \displaystyle for compatibility with mathtext parser.)
    """
    return rf"$\frac{{{numer}}}{{{denom}}}$"

def main():
    # Confusion matrix values (updated to match the matrix in your image)
    TN = 29989750  # True Negatives
    FP = 1110825   # False Positives
    FN = 987374    # False Negatives
    TP = 19554419  # True Positives

    # Calculate basic metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0  # Sensitivity
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0

    # Additional metrics
    # Add small epsilon checks for divisions to avoid ZeroDivisionError
    denom_mcc = math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    mcc = ((TP * TN) - (FP * FN)) / denom_mcc if denom_mcc != 0 else 0

    balanced_accuracy = (recall + specificity) / 2
    informedness = recall + specificity - 1
    npv = TN / (TN + FN) if (TN + FN) != 0 else 0  # Negative Predictive Value
    markedness = precision + npv - 1
    prevalence = (TP + FN) / (TP + TN + FP + FN)
    
    # Diagnostic Odds Ratio (DOR)
    # (TP/FN) / (FP/TN) = (TP * TN) / (FN * FP)
    denom_dor = FN * FP
    dor = (TP * TN) / denom_dor if denom_dor != 0 else float('inf')

    # New imbalance metrics:
    # G-Mean: Geometric mean of Recall and Specificity
    g_mean = math.sqrt(recall * specificity)
    # Fowlkes-Mallows Index (FMI): Geometric mean of Precision and Recall
    fmi = math.sqrt(precision * recall)
    # Cohen's Kappa
    total = TP + TN + FP + FN
    p_o = accuracy
    # p_e = Probability of chance agreement
    p_e = ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / (total**2)
    cohen_kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0

    # Define table headers
    headers = ["Metric", "Equation", "Calculation", "Value"]

    # Build table data
    data = [
        [
            "Accuracy",
            fraction_text("TP + TN", "TP + TN + FP + FN"),
            fraction_text(f"{TP} + {TN}", f"{TP} + {TN} + {FP} + {FN}"),
            f"{accuracy*100:.2f}%"
        ],
        [
            "Precision",
            fraction_text("TP", "TP + FP"),
            fraction_text(f"{TP}", f"{TP} + {FP}"),
            f"{precision*100:.2f}%"
        ],
        [
            "Recall (Sensitivity)",
            fraction_text("TP", "TP + FN"),
            fraction_text(f"{TP}", f"{TP} + {FN}"),
            f"{recall*100:.2f}%"
        ],
        [
            "Specificity",
            fraction_text("TN", "TN + FP"),
            fraction_text(f"{TN}", f"{TN} + {FP}"),
            f"{specificity*100:.2f}%"
        ],
        [
            "F1 Score",
            fraction_text("2*(Precision*Recall)", "(Precision+Recall)"),
            fraction_text(f"2*({precision:.4f}*{recall:.4f})", f"({precision:.4f}+{recall:.4f})"),
            f"{f1_score*100:.2f}%"
        ],
        [
            "False Positive Rate (FPR)",
            fraction_text("FP", "FP+TN"),
            fraction_text(f"{FP}", f"{FP}+{TN}"),
            f"{fpr*100:.2f}%"
        ],
        [
            "Matthews Corr. Coefficient (MCC)",
            fraction_text("(TP*TN - FP*FN)", "sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))"),
            fraction_text(f"({TP}*{TN} - {FP}*{FN})",
                          f"sqrt(({TP}+{FP})({TP}+{FN})({TN}+{FP})({TN}+{FN}))"),
            f"{mcc*100:.2f}%"
        ],
        [
            "Balanced Accuracy",
            fraction_text("(Recall + Specificity)", "2"),
            fraction_text(f"({recall:.4f}+{specificity:.4f})", "2"),
            f"{balanced_accuracy*100:.2f}%"
        ],
        [
            "Informedness (Youden's J)",
            r"$\text{Recall}+\text{Specificity}-1$",
            f"{recall:.4f}+{specificity:.4f}-1",
            f"{informedness*100:.2f}%"
        ],
        [
            "Negative Predictive Value (NPV)",
            fraction_text("TN", "TN + FN"),
            fraction_text(f"{TN}", f"{TN}+{FN}"),
            f"{npv*100:.2f}%"
        ],
        [
            "Markedness",
            r"$\text{Precision}+\text{NPV}-1$",
            f"{precision:.4f}+{npv:.4f}-1",
            f"{markedness*100:.2f}%"
        ],
        [
            "Prevalence",
            fraction_text("(TP + FN)", "(TP + TN + FP + FN)"),
            fraction_text(f"{TP}+{FN}", f"{TP}+{TN}+{FP}+{FN}"),
            f"{prevalence*100:.2f}%"
        ],
        [
            "Diagnostic Odds Ratio (DOR)",
            fraction_text("(TP/FN)", "(FP/TN)"),
            fraction_text(f"({TP}/{FN})", f"({FP}/{TN})"),
            f"{dor:.2f}"
        ],
        [
            "G-Mean",
            r"$\sqrt{\text{Recall}\times\text{Specificity}}$",
            rf"$\sqrt{{{recall:.4f}\times{specificity:.4f}}}$",
            f"{g_mean*100:.2f}%"
        ],
        [
            "Fowlkes–Mallows Index (FMI)",
            r"$\sqrt{\text{Precision}\times\text{Recall}}$",
            rf"$\sqrt{{{precision:.4f}\times{recall:.4f}}}$",
            f"{fmi*100:.2f}%"
        ],
        [
            "Cohen's Kappa",
            r"$\frac{p_o - p_e}{1 - p_e}$",
            rf"$\frac{{{accuracy:.4f}-{p_e:.4f}}}{{1-{p_e:.4f}}}$",
            f"{cohen_kappa*100:.2f}%"
        ]
    ]

    # Create a matplotlib figure and axis with a clean background
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('#fafafa')
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    col_widths = [0.18, 0.28, 0.32, 0.12]  # Adjust as needed
    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=col_widths
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    # Increase table size: scale horizontally by 1.25, vertically by 2.1
    table.scale(1.25, 2.1)

    # Apply custom styling
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(1.0)  # Thicker grid lines
        # Header row
        if row_idx == 0:
            cell.set_facecolor('#40466e')  # Dark background
            cell.set_text_props(color='w', weight='bold')
        else:
            # Alternate row colors for readability
            if row_idx % 2 == 0:
                cell.set_facecolor('#f1f1f2')
            else:
                cell.set_facecolor('white')
            cell.set_text_props(color='black')
            
            # Bold the Metric column
            if col_idx == 0:
                cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.savefig("metrics_table.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

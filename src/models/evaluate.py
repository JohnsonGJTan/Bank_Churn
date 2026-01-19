"""
Model Evaluation Module

This module provides functions for evaluating churn prediction models,
with a focus on metrics relevant to business value:

- Value-based metrics that account for customer balance
- Precision-recall curves for imbalanced classification
- Cross-validated confusion matrices
- Threshold analysis for optimal decision-making

Functions:
    compute_value: Calculate the value metric for a given threshold
    plot_value_curve: Plot value vs threshold curve
    plot_precision_population_curve: Plot precision and population percentage vs threshold
    confusion_matrix_df: Format confusion matrix as a DataFrame
    cv_confusion_matrix: Compute cross-validated confusion matrix with metrics
    eval_PR: Plot precision-recall curves with cross-validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.base import clone


def compute_value(X, y, y_pred_proba, threshold):
    """
    Compute the value of the predicted churners class.
    
    Value = (% Churned Balance) / (% Population)
    where % Churned Balance = (Balance of actual churners in predicted churners) / (Total balance of all actual churners)
    and % Population = (Number of predicted churners) / (Total customers)
    
    Parameters:
    X (pd.DataFrame): Feature matrix containing 'Balance' column
    y (array-like): True labels (1 = churned, 0 = not churned)
    y_pred_proba (array-like): Predicted probabilities of churning
    threshold (float): Threshold for classifying as churner
    
    Returns:
    float: Value metric
    """
    y = np.array(y)
    y_pred_proba = np.array(y_pred_proba)
    
    # Identify predicted churners
    predicted_churners = y_pred_proba >= threshold
    
    # % Population
    pct_population = predicted_churners.sum() / len(y)
    
    if pct_population == 0:
        return 0.0
    
    # Total balance of all actual churners
    total_churned_balance = X.loc[y == 1, 'Balance'].sum()
    
    if total_churned_balance == 0:
        return 0.0
    
    # Balance of actual churners in predicted churners
    actual_churners_in_predicted = (y == 1) & predicted_churners
    churned_balance_in_predicted = X.loc[actual_churners_in_predicted, 'Balance'].sum()
    
    # % Churned Balance
    pct_churned_balance = churned_balance_in_predicted / total_churned_balance
    
    # Value
    value = pct_churned_balance / pct_population
    
    return value

def plot_threshold_analysis_grid(X, y, y_pred_proba):
    """
    Create a 2x2 grid of threshold analysis plots.
    
    Layout:
        Row 1 (Cumulative): Value curve and Precision-Population curve by threshold
        Row 2 (Binned): Value and Precision-Population by probability bin
    
    Parameters:
        X (pd.DataFrame): Feature matrix containing 'Balance' column
        y (array-like): True labels (1 = churned, 0 = not churned)
        y_pred_proba (array-like): Predicted probabilities of churning
        
    Returns:
        None: Displays the 2x2 grid of plots
    """
    y = np.array(y)
    y_pred_proba = np.array(y_pred_proba)
    
    fig = plt.figure(figsize=(14, 10))
    
    # --- Row 1, Col 1: Value Curve (non-binned) ---
    ax1 = fig.add_subplot(2, 2, 1)
    thresholds = np.linspace(0.01, 0.99, 99)
    values = [compute_value(X, y, y_pred_proba, t) for t in thresholds]
    ax1.plot(thresholds, values, color='steelblue', label='Value')
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Baseline (1.0)')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Value')
    ax1.set_title('Value vs Threshold (Cumulative)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Row 1, Col 2: Precision-Population Curve (non-binned) ---
    ax2 = fig.add_subplot(2, 2, 2)
    thresholds_discrete = np.arange(0.0, 1.0, 0.1)
    pct_populations = []
    precisions = []
    for t in thresholds_discrete:
        predicted_churners = y_pred_proba >= t
        pct_pop = (predicted_churners.sum() / len(y)) * 100
        pct_populations.append(pct_pop)
        precisions.append(precision_score(y, predicted_churners.astype(int), zero_division=0))
    
    ax2.bar(thresholds_discrete, pct_populations, width=0.05, alpha=0.7, color='skyblue', label='Population %')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Population %', color='skyblue')
    ax2.tick_params(axis='y', labelcolor='skyblue')
    ax2.set_xticks(thresholds_discrete)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(thresholds_discrete, precisions, color='red', marker='o', label='Precision')
    ax2_twin.set_ylabel('Precision', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2.set_title('Precision & Population % vs Threshold (Cumulative)')
    ax2.grid(True, alpha=0.3)
    
    # --- Row 2, Col 1: Value Curve (binned) ---
    ax3 = fig.add_subplot(2, 2, 3)
    bin_edges = np.arange(0.0, 1.1, 0.1)
    bin_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]
    values_binned = []
    total_churned_balance = X.loc[y == 1, 'Balance'].sum()
    
    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i+1]
        if i == len(bin_edges) - 2:
            in_bin = (y_pred_proba >= lower) & (y_pred_proba <= upper)
        else:
            in_bin = (y_pred_proba >= lower) & (y_pred_proba < upper)
        pct_population = in_bin.sum() / len(y)
        if pct_population == 0 or total_churned_balance == 0:
            values_binned.append(0.0)
        else:
            actual_churners_in_bin = (y == 1) & in_bin
            churned_balance_in_bin = X.loc[actual_churners_in_bin, 'Balance'].sum()
            pct_churned_balance = churned_balance_in_bin / total_churned_balance
            values_binned.append(pct_churned_balance / pct_population)
    
    x_pos = np.arange(len(bin_labels))
    ax3.bar(x_pos, values_binned, color='steelblue', alpha=0.8, edgecolor='black')
    ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Baseline (1.0)')
    ax3.set_xlabel('Probability Bin')
    ax3.set_ylabel('Value')
    ax3.set_title('Value by Probability Bin')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # --- Row 2, Col 2: Precision-Population Curve (binned) ---
    ax4 = fig.add_subplot(2, 2, 4)
    pct_populations_binned = []
    precisions_binned = []
    
    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i+1]
        if i == len(bin_edges) - 2:
            in_bin = (y_pred_proba >= lower) & (y_pred_proba <= upper)
        else:
            in_bin = (y_pred_proba >= lower) & (y_pred_proba < upper)
        pct_pop = (in_bin.sum() / len(y)) * 100
        pct_populations_binned.append(pct_pop)
        if in_bin.sum() == 0:
            precisions_binned.append(0.0)
        else:
            precisions_binned.append(y[in_bin].mean())
    
    ax4.bar(x_pos, pct_populations_binned, alpha=0.7, color='skyblue', edgecolor='black', label='Population %')
    ax4.set_xlabel('Probability Bin')
    ax4.set_ylabel('Population %', color='skyblue')
    ax4.tick_params(axis='y', labelcolor='skyblue')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_pos, precisions_binned, color='red', marker='o', linewidth=2, label='Precision')
    ax4_twin.set_ylabel('Precision (Churn Rate in Bin)', color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4_twin.set_ylim(0, 1)
    ax4.set_title('Precision & Population % by Probability Bin')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def confusion_matrix_df(y_actual, y_pred):
    """
    Create a formatted confusion matrix as a DataFrame.
    
    Parameters:
        y_actual (array-like): True binary labels
        y_pred (array-like): Predicted binary labels
        
    Returns:
        pd.DataFrame: Confusion matrix with labeled rows and columns
    """
    
    cm = confusion_matrix(y_actual, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    return cm_df

def cv_confusion_matrix(pipeline, X, y, cv, threshold=0.5):
    """
    Compute cross-validated confusion matrix and classification metrics.
    
    Uses cross_val_predict to generate out-of-fold predictions, then
    calculates confusion matrix and key metrics (precision, recall, F1).
    
    Parameters:
        pipeline: Sklearn pipeline or estimator with predict_proba method
        X (array-like): Feature matrix
        y (array-like): True binary labels
        cv: Cross-validation splitter (e.g., StratifiedKFold)
        threshold (float, optional): Classification threshold. Default is 0.5.
        
    Returns:
        pd.DataFrame: Confusion matrix with labeled rows and columns
        
    Side Effects:
        Prints precision, recall, and F1 score to console
    """
    # Get cross-validated probability predictions
    y_probas_cv = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method='predict_proba'
    )[:, 1]
    
    # Convert probabilities to binary predictions using threshold
    y_pred_cv = (y_probas_cv >= threshold).astype(int)
    
    # Compute confusion matrix DataFrame
    cm_df = confusion_matrix_df(y, y_pred_cv)
    
    # Compute additional metrics
    precision = precision_score(y, y_pred_cv)
    recall = recall_score(y, y_pred_cv)
    f1 = f1_score(y, y_pred_cv)
    
    print("Cross-validated Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return cm_df

def eval_PR(X, y, pipeline, cv, name, final_pipeline=None, X_final=None, y_final=None, name_final=None):
    """
    Plot precision-recall curves for one or more models with cross-validation.
    
    Creates PR curves using cross-validated predictions for model comparison.
    Optionally adds a final model's PR curve on held-out test data.
    Also plots the prevalence baseline (proportion of positive class).
    
    Parameters:
        X (array-like): Training feature matrix
        y (array-like): Training binary labels
        pipeline: Single pipeline or list of pipelines to evaluate
        cv: Cross-validation splitter (e.g., StratifiedKFold)
        name: Single name (str) or list of names for the pipeline(s)
        final_pipeline (optional): Trained pipeline to evaluate on test set
        X_final (optional): Test feature matrix
        y_final (optional): Test binary labels
        name_final (optional): Name for the final model curve
        
    Returns:
        None: Displays the plot
        
    Note:
        For imbalanced classification, PR curves are more informative than ROC curves.
        The prevalence line shows the precision of a random classifier.
    """
    
    pipelines = pipeline if isinstance(pipeline, list) else [pipeline]
    names = name if isinstance(name, list) else [name]
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    for pipe, n in zip(pipelines, names):
        y_probas_cv = cross_val_predict(
            pipe,
            X,
            y,
            cv=cv,
            method='predict_proba'
        )[:, 1]

        display = PrecisionRecallDisplay.from_predictions(
            y,
            y_probas_cv,
            name=f"{n}",
            plot_chance_level=False,
            ax=ax
        )
    
    # Add final model PR curve if provided
    if final_pipeline is not None and X_final is not None and y_final is not None and name_final is not None:
        y_probas_final = final_pipeline.predict_proba(X_final)[:, 1]
        PrecisionRecallDisplay.from_predictions(
            y_final,
            y_probas_final,
            name=name_final,
            plot_chance_level=False,
            ax=ax
        )
    
    prevalence = sum(y) / len(y)
    ax.plot(
        [0, 1], [prevalence, prevalence], 
        linestyle='--', 
        color='gray', 
        label=f'Prevalence ({prevalence:.2f})'
    )
    
    ax.set_title("CV Precision-Recall Curve")
    ax.legend()
    plt.show()

def calculate_segment_value(df, segment_mask):
    """
    Calculate value metric for a customer segment.
    """
    total_churned_balance = df[df['Exited'] == 1]['Balance'].sum()
    segment_df = df[segment_mask]
    
    churned_balance_pct = (
        segment_df[segment_df['Exited'] == 1]['Balance'].sum() / 
        total_churned_balance
    )
    population_pct = len(segment_df) / len(df)
    
    return churned_balance_pct / population_pct if population_pct > 0 else 0
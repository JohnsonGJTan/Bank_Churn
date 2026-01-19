"""
Visualization Functions for Bank Churn EDA

This module provides plotting functions for exploratory data analysis
of the bank churn dataset, including:

- Distribution plots with statistical tests for numerical features
- Count plots with chi-square tests for categorical features  
- Churn metrics visualization for segmentation analysis

Functions:
    plot_num_features_grid: Visualize numerical features with histograms and boxplots
    plot_bin_features_grid: Visualize categorical features with count and proportion plots
    plot_churn_metrics: Visualize value and churn rate metrics for a binned feature
"""

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_num_features_grid(df, features, target):
    """
    Create a grid of distribution plots for numerical features by target class.
    
    For each feature, creates two plots:
    1. Histogram with KDE curves, colored by target class
    2. Box plot showing distributions by target class (outliers hidden)
    
    Also performs statistical tests and calculates effect sizes:
    - Independent t-test with p-value
    - Mann-Whitney U test with p-value
    - Cohen's d (effect size for t-test)
    - Mann-Whitney r (effect size for U test)
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        features (list): List of numerical feature column names
        target (str): Name of the binary target column (0/1)
        
    Returns:
        pd.DataFrame: Statistical test results for all features including
                     t-statistics, U-statistics, p-values, and effect sizes
    """
    n = len(features)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, feature in enumerate(features):
        ax_hist = axes[i, 0]
        ax_box = axes[i, 1]

        sns.histplot(data=df, x=feature, hue=target, kde=True, ax=ax_hist, legend=True, palette="tab10")
        ax_hist.set_title(f"{feature} Distribution by Churn Status")
        leg = ax_hist.get_legend()
        if leg:
            leg.set_loc('upper right')
        axes[i, 0].tick_params(axis='x', rotation=0)

        sns.boxplot(data=df, x=target, y=feature, ax=ax_box, hue=target, showfliers=False, legend=True, palette="tab10")
        ax_box.set_title(f"Box Plot of {feature} by Churn Status")
        ax_box.set_xlabel('Churn (exited)')
        ax_box.set_ylabel(feature)
        axes[i, 1].tick_params(axis='x', rotation=0)
        leg = ax_box.get_legend()
        if leg:
            leg.set_loc('upper right')

    plt.tight_layout()
    plt.show()
    
    # Statistical tests and effect sizes
    stats_list = []
    for feature in features:
        group0 = df[df[target] == 0][feature]
        group1 = df[df[target] == 1][feature]
        t_stat, t_p = stats.ttest_ind(group0, group1)
        u_stat, u_p = stats.mannwhitneyu(group0, group1, alternative='two-sided')
        
        mean0 = group0.mean()
        mean1 = group1.mean()
        sd0 = group0.std()
        sd1 = group1.std()
        cohens_d = (mean1 - mean0) / ((sd0**2 + sd1**2) / 2)**0.5
        n0 = len(group0)
        n1 = len(group1)
        r_mann = 1 - (2 * u_stat) / (n0 * n1)
        
        stats_list.append({
            'Feature': feature,
            't_stat': round(t_stat, 4),
            't_p': round(t_p, 4),
            'U_stat': round(u_stat, 4),
            'U_p': round(u_p, 4),
            "Cohen's d": round(cohens_d, 4),
            "Mann r": round(r_mann, 4)
        })
    
    return pd.DataFrame(stats_list)

def plot_bin_features_grid(df, features, target):
    """
    Create a grid of count plots for categorical/binned features by target class.
    
    For each feature, creates two plots:
    1. Count plot showing the number of samples in each category, colored by target
    2. Horizontal stacked bar plot showing churn rate proportions within each category
    
    Also performs chi-square test for independence:
    - Chi-square statistic and p-value
    - Cramér's V (effect size for chi-square test, ranges 0-1)
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        features (list): List of categorical feature column names
        target (str): Name of the binary target column (0/1)
        
    Returns:
        pd.DataFrame: Chi-square test results for all features including
                     test statistic, p-value, and Cramér's V effect size
    """
    n = len(features)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, feature in enumerate(features):
        ax0 = axes[i, 0]
        ax1 = axes[i, 1]

        sns.countplot(x=feature, hue=target, data=df, ax=ax0, palette="tab10")
        ax0.set_title(f"Churn Distribution by {feature}")
        leg = ax0.get_legend()
        if leg:
            leg.set_loc('upper right')
        axes[i, 0].tick_params(axis='x', rotation=0)
        for p in ax0.patches:
            height = p.get_height()
            if height > 0:
                ax0.annotate(int(height), (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom', fontsize=9, color='black')

        prop_df = df.groupby([feature, target], observed=True).size().unstack().fillna(0)
        prop_df = prop_df.div(prop_df.sum(axis=1), axis=0) * 100
        prop_df.plot(kind='barh', stacked=True, ax=ax1)
        ax1.set_title(f"Churn Rates by {feature}")
        ax1.set_xlabel('Proportion (%)')
        ax1.set_ylabel(feature)
        axes[i, 1].set_xlim(0, 100)
        axes[i, 1].tick_params(axis='y', rotation=0)
        leg = ax1.get_legend()
        if leg:
            leg.set_loc('upper right')

        for p in ax1.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            if width > 0:
                ax1.annotate(f'{width:.1f}%', (x + width/2, y + height/2), ha='center', va='center', fontsize=8, color='white')

    plt.tight_layout()
    plt.show()

    # Statistical tests for categorical features
    stats_list = []
    for feature in features:
        contingency_table = pd.crosstab(df[feature], df[target])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        N = contingency_table.sum().sum()
        k = contingency_table.shape[0]
        r = contingency_table.shape[1]
        cramers_v = (chi2 / (N * min(k-1, r-1)))**0.5 if N * min(k-1, r-1) > 0 else 0
        stats_list.append({
            'Feature': feature,
            'Chi2': round(chi2, 4),
            'p': round(p, 4),
            "Cramer's V": round(cramers_v, 4)
        })
    
    return pd.DataFrame(stats_list)

def plot_churn_metrics(feature, df_binned, ax_pop, ax_delta):
    """
    Plot value ratio and churn rate for a binned feature.
    
    Creates two bar plots to assess segment value:
    
    1. Value Ratio (ax_pop): (% Churned Balance / % Population)
       - Value > 2: High-value segment (red)
       - Value > 1.5: Medium-value segment (orange)  
       - Value <= 1.5: Low-value segment (blue)
       
    2. Churn Rate (ax_delta): Raw churn rate with baseline at overall churn rate
       - Above baseline: Higher churn than average (red)
       - Below baseline: Lower churn than average (cyan)
    
    Parameters:
        feature (str): Name of the binned feature column
        df_binned (pd.DataFrame): Dataframe with binned features and 'Exited' target
        ax_pop (matplotlib.axes.Axes): Axes for value ratio plot
        ax_delta (matplotlib.axes.Axes): Axes for churn rate plot
        
    Returns:
        None: Modifies the provided axes in-place
        
    Note:
        Requires 'Balance' and 'Exited' columns in df_binned
    """
    total_churned_balance = df_binned[df_binned['Exited'] == 1]['Balance'].sum()
    overall_churn_rate = df_binned['Exited'].mean()
    
    colors = sns.color_palette("tab10")
    
    # Population %
    pop_pct = (df_binned.groupby(feature, observed=True).size() / len(df_binned) * 100).sort_index()
    
    # Churned Balance %
    balance_pct = (df_binned[df_binned['Exited'] == 1].groupby(feature, observed=True)['Balance'].sum() / total_churned_balance * 100).sort_index()
    
    # Ratio: Churned Balance % / Population %
    ratio = balance_pct / pop_pct
    
    # Plot Ratio
    x = np.arange(len(ratio.index))
    colors_ratio = [colors[3] if r > 2 else (colors[1] if r > 1.5 else colors[0]) for r in ratio.values]
    ax_pop.bar(x, ratio.values, color=colors_ratio)
    ax_pop.set_xticks([])
    ax_pop.set_xticklabels([])
    ax_pop.set_title(f'{feature}')
    ax_pop.set_ylabel('Value')
    ax_pop.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Baseline (1.0)')
    ax_pop.legend()
    
    # Churn Rate (raw values with x-axis at overall churn rate)
    churn_rate = df_binned.groupby(feature, observed=True)['Exited'].mean().sort_index() * 100
    x_rate = np.arange(len(churn_rate.index))
    colors_churn = [colors[3] if cr > overall_churn_rate * 100 else colors[2] for cr in churn_rate.values]
    ax_delta.bar(x_rate, churn_rate.values - overall_churn_rate * 100, bottom=overall_churn_rate * 100, color=colors_churn)
    ax_delta.axhline(y=overall_churn_rate * 100, color='black', linestyle='-', linewidth=1, label=f'Overall ({overall_churn_rate*100:.1f}%)')
    ax_delta.set_xticks(x_rate)
    ax_delta.set_xticklabels(churn_rate.index, rotation=0)
    ax_delta.set_ylabel('Churn Rate (%)')
    
    # Set y-axis limits to tightly fit the data with small padding
    min_churn = min(churn_rate.values)
    max_churn = max(churn_rate.values)
    padding = (max_churn - min_churn) * 0.15
    ax_delta.set_ylim(min_churn - padding, max_churn + padding)
    ax_delta.legend()
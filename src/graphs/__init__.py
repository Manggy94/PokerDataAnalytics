import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chisquare

def comparison_bar_plot(s1, s2, x_title, y_title, title):
    s1_norm = s1 / s1.sum()
    s2_norm = s2 / s2.sum()
    merged_df = pd.merge(s1_norm, s2_norm, left_index=True, right_index=True).astype(float)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=merged_df.index, y=merged_df[s1.name], name="Empirical Values"), )
    fig.add_trace(go.Bar(x=merged_df.index, y=merged_df[s2.name], name="Theoretical Values"))
    fig.update_layout(barmode='group', xaxis_title=x_title, yaxis_title=y_title, title=title)
    fig.show()
    print("Chi-square test:")
    print("H0: The observed frequency distribution is equal to the expected frequency")
    print("H1: The observed frequency distribution is not equal to the expected frequency")
    chi2_stat, p_val = chisquare(f_obs=merged_df[s1.name], f_exp=merged_df[s2.name])
    print(f"Chi2 Stat: {chi2_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Reject H0: The observed frequency is not equal to the expected frequency")
    else:
        print("Fail to reject H0: The observed frequency is equal to the expected frequency")

def freq_comparison_bar_plot(s1, s2, x_title, y_title, title):
    s1_freq = s1.value_counts().sort_index()
    s2_freq = s2.value_counts().sort_index()
    comparison_bar_plot(s1=s1_freq, s2=s2_freq, x_title=x_title, y_title=y_title, title=title)

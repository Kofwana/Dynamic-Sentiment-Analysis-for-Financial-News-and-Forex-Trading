import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
from helper_functions import sentiment_mae

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('sentiment_predictions_single_article.csv', parse_dates=['published_at'])
df2 = pd.read_csv('sentiment_predictions_allday_articles.csv', parse_dates=['published_at'])

# Define sentiment columns
cols_sent = ['true_sentiment', 'finbert_sentiment', 'finbert_sentiment_a', 'gpt_sentiment_p1',
             'gpt_sentiment_p2', 'gpt_sentiment_p3', 'gpt_sentiment_p4', 'gpt_sentiment_p7']

# Classification Metrics
classification_metrics = {
    'Model': ['FinBERT', 'FinBERT-A', 'GPT-P1', 'GPT-P2', 'GPT-P3', 'GPT-P4', 'GPT-P4A'],
    'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'S-MAE']
}

# Calculate and print classification report for each model
for model in cols_sent[1:]:
    print(f"Classification report for {model.upper()}:")
    report = classification_report(df['true_sentiment'], df[model], output_dict=True)
    
    for key, value in report.items():
        if isinstance(value, dict):
            # If the value is a dictionary, print its items directly
            print(f"{key}:")
            for k, v in value.items():
                if isinstance(v, (int, float)):  # Check if the value is numerical
                    print(f"  {k}: {round(v, 3)}")  # Round the numerical value
                else:
                    print(f"  {k}: {v}")  # Print as it is if not numerical
        elif isinstance(value, (int, float)):  # Check if the value is numerical
            print(f"{key}: {round(value, 3)}")  # Round the numerical value
        else:
            print(f"{key}: {value}")  # Print as it is if not numerical
    
    print("-" * 50)

"""
# Calculate metrics for each model
model_metrics = []
for model in cols_sent[1:]:
    model_name = model.replace('_sentiment', '').upper()
    accuracy = accuracy_score(df['true_sentiment'], df[model])
    precision = precision_score(df['true_sentiment'], df[model], average='weighted')
    recall = recall_score(df['true_sentiment'], df[model], average='weighted')
    f1 = f1_score(df['true_sentiment'], df[model], average='weighted')
    mae = sentiment_mae(df['true_sentiment'], df[model])
    model_metrics.append([model_name, accuracy, precision, recall, f1, mae])

# Create DataFrame for classification metrics
metrics_df = pd.DataFrame(model_metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'S-MAE'])

# Print classification results
print('Performance Results in Sentiment Classification')
print("-" * 50)
print(metrics_df.round(3))
print("-" * 50)
"""
# Plot Correlation Matrix
corr_matrix = df[cols_sent].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.xticks(rotation=45)
plt.title("Correlation Matrix: Sentiment vs. Daily Returns")
plt.savefig('fig_res_corr.png', dpi=500)

# Calculate and print classification report for each model
for model in cols_sent[1:]:
    print(f"Classification report for {model.upper()}:")
    report = classification_report(df['true_sentiment'], df[model], output_dict=True)
    print({key: round(value, 3) for key, value in report.items()})
    print("-" * 50)

# Daily Sentiment Analysis
print('Daily Sentiment Analysis')
print("#" * 50)

# Merge sentiment dataframes
merged_df = pd.merge(df, df2, on=['published_at', 'ticker'], how='outer', suffixes=('', '_day'))

# Group by date and ticker
grouped_df = merged_df.groupby(['published_at', 'ticker']).agg({
    'true_sentiment': 'sum',
    'finbert_sentiment': 'sum',
    'finbert_sentiment_a': 'sum',
    'gpt_sentiment_p1': 'sum',
    'gpt_sentiment_p2': 'sum',
    'gpt_sentiment_p3': 'sum',
    'gpt_sentiment_p4': 'sum',
    'gpt_sentiment_p5': 'sum',
    'gpt_sentiment_p6': 'sum',
    'gpt_sentiment_p7': 'sum',
    'finbert_sentiment_n': 'sum',
    'finbert_sentiment_a_n': 'sum',
    'gpt_sentiment_p1n': 'sum',
    'gpt_sentiment_p2n': 'sum',
    'gpt_sentiment_p3n': 'sum',
    'gpt_sentiment_p4n': 'sum',
    'gpt_sentiment_p5n': 'sum',
    'gpt_sentiment_p6n': 'sum',
    'gpt_sentiment_p7n': 'sum',
    'daily_returns': 'mean'
}).reset_index()

# Calculate Naive Sentiment
grouped_df['naive_sentiment'] = grouped_df.groupby('ticker')['true_sentiment'].transform(lambda x: x.mode().iloc[0])

# Calculate correlation between sentiment and returns
correlation_df = grouped_df[cols_sent + ['daily_returns']].corr()['daily_returns']

# Print correlation results
print('Correlation of Predicted Sentiment and Forex pair returns')
print("-" * 50)
print(correlation_df)
print("-" * 50)


# Plot Correlation Matrix
corr_matrix = df[cols_sent].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix: Sentiment vs. Daily Returns")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# Plotting Additional Correlation Plots
plt.figure(figsize=(12, 8))

for i, col in enumerate(cols_sent[1:], 1):
    plt.subplot(2, 3, i)
    plt.scatter(df[col], df['true_sentiment'], alpha=0.5)
    plt.xlabel(col.replace('_sentiment', '').upper())
    plt.ylabel('True Sentiment')
    plt.title(f'Correlation: {col.replace("_sentiment", "").upper()} vs. True Sentiment')

plt.tight_layout()
plt.savefig('additional_correlation_plots.png', dpi=300)
plt.show()
"""Optimize Imports:
Import only necessary functions from libraries to reduce memory usage and improve code readability.
Use Groupby and Aggregation:
Utilize groupby and aggregation functions like agg to compute metrics more efficiently instead of looping through each ticker.
Handle NaN Values:
Handle any NaN values that may arise during computations to prevent errors and ensure accurate results.
Optimize DataFrame Operations:
Optimize DataFrame operations to reduce redundancy and improve performance.
Plotting Improvements:
Adjust the plot settings for better visualization, including rotating x-axis labels in the correlation heatmap.
Refactoring and Code Organization:
Refactor the code for better readability and organization, including comments for clarity."""
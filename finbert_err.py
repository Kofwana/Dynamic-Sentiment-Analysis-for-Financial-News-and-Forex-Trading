import pandas as pd
from sklearn.metrics import classification_report

# Load your data and calculate classification report
df = pd.read_csv('sentiment_predictions_single_article.csv', parse_dates=True)
s = df[['true_sentiment', 'finbert_sentiment', 'finbert_sentiment_a', 'gpt_sentiment_p1', 'gpt_sentiment_p2',
        'gpt_sentiment_p3', 'gpt_sentiment_p4', 'gpt_sentiment_p7']]

# Loop through each model and calculate classification report
for model in s.columns[1:]:
    print(f"Classification report for {model}:")
    report = classification_report(s['true_sentiment'], s[model], output_dict=True)
    print(report.items())  # Print the contents of report.items()

    for key, value in report.items():
        if isinstance(value, dict):
            # If the value is a dictionary, print each key-value pair separately
            print(key, ":")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            # Round numeric values only
            print(key, ":", round(value, 3))

    print("-" * 50)

import pandas as pd
from results_analysis import calculate_metrics, print_classification_report, calculate_daily_sentiment_analysis, \
    plot_correlation_heatmap, calculate_correlation
from helper_functions_modular import cols_sent

if __name__ == "__main__":
    # Load data and define other necessary variables
    df = pd.read_csv('sentiment_predictions_single_article.csv', parse_dates=True)
    df2 = pd.read_csv('sentiment_predictions_allday_articles.csv', parse_dates=True)

    s = df[cols_sent]

    # Calculate metrics and print classification report
    metrics_df = calculate_metrics(s)
    print("Performance Results in Sentiment Classification")
    print("-" * 50)
    print(metrics_df)
    print("-" * 50)

    print("Classification report per model:")
    print_classification_report(s)

    # Daily sentiment analysis and correlation calculations
    daily_sentiment_and_returns = calculate_daily_sentiment_analysis(df, df2)
    plot_correlation_heatmap(daily_sentiment_and_returns)
    calculate_correlation(daily_sentiment_and_returns)


import matplotlib.pyplot as plt
import seaborn as sns
from helper_functions import compact_correlation_df

# plotting the correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(compact_correlation_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

plt.title("Correlation Matrix: Predicted Sentiment vs. Forex Pair Returns")
plt.xlabel("Forex Pair Tickers")
plt.ylabel("Predicted Sentiment Models")

plt.savefig('fig_sentiment_forex_correlation.png', dpi=300)
plt.show()

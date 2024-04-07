
---

# Dynamic Sentiment Analysis for Financial News and Forex Trading

## Project Overview

This project aims to enhance the existing codebase for dynamic sentiment analysis of financial news articles and headlines, specifically targeting the forex trading domain. The project leverages machine learning models, including FinBERT and GPT (Generative Pre-trained Transformer) models, to analyze the sentiment of news articles and provide insights that can potentially inform forex trading decisions.

## Improvements in Code

### 1. Modularity and Readability
- The codebase has been modularized into separate files for helper functions, prompts, and result analysis to improve readability and maintainability.
- Helper functions such as `get_sentiment`, `extract_sentiment`, and `sentiment_to_numeric` are organized into a separate file for better code organization.

### 2. Enhanced Data Handling
- Improved data handling and processing techniques, such as loading prompts from JSON files, handling exceptions, and integrating pandas for data manipulation, have been implemented.
- The codebase now includes functions to calculate sentiment Mean Absolute Error (MAE), highlight maximum and minimum values in data, and generate performance metrics for sentiment analysis models.

### 3. Result Visualization
- Added functionality to visualize and analyze the results using matplotlib and seaborn libraries.
- Included heatmap visualization for correlation analysis between sentiment predictions and daily forex returns.
- Generated additional correlation plots to explore relationships between sentiment scores and true sentiment labels.

## Observed Enhancements

The enhancements made to the codebase resulted in significant improvements in sentiment analysis accuracy and performance evaluation. Notable changes include:
- Improved precision, recall, and F1-scores across FinBERT and GPT sentiment analysis models.
- Reduced sentiment Mean Absolute Error (MAE) values, indicating better alignment between predicted and true sentiment labels.

## Additional Plots and Insights

### Heatmap: Correlation Matrix
The heatmap visualization of the correlation matrix provides insights into the relationships between sentiment predictions from different models and their impact on daily forex returns. Strong correlations can indicate potential trading signals or sentiment-driven market movements.

### Scatter Plots: Correlation Analysis
Additional scatter plots illustrate the correlation between individual sentiment scores (e.g., FinBERT, GPT-P1, GPT-P2) and true sentiment labels. These plots help identify patterns and trends in sentiment analysis accuracy across different models and sentiment categories.

### Cumulative Distribution Function (CDF) Plot
The CDF plot showcases the distribution of predicted sentiment scores, offering a comprehensive view of sentiment variability and confidence levels across different sentiment analysis models.

## Goal of the Project

The primary goal of this project is to build a robust sentiment analysis system tailored for financial news in the forex trading domain. Key objectives include:
1. **Sentiment Analysis:** Develop and train machine learning models (FinBERT, GPT models) for accurate sentiment analysis of financial news articles.
2. **Dynamic Prompts:** Generate dynamic prompts based on the type of news content (headline or article) to extract sentiment using appropriate models.
3. **Performance Evaluation:** Evaluate the performance of sentiment analysis models using metrics such as accuracy, precision, recall, F1-score, and sentiment MAE.
4. **Correlation Analysis:** Explore the correlation between predicted sentiment scores and daily forex pair returns to identify potential trading signals.

## Project Structure

The project structure includes the following files and directories:
- `helper_functions.py`: Contains helper functions for sentiment analysis, data handling, and result extraction.
- `prompts.json`: JSON file with prompts for different types of sentiment analysis tasks.
- `results_analysis.py`: Main script for sentiment analysis using machine learning models, result visualization, and performance metrics calculation.
- `README.md`: Project documentation file explaining improvements, goals, and usage instructions.

## Getting Started

1. **Environment Setup:**
   - Ensure you have Python installed on your system along with the required libraries specified in `requirements.txt`.
   - Create a virtual environment and activate it:
     ```bash
     python -m venv venv
     source venv/bin/activate  # For Unix/Linux
     # OR
     .\venv\Scripts\activate  # For Windows
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Running the Code:**
   - Modify the code as needed, such as adding your OpenAI API key in the appropriate file.
   - Execute the main script for sentiment analysis:
     ```bash
     python results.py
     ```

3. **Result Interpretation:**
   - Explore the generated results, performance metrics, and visualizations to gain insights into sentiment analysis and correlation with forex returns.

## Contributors

- Kofwana Lawson
- Vydeepthi Dhulipalla

---

Citations: G. Fatouros, J. Soldatos, K. Kouroumali et al., Transforming sentiment analysis in the financial domain with ChatGPT. Machine Learning with Applications (2023), doi: https://doi.org/10.1016/j.mlwa.2023.100508.
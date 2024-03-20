import openai
import time
import numpy as np
import logging
import json
from prompt_generator import get_dynamic_prompt
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sentiment(ticker, content, prompt, max_retries=5):
    """
    Get sentiment using a specified prompt.

    Parameters:
    - ticker: The forex ticker.
    - content: The article or headline.
    - prompt: The dynamically generated prompt.
    - max_retries: Maximum number of retries in case of an error.

    Returns:
    - Sentiment, completion tokens, prompt tokens, response time.
    """
    for retry in range(max_retries):
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{'role': 'system', 'content': prompt}],
                temperature=0,
                max_tokens=16,
                top_p=1,
            )
            end_time = time.time()
            response_time = end_time - start_time

            return response.choices[0].message.content, response.usage['completion_tokens'], response.usage['prompt_tokens'], response_time
        except Exception as e:
            logger.error(f"Error while processing '{content}' (Retry {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                # Sleep for a while before retrying
                time.sleep(0.1)
            else:
                # If all retries fail, return NaN values
                return np.NaN, np.NaN, np.NaN, np.NaN

def extract_sentiment(sentiment: str):
    sentiment = sentiment.lower()
    if "positive" in sentiment:
        return "Positive"
    elif "neutral" in sentiment:
        return "Neutral"
    elif "negative" in sentiment:
        return "Negative"
    elif "sell" in sentiment:
        return "Negative"
    elif "buy" in sentiment:
        return "Positive"
    else:
        return sentiment

def sentiment_to_numeric(sentiment: str):
    sentiment = sentiment.lower()
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return -1
    elif sentiment == 'neutral':
        return 0
    else:
        return np.nan

cols = ['published_at', 'ticker', 'title', 'text', 'true_sentiment', 'finbert_sentiment', 'finbert_sentiment_a',
        'gpt_sentiment_p1', 'gpt_completion_tokens_p1', 'gpt_prompt_tokens_p1', 'gpt_time_p1',
        'gpt_sentiment_p2', 'gpt_completion_tokens_p2', 'gpt_prompt_tokens_p2', 'gpt_time_p2',
        'gpt_sentiment_p3', 'gpt_completion_tokens_p3', 'gpt_prompt_tokens_p3', 'gpt_time_p3',
        'gpt_sentiment_p4', 'gpt_completion_tokens_p4', 'gpt_prompt_tokens_p4', 'gpt_time_p4',
        'gpt_sentiment_p7', 'gpt_completion_tokens_p7', 'gpt_prompt_tokens_p7', 'gpt_time_p7',
        'finbert_sentiment_n', 'finbert_sentiment_a_n',
        'gpt_sentiment_p1n', 'gpt_completion_tokens_p1n', 'gpt_prompt_tokens_p1n', 'gpt_time_p1n',
        'gpt_sentiment_p2n', 'gpt_completion_tokens_p2n', 'gpt_prompt_tokens_p2n', 'gpt_time_p2n',
        'gpt_sentiment_p3n', 'gpt_completion_tokens_p3n', 'gpt_prompt_tokens_p3n', 'gpt_time_p3n',
        'gpt_sentiment_p4n', 'gpt_completion_tokens_p4n', 'gpt_prompt_tokens_p4n', 'gpt_time_p4n',
        'gpt_sentiment_p7n', 'gpt_completion_tokens_p7n', 'gpt_prompt_tokens_p7n', 'gpt_time_p7n']

cols_sent = ['published_at', 'ticker', 'true_sentiment', 'finbert_sentiment', 'finbert_sentiment_a',
             'gpt_sentiment_p1', 'gpt_sentiment_p2', 'gpt_sentiment_p3', 'gpt_sentiment_p4', 'gpt_sentiment_p7',
             'finbert_sentiment_n', 'finbert_sentiment_a_n', 'gpt_sentiment_p1n', 'gpt_sentiment_p2n',
             'gpt_sentiment_p3n', 'gpt_sentiment_p4n', 'gpt_sentiment_p7n']

cols_tokens = ['published_at', 'ticker', 'gpt_time_p1', 'gpt_time_p2', 'gpt_time_p3', 'gpt_time_p4', 'gpt_time_p7',
               'gpt_time_p1n', 'gpt_time_pn2', 'gpt_time_p3n', 'gpt_time_p4n', 'gpt_time_p7n',
               'gpt_prompt_tokens_p1', 'gpt_prompt_tokens_p2', 'gpt_prompt_tokens_p3', 'gpt_prompt_tokens_p4',
               'gpt_prompt_tokens_p7',
               'gpt_prompt_tokens_p1n', 'gpt_prompt_tokens_p2n', 'gpt_prompt_tokens_p3n', 'gpt_prompt_tokens_p4n',
               'gpt_prompt_tokens_p7n',
               'gpt_completion_tokens_p1', 'gpt_completion_tokens_p2', 'gpt_completion_tokens_p3',
               'gpt_completion_tokens_p4', 'gpt_completion_tokens_p7',
               'gpt_completion_tokens_p1n', 'gpt_completion_tokens_p2n', 'gpt_completion_tokens_p3n',
               'gpt_completion_tokens_p4n', 'gpt_completion_tokens_p7n']

cols_sent_day = ['published_at', 'ticker', 'gpt_sentiment_p5', 'gpt_sentiment_p5n', 'gpt_sentiment_p6',
                 'gpt_sentiment_p6n']
cols_tokens_day = ['published_at', 'ticker',
                   'gpt_completion_tokens_p5', 'gpt_prompt_tokens_p5',
                   'gpt_completion_tokens_p5n', 'gpt_prompt_tokens_p5n',
                   'gpt_completion_tokens_p6', 'gpt_prompt_tokens_p6',
                   'gpt_completion_tokens_p6n', 'gpt_prompt_tokens_p6n',
                   'gpt_time_p5', 'gpt_time_p5n', 'gpt_time_p6', 'gpt_time_p6n']

def sentiment_mae(y_true, y_pred):
    mae = np.mean(np.abs(np.array(y_pred) - np.array(y_true)))
    return mae

def highlight_max(s):
    """Highlight the maximum in a Series."""
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

if __name__ == "__main__":
    # Load data and define other necessary variables

    for prompt_type in prompts.keys():
        if 'A' not in prompt_type:
            content = df.title[row]
        else:
            content = df.text[row]
        
        dynamic_prompt = get_dynamic_prompt(df.ticker[row], content, prompt_type)
        result = get_sentiment(df.ticker[row], content, dynamic_prompt)
        print(f'{prompt_type}: {result}')
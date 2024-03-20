def get_dynamic_prompt(ticker, content, prompt_type):
    if prompt_type in ['GPT-P4A', 'GPT-P4AN']:
        prompt = f"Predict sentiment for ticker {ticker} based on the following headline: '{content}'"
    elif prompt_type in ["GPT-P3", "GPT-P3N"]:
        prompt = f"Analyze sentiment for the headline: '{content}'"
    else:
        prompt = f"Predict sentiment for ticker {ticker} based on the following article: '{content}'"

    return prompt

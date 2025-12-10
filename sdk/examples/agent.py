from evalyn_sdk import eval


@eval
def classify_sentiment(user_input: str) -> str:
    """
    Tiny example agent for demos. Replace this with your real LLM call.
    """
    text = user_input.lower()
    if "bad" in text or "terrible" in text:
        return "negative"
    if "good" in text or "great" in text:
        return "positive"
    return "neutral"


if __name__ == "__main__":
    print(classify_sentiment("this is great"))

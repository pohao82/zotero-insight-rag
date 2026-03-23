# A simple way to clean the query for the vector engine
def distill_query(query):
    # Remove common question starters
    stop_phrases = ["what is", "tell me about", "show me", "can you find"]
    clean_query = query.lower()
    for phrase in stop_phrases:
        clean_query = clean_query.replace(phrase, "")
    return clean_query.strip()

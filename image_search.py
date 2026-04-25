import requests

def get_topic_image(topic: str) -> str | None:
    try:
        # Search Wikipedia for the topic
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "format": "json",
            "titles": topic,
            "prop": "pageimages",
            "pithumbsize": 500
        }
        response = requests.get(search_url, params=search_params, timeout=5)
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            thumbnail = page.get("thumbnail", {})
            if thumbnail:
                return thumbnail.get("source")
        return None
    except Exception:
        return None


def extract_topic(answer: str, question: str) -> str:
    # Use first few words of question as topic
    words = question.strip().split()
    topic = " ".join(words[:4])
    return topic
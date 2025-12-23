import requests
from bs4 import BeautifulSoup

def scrape_article(url: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("h1")
    paragraphs = soup.find_all("p")

    text = " ".join(p.get_text(strip=True) for p in paragraphs)

    return {
        "url": url,
        "title": title_tag.get_text(strip=True) if title_tag else "",
        "text": text
    }

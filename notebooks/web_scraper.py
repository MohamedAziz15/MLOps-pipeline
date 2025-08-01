import requests
from bs4 import BeautifulSoup
import os

def scrape_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join(p.get_text() for p in paragraphs)

def save_scraped_text(url, out_path):
    text = scrape_text_from_url(url)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    urls = [
        "https://afdc.energy.gov/fuels/electricity_locations.html",
        "https://en.wikipedia.org/wiki/Charging_station"
    ]
    for url in urls:
        fname = url.split("/")[-1].split(".")[0] + ".txt"
        save_scraped_text(url, f"./data/processed/{fname}")










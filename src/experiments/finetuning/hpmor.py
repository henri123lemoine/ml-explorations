# This experiment is about finetuning a Qwen model on The Book of HPMOR Fanfics: https://www.lesswrong.com/posts/uWM5auewdKjdJJG9t/the-book-of-hpmor-fanfics

import logging
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

from src.utils.cache import cache

logger = logging.getLogger(__name__)

HPMOR_EXP_PATH = Path(__file__).resolve().parent
DATA_PATH = HPMOR_EXP_PATH / "data"
PLOTS_PATH = DATA_PATH / "plots"
CHAPTERS_PATH = DATA_PATH / "chapters"
LOG_PATH = DATA_PATH / "logs"

PLOTS_PATH.mkdir(parents=True, exist_ok=True)
CHAPTERS_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH.mkdir(parents=True, exist_ok=True)

HPMOR_BASE = "https://hpmor.com"
CHAPTERS_ROUTE = "/chapter/"
CHAPTER_COUNT = 122
# Full route e.g.: https://hpmor.com/chapter/1

MAX_RETRIES = 3


def extract_chapter(number: int) -> dict:
    url = f"{HPMOR_BASE}{CHAPTERS_ROUTE}{number}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "text/html",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    title_div = soup.find("div", id="chapter-title")
    title = title_div.contents[0].replace("\n", " ").strip()

    content_div = soup.find("div", id="storycontent")
    content = md(str(content_div))

    return {"title": title, "content": content}


@cache.disk_cache(key_prefix="hpmor_chapter", serializer="json")
def fetch_chapter(number: int) -> dict:
    return extract_chapter(number)


def get_chapter(number: int) -> dict:
    for attempt in range(MAX_RETRIES):
        time.sleep(1)
        try:
            return fetch_chapter(number)
        except Exception as e:
            logger.error(f"Error extracting chapter {number} (attempt {attempt + 1}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)  # Exponential backoff
    raise Exception(f"Failed to extract chapter {number} after {MAX_RETRIES} attempts")


def extract_hpmor() -> str:
    chapters = {}

    with tqdm(total=CHAPTER_COUNT, desc="Extracting chapters") as pbar:
        for i in range(1, CHAPTER_COUNT + 1):
            try:
                chapters[i] = get_chapter(i)
                pbar.update(1)
            except Exception as e:
                logger.error(f"Failed to get chapter {i}: {str(e)}")

    if len(chapters) != CHAPTER_COUNT:
        logger.warning(f"Only {len(chapters)}/{CHAPTER_COUNT} chapters were successfully extracted")

    full_text = "\n\n".join(
        f"# {chapters.get(i, {'title': f'Chapter {i}', 'content': '[Content missing]'})['title']}\n\n{chapters.get(i, {'content': ''})['content']}"
        for i in range(1, CHAPTER_COUNT + 1)
    )

    output_file = DATA_PATH / "hpmor.md"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info(f"Successfully saved HPMOR text to {output_file}")
    except IOError as e:
        logger.error(f"Error saving HPMOR text to file: {str(e)}")

    return full_text


if __name__ == "__main__":
    extract_hpmor()

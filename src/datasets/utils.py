import logging
import re
import time
from pathlib import Path
from typing import Callable

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.utils.cache import cache

logger = logging.getLogger(__name__)


class DatasetExtractor:
    def __init__(
        self,
        base_url: str,
        route: str,
        total_items: int,
        output_dir: str | Path,
        cache_key_prefix: str,
        dataset_name: str = "",
        dataset_author: str = "",
        max_retries: int = 3,
        retry_delay: int = 1,
        request_delay: float = 1.0,
    ):
        self.base_url = base_url
        self.route = route
        self.total_items = total_items
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_key_prefix = cache_key_prefix
        self.dataset_name = dataset_name
        self.dataset_author = dataset_author
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self.last_request_time = 0

    def extract_item(self, item_number: int) -> dict[str, str]:
        """
        Extract a single item from the dataset.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement extract_item method")

    def fetch_item(self, item_number: int) -> dict[str, str]:
        """Fetch and cache a single item"""
        logger.debug(f"Cache miss for item {item_number}. Extracting data.")
        self._delay_if_needed()
        return self.extract_item(item_number)

    def _delay_if_needed(self):
        """Delay the request if necessary to respect the request_delay"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            time.sleep(self.request_delay - time_since_last_request)
        self.last_request_time = time.time()

    def get_item(self, item_number: int) -> dict[str, str]:
        """Get a single item with retries"""
        for attempt in range(self.max_retries):
            try:
                return self.fetch_item(item_number)
            except Exception as e:
                logger.error(
                    f"Error extracting item {item_number} (attempt {attempt + 1}): {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff
        raise Exception(f"Failed to extract item {item_number} after {self.max_retries} attempts")

    def extract_dataset(self) -> str:
        """Extract the entire dataset"""
        items = {}

        with tqdm(total=self.total_items, desc="Extracting items") as pbar:
            for i in range(1, self.total_items + 1):
                try:
                    items[i] = self.get_item(i)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to get item {i}: {str(e)}")

        if len(items) != self.total_items:
            logger.warning(
                f"Only {len(items)}/{self.total_items} items were successfully extracted"
            )

        return self.format_dataset(items)

    def format_dataset(self, items: dict[int, dict[str, str]]) -> str:
        """
        Format the extracted dataset.
        This method can be overridden by subclasses if needed.
        """
        header = f"# {self.dataset_name}\n\n"
        if self.dataset_author:
            header += f"by {self.dataset_author}\n\n"

        content = "\n\n".join(
            f"## {items.get(i, {'title': f'Chapter {i}', 'content': '[Content missing]'})['title']}\n\n{items.get(i, {'content': ''})['content']}"
            for i in range(1, self.total_items + 1)
        )

        return header + content

    def save_dataset(self, content: str, filename: str) -> None:
        """Save the dataset to a file"""
        output_file = self.output_dir / filename
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Successfully saved dataset to {output_file}")
        except IOError as e:
            logger.error(f"Error saving dataset to file: {str(e)}")


class WebPageExtractor(DatasetExtractor):
    def __init__(
        self,
        base_url: str,
        route: str,
        total_items: int,
        output_dir: str | Path,
        cache_key_prefix: str,
        title_selector: str,
        content_selector: str,
        content_processor: Callable[[str], str] | None = None,
        **kwargs,
    ):
        super().__init__(base_url, route, total_items, output_dir, cache_key_prefix, **kwargs)
        self.title_selector = title_selector
        self.content_selector = content_selector
        self.content_processor = content_processor or (lambda x: x)

    @cache.disk_cache(serializer="json")
    def fetch_item(self, item_number: int) -> dict[str, str]:
        logger.debug(f"Cache miss for item {item_number}. Extracting data.")
        self._delay_if_needed()
        return self.extract_item(item_number)

    def extract_item(self, item_number: int) -> dict[str, str]:
        url = f"{self.base_url}{self.route}{item_number}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "text/html",
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        title_element = soup.select_one(self.title_selector)
        content_element = soup.select_one(self.content_selector)

        if not title_element or not content_element:
            raise ValueError(f"Failed to extract title or content from {url}")

        title = title_element.get_text(strip=True)
        title = re.sub(r"\s+", " ", title) + "\n"

        content = str(content_element)
        content = self.content_processor(content)

        return {"title": title, "content": content}

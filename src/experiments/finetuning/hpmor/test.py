import logging
import re
import time
from pathlib import Path

from markdownify import markdownify

from src.datasets.utils import DatasetExtractor, WebPageExtractor
from src.settings import DATASETS_PATH
from src.utils.cache import cache

HPMOR_EXP_PATH = Path(__file__).resolve().parent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MockExtractor(DatasetExtractor):
    def __init__(self, total_items: int, output_dir: str | Path, cache_key_prefix: str, **kwargs):
        super().__init__("", "", total_items, output_dir, cache_key_prefix, **kwargs)

    def extract_item(self, item_number: int) -> dict[str, str]:
        logger.debug(f"Actually extracting mock item {item_number}")
        time.sleep(self.request_delay)  # Simulate network delay
        return {
            "title": f"Mock Title {item_number}",
            "content": f"This is mock content for item {item_number}.",
        }

    @cache.disk_cache(key_prefix="hpmor_chapter_mock", serializer="json")
    def fetch_item(self, item_number: int) -> dict[str, str]:
        logger.debug(f"Fetching item {item_number}")
        return self.extract_item(item_number)


def hpmor_processor(content: str) -> str:
    content = markdownify(content, heading_style="ATX")
    content = re.sub(r"\n{3,}", "\n\n", content)  # Remove consecutive blank lines
    content = re.sub(
        r"^ +", "", content, flags=re.MULTILINE
    )  # Remove spaces at the beginning of lines
    content = re.sub(r"(\n#+.+)\n+", r"\1\n\n", content)  # Ensure proper spacing around headers
    content = re.sub(r"&[a-zA-Z]+;", "", content)  # Remove any remaining HTML entities
    return content.strip()


def extract_hpmor():
    # extractor = MockExtractor(
    #     total_items=10,
    #     output_dir=DATASETS_PATH,
    #     cache_key_prefix="hpmor_chapter_mock",
    #     request_delay=0.1,
    # )

    extractor = WebPageExtractor(
        base_url="https://hpmor.com",
        route="/chapter/",
        total_items=5,
        output_dir=DATASETS_PATH,
        cache_key_prefix="hpmor_chapter",
        title_selector="#chapter-title",
        content_selector="#storycontent",
        content_processor=hpmor_processor,
    )

    content = extractor.extract_dataset()  # should be slow the first time, then fast

    extractor.save_dataset(content, "hpmor.md")
    return content


if __name__ == "__main__":
    extract_hpmor()

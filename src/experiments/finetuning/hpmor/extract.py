import logging
import re
from pathlib import Path

from markdownify import markdownify

from src.datasets.base import WebPageExtractor
from src.settings import DATASETS_PATH

HPMOR_EXP_PATH = Path(__file__).resolve().parent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    extractor = WebPageExtractor(
        base_url="https://hpmor.com",
        route="/chapter/",
        total_items=122,
        output_dir=DATASETS_PATH,
        cache_key_prefix="hpmor_chapter",
        title_selector="#chapter-title",
        content_selector="#storycontent",
        content_processor=hpmor_processor,
        dataset_name="Harry Potter and the Methods of Rationality",
        dataset_author="Eliezer Yudkowsky",
    )

    content = extractor.extract_dataset()  # should be slow the first time, then fast

    extractor.save_dataset(content, "hpmor.md")
    return content


if __name__ == "__main__":
    extract_hpmor()

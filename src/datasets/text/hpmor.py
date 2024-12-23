import logging
import re
from pathlib import Path

import torch
from markdownify import markdownify

from ..extractors.web import WebPageExtractor
from .base import TextDataset

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


class HPMORDataset(TextDataset):
    def __init__(
        self,
        cache_dir: Path,
        split: str = "train",
        tokenizer=None,
    ):
        super().__init__(cache_dir, split, tokenizer)

    def _load_data(self):
        cache_path = self._get_cache_path()
        if cache_path.exists():
            return torch.load(cache_path)

        extractor = WebPageExtractor(
            base_url="https://hpmor.com",
            route="/chapter/",
            total_items=122,
            output_dir=self.cache_dir,
            cache_key_prefix="hpmor_chapter",
            title_selector="#chapter-title",
            content_selector="#storycontent",
            content_processor=hpmor_processor,
            dataset_name="Harry Potter and the Methods of Rationality",
            dataset_author="Eliezer Yudkowsky",
        )

        content = extractor.extract_dataset()
        torch.save(content, cache_path)
        return content


if __name__ == "__main__":
    from src.settings import DATASETS_PATH

    dataset = HPMORDataset(DATASETS_PATH / "hpmor")
    print(len(dataset))

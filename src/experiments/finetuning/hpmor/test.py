from pathlib import Path

from markdownify import markdownify

from src.datasets.utils import WebPageExtractor

HPMOR_EXP_PATH = Path(__file__).resolve().parent
DATA_PATH = HPMOR_EXP_PATH / "data"


def extract_hpmor():
    extractor = WebPageExtractor(
        base_url="https://hpmor.com",
        route="/chapter/",
        total_items=122,
        output_dir=DATA_PATH,
        cache_key_prefix="hpmor_chapter",
        title_selector="#chapter-title",
        content_selector="#storycontent",
        content_processor=markdownify,
    )

    content = extractor.extract_dataset()
    extractor.save_dataset(content, "hpmor.md")
    return content


if __name__ == "__main__":
    extract_hpmor()

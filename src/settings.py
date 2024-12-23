import os
from datetime import datetime
from pathlib import Path

import anthropic
import openai
import replicate
from dotenv import load_dotenv

from src.logging import setup_logging

load_dotenv()

# General
DATE = datetime.now().strftime("%Y-%m-%d")
PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / "data"
DATASETS_PATH = DATA_PATH / "datasets"
IMAGE_PATH = DATA_PATH / "images"
CACHE_PATH = DATA_PATH / ".cache"
MODELS_PATH = DATA_PATH / "models"
for path in [DATASETS_PATH, IMAGE_PATH, CACHE_PATH, MODELS_PATH]:
    os.makedirs(path, exist_ok=True)

# Clients

## OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_CLIENT = openai.Client(api_key=OPENAI_API_KEY)

## Anthropic
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_CLIENT = anthropic.Client(api_key=ANTHROPIC_API_KEY)

## Replicate
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY")
REPLICATE_CLIENT = replicate_client = replicate.Client(api_token=REPLICATE_API_KEY)

# Logging
setup_logging(debug=True)

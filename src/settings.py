import os
from datetime import datetime
from pathlib import Path

import anthropic
import openai
import replicate
from dotenv import load_dotenv

load_dotenv()

# General
DATE = datetime.now().strftime("%Y-%m-%d")
PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / "data"
DATASETS_PATH = DATA_PATH / "datasets"
IMAGE_PATH = DATA_PATH / "images"

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

import os
from datetime import datetime
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

# General
DATE = datetime.now().strftime("%Y-%m-%d")
PROJECT_PATH = PROJECT_DIR = Path(__file__).resolve().parent.parent

# Clients

## OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_CLIENT = openai.Client(api_key=OPENAI_API_KEY)

## Anthropic
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_CLIENT = anthropic.Client(api_key=ANTHROPIC_API_KEY)

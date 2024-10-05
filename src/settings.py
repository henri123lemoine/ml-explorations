import os
from datetime import datetime

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

DATE = datetime.now().strftime("%Y-%m-%d")

# Clients

## OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_CLIENT = openai.Client(api_key=OPENAI_API_KEY)

## Anthropic
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_CLIENT = anthropic.Client(api_key=ANTHROPIC_API_KEY)

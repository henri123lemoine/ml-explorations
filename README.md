# LLM-explorations

## Description

LLM-explorations is a personal project for experimenting with LLMs.

I want to explore model capabilities, SAEs, fine-tuning techniques, RLHF methods, and more.

## Installation

```bash
git clone https://github.com/henri123lemoine/LLM-explorations.git
cd LLM-explorations
```

## Usage

Use `uv` to run the scripts.

E.g.:

```bash
uv run main.py
uv run -m src.models.qwen
```

## Project Goals

1. Simplify adding new models
2. Handle diverse input types (text, images, etc.)
3. Streamline fine-tuning and reinforcement learning processes

## TODOs

- [x] Implement a flexible Model class
- [x] Update Qwen model implementation
- [x] Create Aria model implementation
- [ ] Develop a system to handle various input types
- [ ] Implement processors for text and image inputs
- [ ] Ensure consistency across different model types
- [ ] Remove unnecessary experiments code & refactor for clarity. Improve configuration
- [ ] Create a basic fine-tuning interface for Qwen
- [ ] Play with finetuning:
  - `lora` or more modern methods
- [ ] Play with RLHF/DPO/IPO/etc. Look into the following:
  - `https://github.com/axolotl-ai-cloud/axolotl`
  - `https://ericmitchell.ai/cdpo.pdf`
  - `https://arxiv.org/pdf/2310.12036`
- [ ] Play with SAEs
  - `https://github.com/jbloomAus/SAELens`

## Testing

Use `uvx pytest tests` to run the tests.

## Acknowledgements

This codebase includes some code produced over the course of COMP551 mini-projects with JX Mo and Fraser Ross Lee.

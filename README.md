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

## TODO

1. Refactor base classes for models, inputs, and outputs
   - [ ] Implement a flexible BaseModel class
   - [ ] Create adaptable BaseInput and BaseOutput classes

2. Implement new models using the refactored structure
   - [ ] Update Qwen model implementation
   - [ ] Create Aria model implementation

3. Enhance input processing
   - [ ] Develop a system to handle various input types
   - [ ] Implement processors for text and image inputs

4. Improve output handling
   - [ ] Refine streaming and non-streaming output support
   - [ ] Ensure consistency across different model types

5. Update Pipeline class
   - [ ] Adapt to work with new model, input, and output implementations
   - [ ] Add support for multi-stage pipelines if needed

6. Develop fine-tuning and RL capabilities
   - [ ] Create a basic fine-tuning interface
   - [ ] Implement initial RLHF, DPO, or IPO support

## Testing

Use `uvx pytest tests` to run the tests.

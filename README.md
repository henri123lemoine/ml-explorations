# Machine Learning Explorations

## Description

ML-explorations is a personal project for experimenting with Machine Learning.

I want to explore model capabilities, SAEs, fine-tuning techniques, RLHF methods, and more.

## Installation

```bash
git clone https://github.com/henri123lemoine/ml-explorations.git
cd ML-explorations
```

## Usage

Use `uv` to run the scripts.

E.g.:

```bash
uv run -m src.models.qwen
```

## Experiments

- [Linear and Logistic Regression with GD vs SGD vs Analytical Solution](src/experiments/comp551/mp1/README.md)
- [Classification of Image Data with MLPs and CNNs](src/experiments/comp551/mp2/README.md)
- [Classification of Textual Data with Naive Bayes and BERT finetunes](src/experiments/comp551/mp3/README.md)
- [Reinforcement Learning](src/experiments/RL/README.md)
  - [Bandit Algorithms](src/experiments/RL/bandit_algorithms/bandit_algorithms.ipynb)
  - [Offline RL](src/experiments/RL/offline_RL/offline_RL.ipynb)
  - [Sarsa & Q-learning & Actor-Critic](src/experiments/RL/Sarsa_Q-Learning_Actor-Critic/Sarsa_Q-Learning_Actor-Critic.ipynb)
  - [DQN](src/experiments/RL/DQN/dqn.py)
  - (WIP) [Muzero](src/experiments/RL/muzero/muzero.ipynb)
  - (WIP) PPO
- [Finetuning](src/experiments/finetuning/README.md):
  - (WIP) [HPMOR-aware Chatbot](src/experiments/finetuning/hpmor/README.md)
- Models:
  - [Qwen Model](src/models/qwen.py)
  - [Flux Model](src/models/flux.py)
  - (WIP) [Whisper Model](src/models/audio/whisper.py)

## Project Goals

1. Simplify adding new models
2. Handle diverse input types (text, images, etc.)
3. Streamline fine-tuning and reinforcement learning processes
4. Add experiments

## TODOs

- [x] Implement a flexible Model class
- [x] Update Qwen model implementation
- [x] Create Aria model implementation
- [ ] Develop a system to handle various input types
- [ ] Implement processors for text and image inputs
- [x] Ensure consistency across different model types
- [ ] Remove unnecessary experiments code & refactor for clarity. Improve configuration
- [ ] [Finetuning](src/experiments/finetuning/README.md)
- [ ] Play with SAEs `https://github.com/jbloomAus/SAELens`

## Testing

Use `uvx pytest tests` to run the tests.

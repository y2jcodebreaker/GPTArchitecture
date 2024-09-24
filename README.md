# GPT Language Model

This repository contains an implementation of a GPT (Generative Pre-trained Transformer) language model built using PyTorch. The model leverages multi-head self-attention and feedforward neural networks to generate text based on the input context.

## Overview

The GPT language model is designed to predict the next token in a sequence, making it suitable for various natural language processing tasks, such as text generation, completion, and more. The architecture consists of multiple transformer blocks, each incorporating self-attention mechanisms and feedforward layers.

## Features

- **Token and Position Embeddings**: The model uses embedding layers to convert input tokens into dense vectors and adds positional information.
- **Multi-Head Attention**: Implements multi-head attention to allow the model to focus on different parts of the input sequence simultaneously.
- **Feedforward Neural Network**: Each transformer block contains a feedforward network for additional processing of the attention outputs.
- **Layer Normalization**: Applied after each sub-layer to stabilize the training process.
- **Text Generation**: Includes a `generate` method for producing text based on a given context.****

## Requirements

- Python 3.x
- PyTorch
- NumPy

## Installation

To get started, clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/gpt-language-model.git
cd gpt-language-model
pip install torch numpy
```

## Usage

## Training
To train the model, you can use the following code snippet:
```bash
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## Text Generation
To generate text, use the generate method of the GPTLanguageModel class:
```bash
# Example usage
model = GPTLanguageModel()
model.to(device)
input_indices = torch.tensor([[...]]).to(device)  # Replace with your input tensor
generated_text = model.generate(input_indices, max_new_tokens=1000)
print(decode(generated_text[0].tolist()))
```

# Model Parameters
The model contains approximately 0.078657 million parameters, making it capable of capturing complex patterns in language data.

# Acknowledgements
This implementation is based on tutorials available on YouTube and various resources on transformer models. Special thanks to the Andrej Karpathy who provided guidance throughout the coding process.

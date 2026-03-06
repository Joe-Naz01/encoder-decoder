# Building Transformers from Scratch: A PyTorch Implementation 

This repository contains a modular, from-the-ground-up implementation of the Transformer architecture using PyTorch. It is designed to serve as both a learning resource and a foundation for sequence-to-sequence research.
It is a deep-dive into the internal mechanics of the Transformer model, as originally proposed in the ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762) paper. Instead of utilizing high-level libraries, this repository features a modular, ground-up implementation using PyTorch.

The project is divided into three major architectural milestones:

- The Encoder: 
               - Implementation of Multi-Head Self-Attention and Feed-Forward Sublayers.
               - Features Positional Encoding to help the model understand word order and sequence structure.
               - Includes a classification head for tasks like sentiment analysis.

- The Decoder: 
               - Implementation of Causal (Masked) Self-Attention to prevent the model from "cheating" by looking at future tokens during training.
               - Construction of the decoder body to project internal states into vocabulary-sized probabilities.

- The Full Encoder-Decoder Bridge:
               - Creation of the Cross-Attention mechanism, which allows the decoder to "look back" at the encoder's output.
               - Integration of all modules into a single Transformer class capable of handling full sequence-to-sequence tasks.

## Project Highlights
- **Modular Design:** Every component (Multi-Head Attention, Positional Encoding, Feed-Forward layers) is built as a standalone PyTorch module.
- **Full Encoder-Decoder:** Implements the complete architecture, including the critical Cross-Attention bridge.
- **Masking Logic:** Features custom implementation for both Source Masks and Causal (Triangular) Target Masks to ensure proper training logic.

## Repository Contents
- **`encoder_decoder.ipynb`**: The master notebook containing the step-by-step construction of the Transformer.
- **Custom Modules:** Ground-up code for `MultiHeadAttention`, `EncoderLayer`, `DecoderLayer`, and `PositionalEncoding`.

## Environment Setup (Conda)

### 1. Create and Activate the Environment
```bash
git clone https://github.com/Joe-Naz01/encoder-decoder.git
cd encoder-decoder

conda create -n transformer_scratch python=3.10 -y
conda activate transformer_scratch

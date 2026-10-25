# char-gpt-shakespeare

A character-level GPT written from scratch in PyTorch (~212K parameters), trained on Tiny Shakespeare on a CPU-only laptop. It includes a dropout experiment and sampling controls (temperature and top-k).

This project was built to understand how transformers work, line by line. It is a learning project, not a production model.

## What it does:

The model reads 64 characters at a time and predicts the next one. Trained on about 1 MB of Shakespeare, it learns the shape of a play script: speaker names in capitals followed by colons, line breaks, short lines, and mostly real-looking words.

```
KING RICHARD II:
I would the no, that noble come my eyes,
For thathink, dreamen wath our plant her,
Tranio inducted it them comment, hence lime
From the fall found deprent not were, thought hast madder
```

*Sampled at temperature 0.8, top-k 20. The format is learned well. The meaning is not, and invented words like "wath" and "dreamen" show where a model this small runs out of capacity.*

## Results:

<img width="365" height="161" alt="llm analysis" src="https://github.com/user-attachments/assets/31e73bc8-96bf-4392-ab94-4cff6eb3a28c" />


What I learned:
- **Attention matters.** A bigram baseline (a lookup table that sees only the previous character) stalled near 2.4 in my earlier runs. This transformer reached 1.705 validation loss. The bigram was measured separately and is not included in this repo.
- **A smaller train/val gap is not the goal.** Dropout 0.2 shrank the gap from 0.211 to 0.145, but validation loss got worse (1.774 vs 1.705). The model is small and still under-trained at 3,000 steps, so dropout slowed learning instead of fixing overfitting. Both losses were still falling when training stopped.
- **Sampling changes the text without changing the model.** On the same trained weights, temperature 0.5 was tidy but repetitive, 1.3 was messy, and temperature 0.8 with top-k 20 gave the best balance in my judgment. Top-k removes the long tail of unlikely characters that cause invented words.
- **Change one thing at a time.** Both runs share the same seed and settings except dropout, so the difference can be attributed to dropout alone.

## Architecture:
characters → token embedding + position embedding
           → [ LayerNorm → multi-head self-attention → add → LayerNorm → feed-forward → add ] × 4
           → LayerNorm → linear layer → scores for the next character

| Vocabulary = 65 characters |
| Context length (`block_size`) = 64 |
| Embedding size (`n_embd`) = 64 |
| Attention heads (`n_head`) = 4 (16 dimensions each) |
| Blocks (`n_layer`) = 4 |
| Feed-forward width = 4 × `n_embd`, with ReLU |
| Batch size = 32 |
| Optimizer = AdamW, learning rate 3e-3 |
| Parameters = 211,777 |

Details:
- Causal self-attention: a lower-triangular mask stops each position from seeing later positions.
- Attention scores are scaled by 1/√(head size) before softmax.
- Residual connections around both sub-layers, with LayerNorm applied before each.
- Learned position embeddings.
- Generation crops the context to the last 64 characters and supports temperature and top-k sampling.

## Quickstart
In your terminal:
```
git clone https://github.com/AbirGoated/char-gpt-shakespeare.git
cd char-gpt-shakespeare
pip install torch
python llm.py
```
The repo includes the trained weights, so the default run loads the model and prints four samples at different sampling settings. The script downloads the Tiny Shakespeare text each run, so it needs an internet connection.
To train from scratch, set this at the top of `llm.py`:
TRAIN = True

Training takes roughly 30 minutes on a CPU laptop. It prints a progress line every 500 steps, a markdown results table at the end, and saves the weights to `gpt.pt`. Set `TRAIN = False` afterwards to reload them. To try the dropout experiment, change `dropout = 0.2` and use a different `CHECKPOINT` name so you keep both models.

## Limitations:
- It imitates the *format* of Shakespeare. It does not understand language and cannot answer questions or follow instructions.
- At about 212K parameters and 1 MB of text, it invents words and loses track of meaning within a sentence.
- Character-level tokenization makes sequences long and spelling errors common. A sub-word (BPE) tokenizer would help.
- Results come from single runs with one seed, so small differences in loss should not be over-interpreted.

  
## Acknowledgements:
- Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video and the [nanoGPT](https://github.com/karpathy/nanoGPT) project, which this implementation follows closely in structure.
- The Tiny Shakespeare dataset, from Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) repository. The underlying text is Shakespeare's and is in the public domain.

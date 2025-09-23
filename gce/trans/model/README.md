---
base_model: google/gemma-3-4b-pt
library_name: transformers
model_name: gemma-3-4b-trans-en-ko
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for gemma-3-4b-trans-en-ko

This model is a fine-tuned version of [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="sayouzone25/gemma-3-4b-trans-en-ko", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.15.2
- Transformers: 4.55.2
- Pytorch: 2.8.0
- Datasets: 3.3.2
- Tokenizers: 0.21.4

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallouédec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```
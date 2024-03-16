# Stealing Part of a Language Model
An unofficial implementation of ["Stealing Part of a Production Language Model"](https://arxiv.org/abs/2403.06634)

### Details
Attack reimplementations are for research and model safety/defense purposes alone.

Llama 2 7b:
- Recover hidden dim $\pm 1$: 4095
- Predict RMSNorm as normalization layer
- Last layer reconstructed with an RMS of $2 * 10^{-5}$

### Techniques
With All Logits Available:
- [x] Recovering Hidden Dimensionality
    - [x] Normalization Layer Prediction
- [x] Full Final Layer Extraction

With Top-K Logits and Logit-bias
- [ ] Recover complete logit vector
    - [ ] constrained logit-bias case

Logprob-free
- [ ] Recover complete logit vector


### Citation
```bibtex
@misc{carlini2024stealing,
    title={Stealing Part of a Production Language Model}, 
    author={Nicholas Carlini and Daniel Paleka and Krishnamurthy Dj Dvijotham and Thomas Steinke and Jonathan Hayase and A. Feder Cooper and Katherine Lee and Matthew Jagielski and Milad Nasr and Arthur Conmy and Eric Wallace and David Rolnick and Florian Tramèr},
    year={2024},
    eprint={2403.06634},
    archivePrefix={arXiv},
    primaryClass={cs.CR}
}
```

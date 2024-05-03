# Stealing Part of a Language Model
An unofficial implementation of ["Stealing Part of a Production Language Model"](https://arxiv.org/abs/2403.06634)

### Details
Attack reimplementations are for research and model safety/defense purposes alone. We don't use any proprietary API for the detailed experiments.

Llama 2 7b:
- Recover hidden dim $\pm 1$: 4095
- Predict RMSNorm as normalization layer
- Last layer reconstructed with an RMS of $2 * 10^{-5}$

### Methods
With All Logits Available:
- [x] Recovering Hidden Dimensionality
    - [x] Normalization Layer Prediction
- [x] Full Final Layer Extraction

With Top-K Logits and Logit-bias
- [x] Recover complete logit vector
    - [x] Top-K logits
    - [x] Top-K logprobs
    - [x] Cost-optimal Top-K logprobs variant  
    - [x] Top-1 Logprob
        - Have not tested logit recovery due to limited resources. 

Logprob-free
- [x] Recover complete logit vector
    - [x] Binary Search
    - [x] Hyperrectangle Relaxation Center
        - [x] With better queries
        - Bounding methods can be referenced [here](https://github.com/dpaleka/stealing-part-lm-supplementary/tree/main/optimize_logit_queries/bounders).
    - Have not tested logit recovery due to limited resources. 

### Extras
- [ ] Optimized Top-K logprobs method with linear constraint
- [ ] Shortest path formulation of logprob-free attack

### Citation
Authors published their own supplementary code after this repo was made, so please do reference theirs for any additional necessary clarity. You can find their repository [here](https://github.com/dpaleka/stealing-part-lm-supplementary).

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

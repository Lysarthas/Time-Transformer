# Time-Transformer

<p align="center">
<img src=imgs/timetransformer.png />
</p>

Pytnon implementation paper "[**Time-Transformer: Integrating Local and Global Features for Better Time Series Generation**](https://epubs.siam.org/doi/10.1137/1.9781611978032.37)" (SDM24).

**Updates**:

*2024.09.17* -- **Time-Transformer** was applied for time series representation learning / anomaly detection task, and achieved third place in *Volvo Discovery Challenge at ECML-PKDD 2024*.

*2025.05.11* -- Recent extension project revealed the difficulty of adversarial training. Thus, we provide VAE based surrogate with time-transformer module. We use the VAE backbone implemented in "[*TimeVAE*](https://github.com/abudesai/timeVAE)".

**Instructions**:

Jupyter Notebook "**tutorial**" provide a tutorial for training and evaluating with different metrics (using "**sine_cpx**" dataset). FID score are calculated with "**fid_score**" in `ts2vec`, directly using model "[*TS2Vec*](https://github.com/yuezhihan/ts2vec)".

The model is built with "*tensorflow2*", please check the "*requirement.txt*" and decide which package you need to run the model.

If you find this model useful and put it in your publication, we encourage you to add the following references:
```bibtex
@inproceedings{liu2024time,
  title={Time-Transformer: Integrating Local and Global Features for Better Time Series Generation},
  author={Liu, Yuansan and Wijewickrema, Sudanthi and Li, Ang and Bester, Christofer and O'Leary, Stephen and Bailey, James},
  booktitle={Proceedings of the 2024 SIAM International Conference on Data Mining (SDM)},
  pages={325--333},
  year={2024},
  organization={SIAM}
}
```

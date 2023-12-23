# Time-Transformer

<p align="center">
<img src=imgs/timestransformer.png />
</p>

Pytnon implementation paper "[**Time-Transformer: Integrating Local and Global Features for Better Time Series Generation**](https://arxiv.org/abs/2312.11714)" (SDM24).

Jupyter Notebook "**tutorial**" provide a tutorial for training and evaluating with different metrics. FID score are calculated with "**fid_score**" in `ts2vec`, directly using model "[**TS2Vec**](https://github.com/yuezhihan/ts2vec)".

The model is built with "*tensorflow2*", please check the "**requirement.txt**" and decide which package you need to run the model.

If you find this model useful and put it in your publication, we encourage you to add the following references:
```bibtex
@misc{liu2023timetransformer,
      title={Time-Transformer: Integrating Local and Global Features for Better Time Series Generation}, 
      author={Yuansan Liu and Sudanthi Wijewickrema and Ang Li and Christofer Bester and Stephen O'Leary and James Bailey},
      year={2023},
      eprint={2312.11714},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

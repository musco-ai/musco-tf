# ![](https://user-images.githubusercontent.com/11778655/66068156-bef1a880-e555-11e9-8d26-094071133a11.png) MUSCO: Multi-Stage COmpression of neural networks

This repository contains supplementary code for the paper [MUSCO: Multi-Stage COmpression of neural networks](https://arxiv.org/pdf/1903.09973.pdf). 
It demonstrates how a neural network with convolutional and fully connected layers can be compressed using iterative tensor decomposition of weight tensors.

## Requirements
```
numpy
scipy
scikit-tensor-py3
tensorly-musco
absl-py
tqdm
tensorflow-gpu (TensorRT support)
```

## Installation
```
pip install musco-tf
```

## Quick Start
```python
from musco.tf import CompressorVBMF, Optimizer

model = load_model("model.h5")
compressor = CompressorVBMF(model)

while True:
    model = compressor.compress_iteration(number=5)
    
    # Fine-tune compressed model.

# Compressor decomposes 5 layers on each iteration
# and returns compressed model. You have to fine-tune
# model after each iteration to restore accuracy.
# Compressor automatically selects the best parameters
# for decomposition on each iteration.

# You can freeze and quantize model after compression.
optimizer = Optimizer(precision="FP16", max_batch_size=16)
optimizer.freeze(model)
optimizer.optimize("frozen.pb")
```

## Citing
If you used our research, we kindly ask you to cite the corresponding [paper](https://arxiv.org/abs/1903.09973).

```
@article{gusak2019one,
  title={MUSCO: Multi-Stage Compression of neural networks},
  author={Gusak, Julia and Kholiavchenko, Maksym and Ponomarev, Evgeny and Markeeva, Larisa and Oseledets, Ivan and Cichocki, Andrzej},
  journal={arXiv preprint arXiv:1903.09973},
  year={2019}
}
```

## License
Project is distributed under [Apache License 2.0](https://github.com/musco-ai/musco-tf/blob/master/LICENSE).
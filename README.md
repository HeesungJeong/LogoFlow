

## LogoFlow- Pytorch

Implementation of <a href="https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html">rectified flow</a> and some of its followup research


<img src="./images/oxford-flowers.sample.png" width="350px"></img>

*32 batch size, 11k steps oxford flowers*

## Install

```bash
$ pip install rectified-flow-pytorch
```

## Usage

```python
import torch
from rectified_flow_pytorch import RectifiedFlow, Unet

model = Unet(dim = 64)

rectified_flow = RectifiedFlow(model)

images = torch.randn(1, 3, 256, 256)

loss = rectified_flow(images)
loss.backward()

sampled = rectified_flow.sample()
assert sampled.shape[1:] == images.shape[1:]
```

# do the above for many real images

reflow = Reflow(rectified_flow)

reflow_loss = reflow()
reflow_loss.backward()

# then do the above in a loop many times for reflow - you can reflow multiple times by redefining Reflow(reflow.model) and looping again

sampled = reflow.sample()
assert sampled.shape[1:] == images.shape[1:]
```

With a `Trainer` based on `accelerate`

```python
import torch
from rectified_flow_pytorch import RectifiedFlow, ImageDataset, Unet, Trainer

model = Unet(dim = 64)

rectified_flow = RectifiedFlow(model)

img_dataset = ImageDataset(
    folder = './path/to/your/images',
    image_size = 256
)

trainer = Trainer(
    rectified_flow,
    dataset = img_dataset,
    num_train_steps = 70_000,
    results_folder = './results'   # samples will be saved periodically to this folder
)

trainer()
```

## Examples

Quick test on oxford flowers

```bash
$ pip install .[examples]
```

Then

```bash
$ python train_oxford.py
```

## Citations

```bibtex
@article{Liu2022FlowSA,
    title   = {Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
    author  = {Xingchao Liu and Chengyue Gong and Qiang Liu},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2209.03003},
    url     = {https://api.semanticscholar.org/CorpusID:252111177}
}
```


```bibtex
@article{Park2025FlowQ,
    title   = {Flow Q-Learning},
    author  = {Seohong Park and Qiyang Li and Sergey Levine},
    journal = {ArXiv},
    year    = {2025},
    volume  = {abs/2502.02538},
    url     = {https://api.semanticscholar.org/CorpusID:276107180}
}
```

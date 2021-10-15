# Contrastive Predictive Coding

The CPC-big model and k-means checkpoint used in the paper [Analyzing Speaker Information in Self-Supervised Models to Improve Zero-Resource Speech Processing](https://arxiv.org/abs/2108.00917).

Contrastive predictive coding (CPC) aims to learn representations of speech by distinguishing future observations from a set of negative examples. 
Previous work has shown that linear classifiers trained on CPC features can accurately predict speaker and phone labels. 
However, it is unclear how the features actually capture speaker and phonetic information, and whether it is possible to normalize out the irrelevant details (depending on the downstream task). 
In this paper, we first show that the per-utterance mean of CPC features captures speaker information to a large extent. 
Concretely, we find that comparing means performs well on a speaker verification task. 
Next, probing experiments show that standardizing the features effectively removes speaker information. 
Based on this observation, we propose a speaker normalization step to improve acoustic unit discovery using K-means clustering of CPC features. 
Finally, we show that a language model trained on the resulting units achieves some of the best results in the ZeroSpeech2021~Challenge.

## Basic Usage

```python
import torch, torchaudio
from sklearn.preprocessing import StandardScaler

# Load model checkpoints
cpc = torch.hub.load("bshall/cpc:main", "cpc").cuda()
kmeans = torch.hub.load("bshall/cpc:main", "kmeans50")

# Load audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

x = cpc.encode(wav).squeeze().cpu().numpy()  # Encode
x = StandardScaler().fit_transform(x)  # Speaker normalize
codes = kmeans.predict(x)  # Discretize
```

Note that the `encode` function is stateful (keeps the hidden state of the LSTM from previous calls).

## Encode an Audio Dataset

Clone the repo and use the `encode.py` script:

```
usage: encode.py [-h] in_dir out_dir

Encode an audio dataset using CPC-big (with speaker normalization and discretization).

positional arguments:
  in_dir      Path to the directory to encode.
  out_dir     Path to the output directory.

optional arguments:
  -h, --help  show this help message and exit
```
# cpc
The CPC-big model from https://arxiv.org/abs/2011.11588

```python
import torch, torchaudio
from sklearn.preprocessing import StandardScaler

# Load model checkpoints
cpc, kmeans = torch.hub.load("bshall/cpc", "cpc")
cpc = cpc.cuda()

# Load audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

x = cpc.encode(wav).squeeze().cpu().numpy()  # Encode
x = StandardScaler().fit_transform(x)  # Speaker normalize
codes = kmeans.predict(x)  # Discretize
```
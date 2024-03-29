# Adaptive Multi-Scale Positional Encoding (AM-SPE)
Designed to enhance Transformer-based models by introducing dynamic, scale-adaptive positional encodings.


## Mathematical Formula

### Standard Sinusoidal Positional Encoding

The formula for standard sinusoidal positional encoding for a position $i\$ and dimension $d\$ is defined as follows:

```math
PE(i, 2k) = \sin\left(\frac{i}{10000^{2k/D}}\right)
```

```math
PE(i, 2k+1) = \cos\left(\frac{i}{10000^{2k/D}}\right)
```

where:
- $i\$ is the position in the sequence,
- $d\$ is the dimension, with $k\$ being the floor of $d/2\$,
- $D\$ is the total number of dimensions in the positional encoding vector.

### Coarse and Detailed Encodings

Coarse and detailed positional encodings modify the scale of the sinusoidal functions:

#### Coarse Encoding

```math
PE_{coarse}(i, 2k) = \sin\left(N_{coarse} \cdot \frac{i}{10000^{2k/D}}\right)
```

```math
PE_{coarse}(i, 2k+1) = \cos\left(N_{coarse} \cdot \frac{i}{10000^{2k/D}}\right)
```

#### Detailed Encoding

```math
PE_{detailed}(i, 2k) = \sin\left(\frac{i}{10000^{2k/D}}\right)
```

```math
PE_{detailed}(i, 2k+1) = \cos\left(\frac{i}{10000^{2k/D}}\right)
```

### Adaptive Encoding

The adaptive encoding is a blend of the coarse and detailed encodings, weighted by a parameter $\\alpha\$:

```math
PE_{adaptive}(i, d) = \alpha \cdot PE_{detailed}(i, d) + (1 - \alpha) \cdot PE_{coarse}(i, d)
```

where:
- $\\alpha\$ is the adaptivity parameter, with a range of $\[0, 1]\$, determining the blend ratio between detailed and coarse encodings.



## Wolfram Notebook

https://www.wolframcloud.com/obj/4b210575-9a1a-4597-b2aa-58359b706a3f

## MATLAB

```Matlab
% AdaptiveMultiScalePositionalEncoding.m
classdef AdaptiveMultiScalePositionalEncoding
    properties
        d_model
        max_len
        alpha
        pe_coarse
        pe_detail
    end

    methods
        function obj = AdaptiveMultiScalePositionalEncoding(d_model, max_len)
            if nargin > 0
                obj.d_model = d_model;
                obj.max_len = max_len;
                obj.alpha = 0;
                
                position = (0:(max_len-1))';
                div_term = exp((0:2:(d_model-1)) * -(log(10000.0) / d_model));
                
                pe_coarse = zeros(max_len, d_model);
                pe_coarse(:, 1:2:end) = sin(position * div_term * 10);
                pe_coarse(:, 2:2:end) = cos(position * div_term * 10);
                obj.pe_coarse = pe_coarse;
                
                pe_detail = zeros(max_len, d_model);
                pe_detail(:, 1:2:end) = sin(position * div_term);
                pe_detail(:, 2:2:end) = cos(position * div_term);
                obj.pe_detail = pe_detail;
            else
                error('Not enough input arguments.');
            end
        end
        
        function output = encode(obj, seq_len, detail_level)
            if nargin < 3
                detail_level = 0.5;
            end
            
            alpha = sigmoid(obj.alpha);
            adaptive_pe = alpha * obj.pe_coarse(1:seq_len, :) + ...
                (1 - alpha) * detail_level * obj.pe_detail(1:seq_len, :);
            
            output = adaptive_pe;
        end
    end
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end
```

```Matlab
% useEncoding.m
d_model = 512;
max_len = 1000;

am_spe = AdaptiveMultiScalePositionalEncoding(d_model, max_len);

seq_len = 100;
detail_level = 0.5;
adaptive_pe = am_spe.encode(seq_len, detail_level);

% Display size of the adaptive positional encoding
disp(size(adaptive_pe));

```

## Python

```python
import torch
import torch.nn as nn
import numpy as np

class AdaptiveMultiScalePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(AdaptiveMultiScalePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Learnable parameter to balance coarse and detailed encodings
        self.alpha = nn.Parameter(torch.zeros(1))

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe_coarse = torch.zeros(max_len, d_model)
        pe_coarse[:, 0::2] = torch.sin(position * div_term * 10)
        pe_coarse[:, 1::2] = torch.cos(position * div_term * 10)
        self.register_buffer('pe_coarse', pe_coarse.unsqueeze(0))

        pe_detail = torch.zeros(max_len, d_model)
        pe_detail[:, 0::2] = torch.sin(position * div_term)
        pe_detail[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_detail', pe_detail.unsqueeze(0))

    def forward(self, x, detail_level=0.5):
        """
        x: Input embeddings (batch_size, seq_len, d_model)
        detail_level: Threshold to determine the level of detail needed (0 to 1)
        """
        seq_len = x.size(1)
        alpha = torch.sigmoid(self.alpha)

        adaptive_pe = alpha * self.pe_coarse[:, :seq_len] + (1 - alpha) * self.pe_detail[:, :seq_len] * detail_level

        x = x + adaptive_pe
        return x
```

```python
# Example
d_model = 512
max_len = 1000
batch_size = 32
seq_len = 100

# Simulate embeddings
input_embeddings = torch.randn(batch_size, seq_len, d_model)

am_spe_layer = AdaptiveMultiScalePositionalEncoding(d_model, max_len)

output_embeddings = am_spe_layer(input_embeddings, detail_level=0.5)

print(output_embeddings.shape)
print(output_embeddings)
```

## Citation

If you use Adaptive Multi-Scale Positional Encoding (AM-SPE) in your research, please cite the following work:

```bibtex
@misc{AdaptiveMultiScalePositionalEncoding-2024,
  author = {Aakash Apoorv},
  title = {Adaptive Multi-Scale Positional Encoding (AM-SPE)},
  year = {2024},
  howpublished = {\url{https://github.com/ToyMath/AdaptiveMultiScalePositionalEncoding}},
}
```

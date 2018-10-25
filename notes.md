
# Unsqueezing
```python
>>> ts = [torch.tensor(np.random.randn(2, 84, 84)) for _ in range(32)]
>>> torch.sum(torch.cat(ts, 0).view(-1, 2, 84, 84) - torch.cat(
    [t.unsqueeze(0) for t in ts] , 0))
tensor(0., dtype=torch.float64)
```

# Errors
Error: `The number of sizes provided must be greater or equal to the number of dimensions in the tensor`

This may mean that you are trying to set the value of a slice of
a tensor to have more dimensions than in the slice, e.g.

```python
import torch
t = torch.tensor([1, 2, 3])
t[0] = torch.tensor([[1], [2]])
```

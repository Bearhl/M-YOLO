import torch
import numpy as np

output = torch.randn((2, 1, 4, 4))
output = torch.sigmoid(output)
print(torch.argmax(output, dim=1).shape)
print(output)

for score in output:
    T = score.shape[-1]
    print(score.cpu().numpy().ravel())
    sorted_indexes = np.dstack(np.unravel_index(np.argsort(score.cpu().numpy().ravel())[::-1], (T, T))).tolist()
    print(np.dstack(np.unravel_index(np.argsort(score.cpu().numpy().ravel())[::-1], (T, T))).tolist())
    print(len(sorted_indexes[0]))

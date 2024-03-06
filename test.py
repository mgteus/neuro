import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

def test_linear():
    m = nn.Linear(2, 1)
    print('m.weight', m.weight)
    print('m.weight.size()', m.weight.size())
    input = torch.rand(10, 2)

    fig, ax = plt.subplots(1, 3, )
    ax[0].imshow(m.weight.detach().numpy())
    ax[0].set_title('weights')
    ax[1].imshow(input.detach().numpy())
    ax[1].set_title('input')
    output = m(input)
    ax[2].imshow(output.detach().numpy())
    ax[2].set_title('output')
    plt.tight_layout()
    plt.show()
    return 




if __name__ == '__main__':

    test_linear()

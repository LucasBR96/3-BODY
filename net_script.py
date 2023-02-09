from MODEL.dsamp import sampler , stellarDset
from MODEL.network_model import stellar_model

import torch as tc
import torch.nn as tnn
import torch.functional as tfn
import torch.optim as top
import torch.utils.data as tdt


device = "cuda" if tc.cuda.is_available() else "cpu"
lr = 1e-3
tr_batch_size = 100
ts_batch_size = 25
epochs = 100

loss_fn = tnn.MSELoss()
model = stellar_model().to( device )
opm = top.SGD( model.params() )
train_data = iter(
    tdt.DataLoader(
        stellarDset( 0 ),
        batch_size = tr_batch_size,
        shuffle = True
    )
)
test_data = iter(
    tdt.DataLoader(
        stellarDset( 1 ),
        batch_size = ts_batch_size,
        shuffle = True
    )
)

if __name__ == "__main__":
    
    tr_scores = []
    ts_scores = []

    for _ in range( epochs ):

        X , y = next( train_data )
        X , y = X.to( device  ) , y.to( device )
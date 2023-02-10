from MODEL.dsamp import sampler , stellarDset
from MODEL.network_model import stellar_model

import torch as tc
import torch.nn as tnn
import torch.functional as tfn
import torch.optim as top
import torch.utils.data as tdt

import matplotlib.pyplot as plt
from random import shuffle


device = "cuda" if tc.cuda.is_available() else "cpu"
lr = 1e-4
tr_batch_size = 500
ts_batch_size = 500
epochs = 6*( 10**6 )

loss_fn = tnn.L1Loss()
model = stellar_model().to( device )
opm = top.Adam( model.parameters() , lr = lr )

data_sets = list( range( 50 ) )
shuffle( data_sets )
train_data = sampler(
    batch_size = tr_batch_size,
    sets = data_sets[:40]
)

test_data = sampler(
    batch_size = ts_batch_size,
    sets = data_sets[10:]
)


if __name__ == "__main__":
    
    tr_scores = []
    ts_scores = []

    k = 0
    mean_test_loss = 0
    mean_train_loss = 0

    for i in range( epochs ):

        if i%1000 == 0:
            with tc.no_grad():

                X , y = test_data.fetch_data()
                X = X.to( device )
                y = y.to( device )

                y_hat = model( X )
                test_loss_val = loss_fn( y_hat , y ).item()
                
        
        # X , y = next( train_data )
        X , y = train_data.fetch_data()
        X = X.to( device )
        y = y.to( device )

        y_hat = model( X )
        loss = loss_fn( y_hat , y )
        train_loss_val = loss.item()

        opm.zero_grad()
        loss.backward()
        opm.step()

        #Printing the data -----------------------------
        if i%1000 == 0:

            print( f"iter #{i} " + "-"*25 )
            print(f"loss at training: {train_loss_val:.5f}")
            print(f"loss at testing: {test_loss_val:.5f}")
            print()

            print( y[ 0 ] )
            print( y_hat[0] )
            print()

            tr_scores.append( train_loss_val )
            ts_scores.append( test_loss_val )

    #---------------------------
    plt.plot( range( len( tr_scores ) ) , tr_scores , '-r' , label = "train")
    plt.plot( range( len( tr_scores ) ) , ts_scores , '-b' , label = "test")
    plt.legend()
    plt.title( "score over time" )
    plt.show()

from dsamp import sampler , stellarDset
from network_model import stellar_model
from clock import clock
from mov_avg import mob_mean_gen

import torch as tc
import torch.nn as tnn
import torch.functional as tfn
import torch.optim as top
import torch.utils.data as tdt

from typing import *
from random import shuffle
import sys

device = "cuda" if tc.cuda.is_available() else "cpu"

# CK : clock = clock()

class train_app:

    def __init__( self , **kwargs ):

        data_sets = list( range( 100 ) )
        shuffle( data_sets )

        tr_batch_size = kwargs.get('tr_batch_size' , 500 )
        self.train_data = sampler(
            batch_size = tr_batch_size,
            sets = data_sets[:80]
        )

        ts_batch_size = kwargs.get('ts_batch_size' , 500 )
        self.test_data = sampler(
            batch_size = ts_batch_size,
            sets = data_sets[80:]
        )

        self.loss_fn = tnn.L1Loss()
        self.model = stellar_model().to( device )
        lr = kwargs.get( "lr" , 1e-4 )
        self.opm = top.Adam( self.model.parameters() , lr = lr )

        self.iter = 0
        self.record_interval = kwargs.get( "record_interval" , 1000 )
        self.test_rec = mob_mean_gen()
        self.train_rec = mob_mean_gen()

        self.min_loss = sys.maxsize
        self.i_min_loss = None

        self.buff = []
        self.buff_lim = 100

        max_time = kwargs.get( "max_time" , 12 )
        time_type = kwargs.get( "time_type" , "hours" )
        self.ck = clock( max_time , time_type )

    def run( self ):
        
        @self.ck.tick()
        def update():
            self._update_net()
        
        while not self.ck.is_done():

            if self.iter%self.record_interval == 0:

                #---------------------------------------
                # Comparative performance of the network
                # on the training and testing sets
                rec = self._generate_record()

                #----------------------------------------
                # printing the result of the last function
                # on screen
                self._print_rec( rec )

                #-----------------------------------------
                # saving the last record on a buffer. and if
                # the buffer is full, save it on a csv file
                self._push_rec( rec )
                if len( self.buff ) >= self.buff_lim:
                    self._save_buff()

                #------------------------------------------
                # saving the model, if the performance on the
                # test data set has improoved
                self._save_model( rec[ 1 ] )
            
            #------------------------------------------
            # Doing one iteration of the backprop algorithm
            update()

            self.iter += 1

    def _print_rec( self , rec ):

        iter_num , ts_val , tr_val , y , y_hat = rec

        print( f"iter #{iter_num} " + "-"*25 )
        print(f"loss at training: {tr_val:.5f}")
        print(f"loss at testing: {ts_val:.5f}")
        print()

        print( y[ 0 ] )
        print( y_hat[0] )
        print()         

    def _push_rec( self , rec ):

        iter_num , ts_val , tr_val , _ , _ = rec 
        self.buff.append(
            ( iter_num, 
            ts_val,
            tr_val)
        )

    def _save_buff( self ):

        path = "DATA/performance.csv"
        with open( path , "a" ) as f:
            if self.iter == 0:
                f.write(
                    "iter_num,ts_loss,tr_loss" + "\n"
                )
            for rec in self.buff:
                iter_num,ts_loss,tr_loss = rec
                f.write(
                    f"{iter_num},{ts_loss},{tr_loss}" + "\n"
                )
        self.buff.clear() 
    
    def _generate_record( self ):

        with tc.no_grad():

            X , y = self.train_data.fetch_data()
            X = X.to( device )
            y_tr = y.to( device )

            y_hat_tr = self.model( X )
            tr_val = self.loss_fn( y_hat_tr , y_tr ).item()
            tr_val = self.train_rec( tr_val )

            X , y = self.test_data.fetch_data()
            X = X.to( device )
            y = y.to( device )

            y_hat = self.model( X )
            ts_val = self.loss_fn( y_hat , y ).item()
            ts_val = self.test_rec( ts_val )

            return (
                self.iter,
                ts_val,
                tr_val,
                y_tr[ 0 ],
                y_hat_tr[ 0 ]
            )

    def _save_model( self , ts_val : float ):

        if ts_val >= self.min_loss:
            return
        
        self.min_loss = ts_val
        self.i_min_loss = self.iter

        tc.save(
            self.model.parameters(),
            "DATA/model_params.pt"
        )

    # @CK.tick()
    def _update_net( self ):

        X , y = self.train_data.fetch_data()
        X = X.to( device )
        y = y.to( device )

        y_hat = self.model( X )
        loss = self.loss_fn( y_hat , y )

        self.opm.zero_grad()
        loss.backward()
        self.opm.step()

if __name__ == "__main__":

    from time import sleep
    cm = clock()

    @cm.tick()
    def boo( t ):
        t._update_net()
        sleep( .1 )

    t = train_app()
    print( "tempo em ck | tempo em cm" )
    for _ in range( 10 ):
        boo( t )
        # t._update_net()
        s1 = f"{ t.ck.base:.5f}".rjust( 11 )
        s2 = f"{ cm.base:.5f}".rjust( 11 )
        print( s1 + " | " + s2 )
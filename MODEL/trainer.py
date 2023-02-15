try:
    from dsamp import sampler , stellarDset
    from network_model import stellar_model
    from clock import clock
    from mov_avg import mob_mean_gen

except ModuleNotFoundError:
    from MODEL.dsamp import sampler , stellarDset
    from MODEL.network_model import stellar_model
    from MODEL.clock import clock
    from MODEL.mov_avg import mob_mean_gen

import torch as tc
import torch.nn as tnn
import torch.functional as tfn
import torch.optim as top
import torch.utils.data as tdt
import pandas as pd

from typing import *
from random import shuffle
import sys
import os

device = "cuda" if tc.cuda.is_available() else "cpu"

class mod_kern:

    def __init__( self , **kwargs ):

        loss_type = kwargs.get( "loss_fn" , tnn.L1Loss )
        self.loss_fn = loss_type()

        self.model = stellar_model().to( device )
        self.starting_params = self.model.state_dict()

        lr = kwargs.get( "lr" , 1e-4 )
        opm_type = kwargs.get( "opm" , top.Adam )
        self.opm = opm_type( self.model.parameters() , lr = lr )
    
    def update( self , X , y ):

        X = X.to( device )
        y = y.to( device )

        y_hat = self.model( X )
        loss = self.loss_fn( y_hat , y )

        self.opm.zero_grad()
        loss.backward()
        self.opm.step()
    
    def evaluate( self, X_train , y_train , X_test, y_test ):

        tups = [
            ( X_train , y_train ),
            ( X_test , y_test )
        ]

        losses = []
        with tc.no_grad():
            for X , y in tups:
                X = X.to( device )
                y = y.to( device )

                y_hat = self.model( X )
                loss = self.loss_fn( y_hat , y ).item()
                losses.append( loss )

        return losses

    def reset( self , *kwargs ):

        loss_type = kwargs.get( "loss_fn" , tnn.L1Loss )
        self.loss_fn = loss_type()

        self.model.load_state_dict(
            self.starting_params
        )

        lr = kwargs.get( "lr" , 1e-4 )
        opm_type = kwargs.get( "opm" , top.Adam )
        self.opm = opm_type( self.model.parameters() , lr = lr )

class train_app:

    def __init__( self , **kwargs ):

        data_sets = kwargs.get( "data_sets" , list( range( 100 ) ) )
        shuffle( data_sets )
        n = int( .8*len( data_sets ) )

        tr_batch_size = kwargs.get('tr_batch_size' , 500 )
        self.train_data = sampler(
            batch_size = tr_batch_size,
            sets = data_sets[:n]
        )

        ts_batch_size = kwargs.get('ts_batch_size' , 500 )
        self.test_data = sampler(
            batch_size = ts_batch_size,
            sets = data_sets[n:]
        )

        self.kernel = mod_kern( **kwargs )

        self.iter = 0
        self.record_interval = kwargs.get( "record_interval" , 1000 )
        self.test_rec = mob_mean_gen( 25 )
        self.train_rec = mob_mean_gen( 25 )

        self.min_loss = sys.maxsize
        self.i_min_loss = None

        self.buff = []
        self.buff_lim = kwargs.get( "buff_lim" , 100 )

        max_time = kwargs.get( "max_time" , 12 )
        time_type = kwargs.get( "time_type" , "hours" )
        self.ck = clock( max_time , time_type )

    def load_hist( self ):
        
        path = "DATA/performance.csv"
        df = pd.read_csv( path )
        return df.set_index( "iter_num" )

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
                self._save_model( max( rec[ 1 ] , rec[ 2 ] ) )
            
            #------------------------------------------
            # Doing one iteration of the backprop algorithm
            update()
            self.iter += 1

    def _print_rec( self , rec ):

        it = rec["iter_num"]
        print( f"iter #{it} " + "-"*25 )
        print( "time_passed: " + str( self.ck ) )
        print()

        print("losses:")
        tr_val = rec["tr_loss"]
        print(f" at training: {tr_val:.5f}")
        ts_val = rec["ts_loss"]
        print(f" at testing: {ts_val:.5f}")
        if self.iter:
            print(f" best: {self.min_loss:5f}")
        print()

        print( "target | " + rec["target"] )
        print( "made   | " + rec["made"] )
        print()         

    def _push_rec( self , rec ):

        iter_num = rec[ "iter_num" ]
        tr_val = rec[ "tr_val" ]
        ts_val = rec[ "ts_val" ]

        self.buff.append(
            ( iter_num, 
            ts_val,
            tr_val)
        )

    def _save_buff( self ):

        print()
        print( "saving buffer ....." , end = " ")

        path = "DATA/performance.csv"
        with open( path , "a" ) as f:
            if self.buff[0][0] == 0:
                f.write(
                    "iter_num,ts_loss,tr_loss" + "\n"
                )
            for rec in self.buff:
                iter_num,ts_loss,tr_loss = rec
                f.write( f"{iter_num},{ts_loss},{tr_loss}" + "\n")
    
    def _generate_record( self ):

        X_train , y_train = self.train_data.fetch_data()
        X_test , y_test   = self.test_data.fetch_data()
        tr_loss , ts_loss = self.kernel.evaluate( X_train , y_train , X_test , y_test )

        with tc.no_grad():

            X = X_train[ 0 ].to( device )
            y = y_train[ 0 ].to( device )
            y_hat = self.kernel.model( X )

            target = " ".join( f"{x:.5f}".ljust( 7 ) for x in y )
            made = " ".join( f"{x:.5f}".ljust( 7 ) for x in y_hat )

        return {
            "iter_num":self.iter,
            "tr_loss":tr_loss,
            "ts_loss":ts_loss,
            "target": target,
            "made"  : made
        }

    def _save_model( self , ts_val : float ):

        if ts_val >= self.min_loss:
            return
        
        print()
        print( "saving model ....." , end = " ")
        self.min_loss = ts_val
        self.i_min_loss = self.iter
        param = [ x for x in self.model.parameters()]
        tc.save(
            param,
            "DATA/model_params.pt"
        )
        print( "done!\n" )

    # @CK.tick()
    def _update_net( self ):

        X , y = self.train_data.fetch_data()
        self.kernel.update( X , y )

if __name__ == "__main__":

    from time import sleep

    t = train_app()
    @t.ck.tick()
    def update():
        t._update_net()

    cm = clock()
    @cm.tick()
    def boo( ):
        update()
        sleep( .01 )

    print( "tempo em ck | tempo em cm" )
    for _ in range( 10 ):
        boo()
        # t._update_net()
        s1 = f"{ t.ck.base:.5f}".rjust( 11 )
        s2 = f"{ cm.base:.5f}".rjust( 11 )
        print( s1 + " | " + s2 )
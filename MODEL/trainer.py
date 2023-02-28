try:
    from dsamp import stellarDset
    from network_model import stellar_model
    from clock import clock
    from mov_avg import mob_mean_gen

except ModuleNotFoundError:
    from MODEL.dsamp import stellarDset
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
    
    def evaluate( self, X , y ):

        with tc.no_grad():
            X = X.to( device )
            y = y.to( device )

            y_hat = self.model( X )
            loss = self.loss_fn( y_hat , y ).item()

        return loss

    def reset( self , **kwargs ):

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

        self._init_data( **kwargs )

        self._init_recorder( **kwargs )

        self._init_clock( **kwargs )

        self.pos_kernel = mod_kern( **kwargs )
        self.vel_kernel = mod_kern( **kwargs )

        self.iter = 0
        self.min_loss = sys.maxsize
        self.i_min_loss = None

    def _init_data( self , **kwargs ):

        #-------------------------------------------------------------------
        # Defining the simulations
        data_sets = kwargs.get( "data_sets" , list( range( 5*( 10**3 ) ) ) )
        shuffle( data_sets )
        n = int( .8*len( data_sets ) )
        
        #-------------------------------------------------------------------
        # Giving to training
        train_data = stellarDset( sets = data_sets[ :n ] )
        tr_batch_size = kwargs.get('tr_batch_size' , 500 )
        self.train_data = iter( tdt.DataLoader(
            train_data,
            tr_batch_size,
            True
        ) )

        #-------------------------------------------------------------------
        # Giving to testing
        test_data = stellarDset( sets = data_sets[ :n ] )
        ts_batch_size = kwargs.get('ts_batch_size' , 500 )
        self.test_data = iter( tdt.DataLoader(
            test_data,
            ts_batch_size,
            True
        ) )

    def _init_recorder( self , **kwargs ):

        #------------------------------------------------------------
        # If the evolution of the neural network must be saved 
        # to a csv file or printed
        self.to_save = kwargs.get( "to_save" , True )
        self.verbose = kwargs.get( "verbose" , True )

        #-----------------------------------------------------------
        # How much iterations between evaluations, evolution is printed
        # through a mobile mean of the costs
        self.record_interval = kwargs.get( "record_interval" , 1000 )
        mov_avg = kwargs.get( "mov_avg" , 25 )
        self.test_rec = mob_mean_gen( mov_avg )   # To smooth the training curve
        self.train_rec = mob_mean_gen( mov_avg )  # idem

        #-------------------------------------------------------
        # where the evolution will be saved
        self.buff = []
        self.buff_lim = kwargs.get( "buff_lim" , 100 )

    def _init_clock( self , **kwargs ):

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
        
        while True:

            A = self.ck.is_done()
            B = self.iter%self.record_interval == 0
            C = len( self.buff ) >= self.buff_lim

            #------------------------------------------------
            # If time is up or the buffer is full,
            # save its contents on a csv file
            if ( A or C ) and self.to_save:
                self._save_buff()
            
            if B:

                #---------------------------------------
                # Comparative performance of the network
                # on the training and testing sets
                rec = self._generate_record()

                #----------------------------------------
                # printing the result of the last function
                # on screen
                if self.verbose:
                    self._print_rec( rec )

                #-----------------------------------------
                # saving the last record on a buffer.
                if self.to_save: 
                    self._push_rec( rec )

                #------------------------------------------
                # saving the model, if the performance on the
                # test data set has improoved
                self._save_model( rec )
            
            if not A:
                #------------------------------------------
                # Doing one iteration of the backprop algorithm
                update()
                self.iter += 1
            else:
                break

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

        print( "position:")
        print( " target | " + rec["p_target"] )
        print( " made   | " + rec["p_made"] )
        print()

        print( "velocity:")
        print( " target | " + rec["v_target"] )
        print( " made   | " + rec["v_made"] )
        print()         

    def _push_rec( self , rec ):

        iter_num = rec[ "iter_num" ]
        tr_val = rec[ "tr_loss" ]
        ts_val = rec[ "ts_loss" ]

        self.buff.append(
            ( iter_num, 
            ts_val,
            tr_val)
        )

    def _save_buff( self ):

        if self.verbose:
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
        
        if self.verbose:
            print( "done!" )

        self.buff.clear()
    
    def _generate_record( self ):

        record = {}
        record[ "iter_num" ] = self.iter

        for dl in [ "tr" , "ts" ]:

            if dl == "tr":
                dset = self.train_data
            else:
                dset = self.test_data
            
            X , pos , vel = next( dset )
            p_loss = self.pos_kernel.evaluate( X , pos )
            v_loss = self.vel_kernel.evaluate( X , vel )
            record[ dl + "_loss" ] = .5*( p_loss + v_loss )

            if dl == "tr":
                with tc.no_grad():
                
                    X0 = X[ 0 ].to( device )

                    p0 = pos[ 0 ].to( device )
                    p_hat = self.pos_kernel.model( X0 )
                    record["p_target" ] = " ".join( f"{x:.5f}".ljust( 7 ) for x in p0 )
                    record["p_made" ] = " ".join( f"{x:.5f}".ljust( 7 ) for x in p_hat )

                    v0 = vel[ 0 ].to( device )
                    v_hat = self.vel_kernel.model( X0 )
                    record["v_target" ] = " ".join( f"{x:.5f}".ljust( 7 ) for x in v0 )
                    record["v_made" ] = " ".join( f"{x:.5f}".ljust( 7 ) for x in v_hat )

        return record

    def _save_model( self , rec ):

        val = max( rec["ts_loss"] , rec["tr_loss"] )
        if val >= self.min_loss:
            return
        
        if self.verbose:
            print()
            print( "saving model ....." , end = " ")

        self.min_loss = val
        self.i_min_loss = self.iter

        tups = [
            ( self.pos_kernel.model , "pos" ),
            ( self.vel_kernel.model , "vel" )
        ]
        for model , name in tups:
            param = [ x for x in model.parameters()]
            tc.save(
                param,
                f"DATA/{name}_model_params.pt"
            )

        if self.verbose:
            print( "done!\n" )

    # @CK.tick()
    def _update_net( self ):

        X , pos , vel = next( self.train_data )
        self.pos_kernel.update( X , pos )
        self.vel_kernel.update( X , vel )

if __name__ == "__main__":

    from time import sleep

    t = train_app(
        data_sets = list( range( 350 ) ),
        tr_batch_size = 500,
        ts_batch_size = 500,
        lr = 1e-4,
        record_interval = 25,
        # max_time = 1,
        # time_type = "hours",
        buff_lim = 25
    )

    t.run()
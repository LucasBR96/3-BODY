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
import matplotlib.pyplot as plt

from typing import *
from random import shuffle
import sys
import os
import pickle as pck

device = "cuda" if tc.cuda.is_available() else "cpu"

class mod_kern:

    losses = {
        "MAE" : tnn.L1Loss,
        "MSE" : tnn.MSELoss
    }

    opms = {
        "Adam" : top.Adam,
        "AdaGrad" : top.Adagrad,
        "SGD" : top.SGD
    }

    def __init__( self , **kwargs ):

        self.loss_type = kwargs.get( "loss_type" , "MAE" )
        loss_fun = mod_kern.losses.get( self.loss_type , tnn.L1Loss )
        self.loss_fn = loss_fun()

        self.model = stellar_model().to( device )
        # self.starting_params = self.model.state_dict()

        self.lr = kwargs.get( "lr" , 1e-4 )

        self.opm_type = kwargs.get( "opm" , "Adam" )
        opm = mod_kern.opms.get( self.opm_type , top.Adam )
        self.opm = opm( self.model.parameters() , lr = self.lr )
    
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

            y_hat = self.model( X ).detach()
            loss = self.loss_fn( y_hat , y ).item()

        return loss

    def load_from_file( self , path ):
        
        mod_state = tc.load( path )
        self.model.load_state_dict( mod_state )
        opm = mod_kern.opms.get( self.opm_type , top.Adam )
        self.opm = opm( self.model.parameters() , lr = self.lr )

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

    @staticmethod
    def load_hist( name ) -> pd.DataFrame:
        
        path = f"DATA/{name}/performance.csv"
        df = pd.read_csv( path )
        return df.set_index( "iter_num" )
    
    @staticmethod
    def plot_hist( name , log = True ):

        df : pd.DataFrame = train_app.load_hist( name )

        plt.plot(
            df.index,
            df[ "ts_loss" ],
            "--b",
            label = "ts_loss" 
        )

        plt.plot(
            df.index,
            df[ "tr_loss" ],
            "-k",
            label = "tr_loss" 
        )

        plt.legend()
        plt.title(
            f"Evolution of {name}"
        )
        plt.xlabel( "iteration" )
        plt.ylabel( "Error Value" )

        if log:
            plt.yscale( "log" )

        plt.show()
    
    @staticmethod
    def from_dict( app_d : Dict[ str , Any ] ):

        name = app_d["name"]
        app_d.pop( "name" )
        t = train_app( name , **app_d )

        t.iter = app_d["iter"]
        t.min_loss = app_d[ "min_loss"]
        t.i_min_loss = app_d[ "i_min_loss"]
        
        t.ck.base = app_d.get( "base_time" , 0 )
        
        pos_path = f"DATA/{t.name}/pos_model_params.pt"
        t.pos_kernel.load_from_file( pos_path )

        vel_path = f"DATA/{t.name}/vel_model_params.pt"
        t.vel_kernel.load_from_file( vel_path )

        return t

    @staticmethod
    def from_json( name ):

        path = f"DATA/{name}/meta.json"
        f = open( path , "rb" )
        app_d = pck.load( f )
        t = train_app.from_dict( app_d )
        f.close()

        return t
        
    def __init__( self , name : str , **kwargs ):

        self.name : str = name
        try:
            os.mkdir( f"DATA/{self.name}" )
        except FileExistsError:
            pass
        
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
        data_sets = kwargs.get( "sets" , None )
        if data_sets is None:
            data_sets = list( range( 500 ) )
        
        if kwargs.get( "to_shuffle" , True ):
            shuffle( data_sets )
        
        self.sets = data_sets
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
        self.tr_size = tr_batch_size

        #-------------------------------------------------------------------
        # Giving to testing
        test_data = stellarDset( sets = data_sets[ :n ] )
        ts_batch_size = kwargs.get('ts_batch_size' , 500 )
        self.test_data = iter( tdt.DataLoader(
            test_data,
            ts_batch_size,
            True
        ) )
        self.ts_size = ts_batch_size

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
        self.mov_avg = kwargs.get( "mov_avg" , 25 )
        self.test_rec = mob_mean_gen( self.mov_avg )   # To smooth the training curve
        self.train_rec = mob_mean_gen( self.mov_avg )  # idem

        #-------------------------------------------------------
        # where the evolution will be saved
        self.buff = []
        self.buff_lim = kwargs.get( "buff_lim" , 100 )

    def _init_clock( self , **kwargs ):

        self.max_time = kwargs.get( "max_time" , 12 )
        self.time_type = kwargs.get( "time_type" , "hours" )
        self.ck = clock( self.max_time , self.time_type )

    def to_dict( self ) -> Dict:
        
        return {
            "name" : self.name,
            "sets" : self.sets,
            "tr_batch_size" : self.tr_size,
            "ts_batch_size" : self.ts_size,
            "to_save" : self.to_save,
            "verbose" : self.verbose,
            "record_interval" : self.record_interval,
            "mov_avg" : self.mov_avg,
            "buff_lim" : self.buff_lim,
            "max_time" : self.max_time,
            "time_type" : self.time_type,
            "base_time" : self.ck.base,
            "lr" : self.pos_kernel.lr,
            "opm" : self.pos_kernel.opm_type,
            "loss_type" : self.pos_kernel.loss_type,
            "iter" : self.iter,
            "min_loss" : self.min_loss,
            "i_min_loss": self.i_min_loss,
            "to_shuffle": False
        }

    def to_json( self ):

        path = f"DATA/{self.name}/meta.json"
        f = open( path , "wb")
        d = self.to_dict()
        pck.dump( d , f )
        f.close()

    def __str__( self ):

        app_d = self.to_dict()
        lst = []
        for att , val in app_d.items():

            if att == "sets":
                att = "num_sets"
                val = len( val )

            lst.append(
                f"{att} = {val}"
            )
        return "\n".join( lst )
    
    def run( self ):
        
        @self.ck.tick()
        def update():
            self._update_net()
        
        def iter_step():

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

        self.to_json()
        while True:
            try:
                iter_step()
                if self.ck.is_done():
                    break
            except:
                break
        self.to_json()

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

        path = f"DATA/{self.name}/performance.csv"
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
        for model , tp in tups:
            param = model.state_dict()
            tc.save(
                param,
                f"DATA/{self.name}/{tp}_model_params.pt"
            )

        if self.verbose:
            print( "done!\n" )

    def _update_net( self ):

        X , pos , vel = next( self.train_data )
        self.pos_kernel.update( X , pos )
        self.vel_kernel.update( X , vel )

if __name__ == "__main__":

    from time import sleep

    # t = train_app(
    #     "A0",
    #     sets = list( range( 400 ) ),
    #     tr_batch_size = 200,
    #     ts_batch_size = 200,
    #     lr = .8*1e-4,
    #     record_interval = 150,
    #     max_time = 45,
    #     mov_avg = 200,
    #     time_type = "minutes",
    #     buff_lim = 25,
    #     loss_fn = "MSE"
    # )
    # print( t )
    # print()

    # t.run()
    # train_app.plot_hist( "A0" )
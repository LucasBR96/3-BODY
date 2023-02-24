import torch as tc
from torch.utils.data import Dataset , DataLoader
import pandas as pd
import numpy as np

from functools import lru_cache
from random import shuffle , choice
from typing import *

lru_cache( 10**5 )
def get_tup( simu , index ):

    data = pd.read_csv(f"DATA/simu/simulacao_{simu}")
    r1 = data.iloc[ index ].to_dict()
    r2 = data.iloc[ index + 1 ].to_dict()

    X = tc.zeros( 12 )
    X[ 0 ] = r1["x_0"]
    X[ 1 ] = r1["y_0"]
    X[ 2 ] = r1[ "vx_0" ]
    X[ 3 ] = r1[ "vy_0" ]
    X[ 4 ] = r1[ "x_1" ]
    X[ 5 ] = r1[ "y_1" ]
    X[ 6 ] = r1[ "vx_1" ]
    X[ 7 ] = r1[ "vy_1" ]
    X[ 8 ] = r1[ "x_2" ]
    X[ 9 ] = r1[ "y_2" ]
    X[ 10 ] = r1[ "vx_2" ]
    X[ 11 ] = r1[ "vy_2" ]
    
    pos = tc.zeros( 6 )
    pos[ 0 ] = r2[ "x_0" ]
    pos[ 1 ] = r2[ "y_0" ]
    pos[ 2 ] = r2[ "x_1" ]
    pos[ 3 ] = r2[ "y_1" ]
    pos[ 4 ] = r2[ "x_2" ]
    pos[ 5 ] = r2[ "y_2" ]

    vel = tc.zeros( 6 )
    vel[ 0 ] = r2[ "vx_0" ]
    vel[ 1 ] = r2[ "vy_0" ]
    vel[ 2 ] = r2[ "vx_1" ]
    vel[ 3 ] = r2[ "vy_1" ]
    vel[ 4 ] = r2[ "vx_2" ]
    vel[ 5 ] = r2[ "vy_2" ]

    return X , pos , vel

def get_dpoint( sizes , index  ):

    #-----------------------------------------------
    # Sizes = [ 3 , 5 , 9 ]
    #
    # index : 0 1 2 3 4 5 6 7 8
    # set   : 0 0 0 1 1 2 2 2 2
    # s_i   : 0 1 2 0 1 0 1 2 3

    s_arr = sizes[ sizes <= index ]
    simmu = sizes[ len( s_arr ) ]
    if simmu:
        index -= s_arr[ -1 ]
    return simmu , index

class stellarDset( Dataset ):

    def __init__( self , sets : List[ int ] ):

        # super( self ).__init__(  )
        self.sets = sets
        self.data_dict : Dict[ int , Tuple ] = {}

        try:
            meta : pd.DataFrame = pd.read_csv(
                f"DATA/simu/meta.csv"
            )
            meta.set_index( "simmu_id" , inplace = True )
        except FileNotFoundError:
            print( f"meta simulation file not found")
            return

        sizes : np.ndarray = ( meta.iloc[ self.sets ] )[ "size" ].to_numpy()
        self.sizes = ( sizes - 1 ).cumsum()

        self.n = sizes[ -1 ]
        
    def __len__( self ):
        return self.n
    
    def __getitem__(self, index ):
        
        simu , index = get_dpoint( self.sizes , index )
        simu = self.sets[ simu ]
        return get_tup( simu , index )

class sampler:

    def __init__( self , train = True , batch_size = 50 ):

        self.batch_size = batch_size
        nums = range( 80 , 100 , 1 )
        if train:
            nums = range( 80 )

        active_sets = {}
        for i in nums:
            D = DataLoader(
                stellarDset( set_num = i ),
                batch_size = batch_size,
                shuffle = True
            )
            active_sets[ i ] = iter( D )
        
        self.active_sets = active_sets
    
    def __iter__( self ):

        tup = self.fetch_data()
        if tup is None:
            raise StopIteration
        yield tup 

    def fetch_data( self ):
        
        while True:

            if not self.active_sets:
                return None
            
            i = choice( self.active_sets.keys() )
            try:
                D = self.active_sets[ i ]
                return next( D )

            except StopIteration:
                self.active_sets.pop( i )
             

if __name__ == "__main__":

    # get_i1( 6 , 7 )
    # get_i1( 6 , 9 )
    # get_i1( 6 , 1 )
    
    # D = DataLoader( stellarDset( 10 ), batch_size = 5 )
    # X , y = next( iter( D ) )

    s = sampler( batch_size = 5 )
    X , y = next( iter( s ) )

    print( *X , sep = "\n" )

    print()
    print( *y , sep = "\n" )

    


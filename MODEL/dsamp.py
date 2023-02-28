import torch as tc
from torch.utils.data import Dataset , DataLoader
import pandas as pd
import numpy as np

from functools import lru_cache
from random import shuffle , choice
from typing import *

lru_cache( 10**5 )
def get_tup( simu , index ):

    data = pd.read_csv(f"DATA/simu/simulacao_{simu}.csv")
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
    simmu = len( s_arr )
    if simmu:
        index -= s_arr[ -1 ]
    return simmu , index

class stellarDset( Dataset ):

    def __init__( self , sets : List[ int ] , repeats = 10**3 ):

        # super( self ).__init__(  )
        self.sets = sets
        self.repeats = max( repeats , 1 )

        try:
            meta : pd.DataFrame = pd.read_csv(
                f"DATA/meta.csv"
            )
            meta.set_index( "simu_id" , inplace = True )
        except FileNotFoundError:
            print( f"meta simulation file not found")
            return

        sizes : np.ndarray = ( meta.iloc[ self.sets ] )[ "size" ].to_numpy()
        self.sizes = ( sizes - 1 ).cumsum()

        self.n = self.sizes[ -1 ]
        
    def __len__( self ):
        return self.n*self.repeats
    
    def __getitem__(self, index ):
        
        index = index%self.n

        simu , index = get_dpoint( self.sizes , index )
        simu = self.sets[ simu ]
        return get_tup( simu , index )

if __name__ == "__main__":

    L = list( range( 5 ) )
    D = DataLoader( stellarDset( L ), batch_size = 10 , shuffle = True )
    X , pos , vel = next( iter( D ) )

    print( *X , sep = "\n" )
    print()

    print( *pos , sep = "\n" )
    print()

    print( *vel , sep = "\n" )

    


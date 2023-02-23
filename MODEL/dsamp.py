import torch as tc
from torch.utils.data import Dataset , DataLoader
import pandas as pd
import numpy as np

from functools import lru_cache
from random import shuffle ,choice

@lru_cache( 100 )
def get_i1( n : int , i : int ):

    top = n
    bottom = 0 
    while bottom < top - 1:
        
        # print( f"b , t = {bottom} , {top}" )

        mid = ( top + bottom )//2
        # print( f"m = {mid}")

        mid_i = mid*n - ( mid + 1 )*mid//2
        # print( f"mi = {mid_i}")

        if mid_i > i:
            top = mid
        else:
            bottom = mid
        
        # print( "-"*10 )

    # print( str( bottom ) + "\n" )
    return bottom

def get_i2( n , i , i1 ):
    
    bottom = i1*n - ( i1 + 1 )*i1//2
    off_set = i - bottom
    return off_set + ( i1 + 1 ) 


class stellarDset( Dataset ):

    def __init__( self , set_num = 0 ):

        # super( self ).__init__(  )

        self.set_num = set_num
        try:
            self.data = pd.read_csv(
                f"DATA/simu/simulacao_{set_num}.csv"
            )
        except FileNotFoundError:
            print( f"There is no such simulation with index {set_num}")
            return

        n = len( self.data )
        self.n = n - 1
        
    def __len__( self ):
        return self.n
    
    def __getitem__(self, index ):
        
        r1 = self.data.iloc[ index ].to_dict()
        r2 = self.data.iloc[ index ].to_dict()

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

        return X , y 


class sampler:

    def __init__( self , train = True , batch_size = 50 , sets = None ):

        self.batch_size = batch_size

        if ( sets is None ) and train:
            sets = range( 80 )
        elif ( sets is None ):
            sets = range( 80 , 100 , 1 )

        active_sets = {}
        for i in sets:
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
            
            i = choice( list( self.active_sets.keys() ) )
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

    s = sampler( batch_size = 5 , train = False )
    m = iter( s )
    X , y = next( m )
    X , y = next( m )

    print( *X , sep = "\n" )

    print()
    print( *y , sep = "\n" )

    


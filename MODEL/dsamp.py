import torch as tc
from torch.utils.data import Dataset , DataLoader
import pandas as pd
import numpy as np
import matplotlib.cm as cm 
import matplotlib.pyplot as plt

from functools import lru_cache
from random import shuffle , choice
from typing import *

@lru_cache( 10**5 )
def get_tup( simu , index ):

    data = pd.read_csv(f"DATA/simu/simulacao_{simu}.csv")
    r1 = data.iloc[ index ].to_dict()

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
    
    return X.reshape( 3 , 4 )

    # pos = tc.zeros( 6 )
    # pos[ 0 ] = r2[ "x_0" ]
    # pos[ 1 ] = r2[ "y_0" ]
    # pos[ 2 ] = r2[ "x_1" ]
    # pos[ 3 ] = r2[ "y_1" ]
    # pos[ 4 ] = r2[ "x_2" ]
    # pos[ 5 ] = r2[ "y_2" ]

    # vel = tc.zeros( 6 )
    # vel[ 0 ] = r2[ "vx_0" ]
    # vel[ 1 ] = r2[ "vy_0" ]
    # vel[ 2 ] = r2[ "vx_1" ]
    # vel[ 3 ] = r2[ "vy_1" ]
    # vel[ 4 ] = r2[ "vx_2" ]
    # vel[ 5 ] = r2[ "vy_2" ]

    # return X , pos , vel

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

def k_update_stats( mu , sigma , y , w , n , k):

    den = ( n + k )

    v = ( n*mu + k*y )/den

    omega_1 = n*( sigma**2 ) + n*( ( v - mu )**2 )
    omega_2 = k*( w**2 ) + k*( ( v - y )**2 )
    omega_sqr = ( omega_1 + omega_2 )/den
    omega = tc.sqrt( omega_sqr )
    
    return v , omega

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
        S = get_tup( simu , index )
        S_prime = get_tup( simu , index + 1 )

        return S , S_prime

class fetcher:

    def __init__(self , sets : List[ int ], batch_size , repeats = 10**3  ) -> None:
        
        self.batch_size = batch_size
        self.calls = 0

        self.mu : tc.Tensor = None
        self.sigma : tc.Tensor = None

        self.data_source = iter(
            DataLoader(
                stellarDset( sets , repeats = repeats ),
                batch_size = self.batch_size,
                shuffle = True
            )
        )
    
    def __call__(self) -> Tuple[ tc.Tensor , tc.Tensor ]:
        
        S , S_prime = next( self.data_source )
        self._update_norm( S )

        S = self.normalize( S )
        S_prime = self.normalize( S_prime )
        return S , S_prime
    
    def _update_norm( self , S : tc.Tensor ):
        
        y : tc.Tensor = S.mean( dim = 0 )
        w : tc.Tensor = S.std( dim = 0 )

        if self.calls:
            y , w = k_update_stats(
                self.mu,
                self.sigma,
                y , w,
                self.calls,
                self.batch_size
            ) 
        
        self.mu = y
        self.sigma = w
        self.calls += self.batch_size

    def normalize( self , S : tc.Tensor ):        
        return ( S - self.mu )/( self.sigma )

    def unormalize( self , S : tc.Tensor ):
        return S*self.sigma + self.mu


if __name__ == "__main__":

    L = list( range( 100 ) )
    # D = DataLoader( stellarDset( L ), shuffle = True )
    # X , _ = next( iter( D ) )
    
    # sizes = [ 25 , 50 , 100 , 200 , 400 , 800 ]
    sizes = [ 10 , 20 ]
    results = {}
    for sz in sizes:
        test_f = fetcher( L , sz )
        results[ sz ] = list()
        for i in range( 20 ):
            test_f()
            m = test_f.mu.mean()
            results[ sz ].append( m )
    
    cspace = np.linspace( 0 , 1 , len( sizes ) )
    colors = cm.get_cmap( 'cool' )( cspace )
    for cl , sz in zip( colors , sizes ):
        plt.plot(range(20) , results[ sz ] , color = cl , label = str( sz ) )
    plt.legend()
    plt.show()
    

    # print( *X , sep = "\n" )
    # print()

    # print( *pos , sep = "\n" )
    # print()

    # print( *vel , sep = "\n" )

    


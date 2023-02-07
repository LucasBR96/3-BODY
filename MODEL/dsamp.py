import torch as tc
from torch.utils.data import Dataset , DataLoader
import pandas as pd
import numpy as np

from functools import lru_cache
from random import shuffle

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
            data = pd.read_csv(
                f"DATA/simu/simulacao_{set_num}.csv"
            )
        except FileNotFoundError:
            print( f"There is no such simulation with index {set_num}")
            return
        
        dfs = []
        for i in range( 3 ):
            bod : pd.DataFrame = data.loc[ data[ "bod_id" ] == i ][ [ "iter_num" , "x" , "y" , "vx" , "vy" ] ]
            bod = bod.rename( columns = {
                "x":f"x_{i}",
                "y":f"y_{i}",
                "vx":f"vx_{i}",
                "vy":f"vy_{i}"
            })

            bod.set_index(
                np.arange( len( bod ) ),
                inplace = True 
            )
            
            dfs.append( bod )
        self.data = pd.concat( dfs , axis = 1 )

        n = len( self.data )
        self.n = ( n - 1 )*n//2
        
    def __len__( self ):
        return self.n
    
    def __getitem__(self, index ):
        
        n = len( self.data )

        i1 = get_i1( n , index )
        r1 = self.data.iloc[ i1 ].to_dict()

        i2 = get_i2( n , index , i1 )
        r2 = self.data.iloc[ i2 ].to_dict()

        X = tc.zeros( 13 )
        X[ 0 ] = r2[ "iter_num" ] - r1[ "iter_num" ]
        X[ 1 ] = r1["x_0"]
        X[ 2 ] = r1["y_0"]
        X[ 3 ] = r1[ "vx_0" ]
        X[ 4 ] = r1[ "vy_0" ]
        X[ 5 ] = r1[ "x_1" ]
        X[ 6 ] = r1[ "y_1" ]
        X[ 7 ] = r1[ "vx_1" ]
        X[ 8 ] = r1[ "vy_1" ]
        X[ 9 ] = r1[ "x_2" ]
        X[ 10 ] = r1[ "y_2" ]
        X[ 11 ] = r1[ "vx_2" ]
        X[ 12 ] = r1[ "vy_2" ]
        
        y = tc.zeros( 6 )
        y[ 0 ] = r2[ "x_0" ]
        y[ 1 ] = r2[ "y_0" ]
        y[ 2 ] = r2[ "x_1" ]
        y[ 3 ] = r2[ "y_1" ]
        y[ 4 ] = r2[ "x_2" ]
        y[ 5 ] = r2[ "y_2" ]


class sampler:

    def __init__( self , train = True , batch_size = 50 , num_sets = 10 ):

        self.batch_size = batch_size
        self.num_sets = num_sets

        nums = range( 80 , 100 , 1 )
        if train:
            nums = range( 80 )
        nums_lst = list( nums )
        shuffle( nums_lst )
        self.nums_lst = nums_lst

        active_sets = {}
        for i in range( num_sets ):
            x = nums_lst.pop()
            active_sets[ x ] = DataLoader(
                stellarDset( set_num = x ),
                batch_size = batch_size,
                shuffle = True
            )
        
        self.active_sets = active_sets
    
    def __iter__( self ):

        tup = self.fetch_data()
        if tup is None:
            raise StopIteration
        yield tup 

    def fetch_data( self ):
        pass
            
             

if __name__ == "__main__":

    # get_i1( 6 , 7 )
    # get_i1( 6 , 9 )
    # get_i1( 6 , 1 )

    D = DataLoader( stellarDset( 10 ), batch_size = 5 )
    X , y = next( iter( D ) )

    print( *X , sep = "\n" )

    print()
    print( *y , sep = "\n" )



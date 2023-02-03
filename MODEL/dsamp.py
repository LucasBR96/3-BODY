import torch as tc
from torch.utils.data import Dataset
import pandas as pd

from functools import lru_cache

@lru_cache( 100 )
def get_i1( n , i ):

    top = n
    bottom = 0 
    while bottom < top - 1:
        
        print( f"b , t = {bottom} , {top}" )

        mid = ( top + bottom )//2
        print( f"m = {mid}")

        mid_i = mid*n - ( mid + 1 )*mid//2
        print( f"mi = {mid_i}")

        if mid_i > i:
            top = mid
        else:
            bottom = mid
        
        print( "-"*10 )

    print( str( bottom ) + "\n" )
    return bottom

def get_i2( n , i , i1 ):
    pass


class stellarDset( Dataset ):

    def __init__( self , set_num = 0 ):

        super( ).__init__( self )

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
            bod = data.loc[ data[ "bod_id" ] == i ]
            bod.rename( columns = {
                "x":f"x_{i}",
                "y":f"y_{i}",
                "vx":f"vx_{i}",
                "vy":f"vy_{i}"
            })
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

if __name__ == "__main__":

    get_i1( 6 , 7 )
    get_i1( 6 , 9 )
    get_i1( 6 , 1 )
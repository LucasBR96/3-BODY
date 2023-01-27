import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

import math

def sample_simulation( simu_id ):

    df = pd.read_csv( f"DATA/simu/simulacao_{simu_id}.csv")
    # df.columns[].astype

    dfs = []
    for i in range( 3 ):
        pos = df.loc[ df[ "bod_id" ] == i ][ ["iter_num" , "x", "y"] ]
        pos.rename( columns = {"x":f"x_{i}" , "y":f"y_{i}" } , inplace = True )
        pos.set_index( "iter_num" , inplace = True )
        dfs.append( pos )
    
    new_d = pd.concat( dfs , axis = 1  )
    return new_d

def plot_simulation( simu_id , num_points = None ):

    simu_df = sample_simulation( simu_id )

    if num_points is None:
        num_points = len( simu_df )
    num_ticks = int( math.sqrt( num_points ) )

    colors = [ "red" , "green" , "blue" ]
    for i in range( 3 ):

        x = simu_df[ f"x_{i}" ].to_numpy()[ :num_points ]
        y = simu_df[ f"y_{i}" ].to_numpy()[ :num_points ]
        plt.plot( x , y ,color = colors[ i ]  )

        x_ticks = x[ ::num_ticks ]
        y_ticks = y[ ::num_ticks ]
        plt.scatter( x_ticks , y_ticks , color = colors[ i ] , marker = "*" )

    plt.title( f"Simulação #{simu_id}" )
    plt.show()

plot_simulation( 0 )
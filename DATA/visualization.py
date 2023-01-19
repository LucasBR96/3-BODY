import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

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
    print( new_d )
    return new_d

def plot_simulation( simu_df : pd.DataFrame , simu_id ):

    colors = [ "red" , "green" , "blue" ]
    for i in range( 3 ):
        x = simu_df[ f"x_{i}" ].to_numpy()
        y = simu_df[ f"y_{i}" ].to_numpy()
        plt.plot( x , y ,color = colors[ i ]  )
    plt.title( f"Simulação #{simu_id}" )
    plt.show()

df = sample_simulation( 0 )
plot_simulation( df , 0 )
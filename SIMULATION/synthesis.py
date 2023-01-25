# python standard library ----------------------------------------------
from collections import namedtuple
import random as rd
import itertools as it
import time as tm

# third party modules    ----------------------------------------------
import numpy as np

# internal modules ----------------------------------------------------
from simmu import simmu

# CONSTANTS ----------------------------------------------------------
SIMU_HEAD = "iter_num,bod_id,x,y,vx,vy\n"
META_HEAD = "bod_id,mu_x,std_x,mu_y,std_y,mu_vx,std_vx,mu_vy,std_vy\n"

current_simu = 0

def print_iter( tup ):

    iter_num , pos , vel = tup
    print( f"iteration #{iter_num}" + "-"*25 + "\n" )
    for i in range( 3 ):
       print( f"body{i}:")

       spos = f"x , y = {pos[i,0]:.2f} , {pos[i,1]:.2f}"
       print( "\t" + spos )

       svel = f"vx , vy = {vel[i,0]:.2f} , {vel[i,1]:.2f}"
       print( "\t" + svel + "\n" ) 
def record_iteration( simulation_id, tup ):

    fl = open( f"DATA/simu/simulacao_{simulation_id}.csv" , mode = "a" )
    iter_num , pos , vel = tup
    for i in range( 3 ):
        s = f"{iter_num},{i},{pos[i,0]},{pos[i,1]},{vel[i,0]},{vel[i,1]}\n"
        fl.write(s)
    fl.close()

    # print( s )

def simulate( simulation_id , verbose = True ):
    
    fl = open( f"DATA/simu/simulacao_{simulation_id}.csv" , mode = "a" )
    fl.write( SIMU_HEAD )
    fl.close()

    Sm = simmu()
    meta_tup = None 
    while Sm.curr_time < Sm.run_time:

        tup = Sm()
        record_iteration( simulation_id , tup )
        if verbose:
            print_iter( tup )
    #     update_meta( simulation_id , tup , meta_tup )
    
    # record_meta( simulation_id )

simulate( 0 )
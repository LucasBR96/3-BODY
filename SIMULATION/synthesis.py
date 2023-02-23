# python standard library ----------------------------------------------
from collections import namedtuple
import random as rd
import itertools as it
import time as tm

# third party modules    ----------------------------------------------
import numpy as np

# internal modules ----------------------------------------------------
try:
    from simmu import simmu
except ModuleNotFoundError:
    from SIMULATION.simmu import simmu

# CONSTANTS ----------------------------------------------------------
SIMU_HEAD = "iter_num,x_0,y_0,vx_0,vy_0,x_1,y_1,vx_1,vy_1,x_2,y_2,vx_2,vy_2\n"
META_HEAD = "bod_id,mu_x,std_x,mu_y,std_y,mu_vx,std_vx,mu_vy,std_vy\n"

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

    iter_num , pos , vel = tup
    s = f"{iter_num}"
    for i in range( 3 ):
        s += f",{pos[i,0]},{pos[i,1]},{vel[i,0]},{vel[i,1]}"
        if i == 2:
            s += "\n"
    
    fl = open( f"DATA/simu/simulacao_{simulation_id}.csv" , mode = "a" )
    fl.write(s)
    fl.close()

    # print( s )

def simulate( simulation_id , verbose = False , Sm = None , step_size = None ):
    
    fl = open( f"DATA/simu/simulacao_{simulation_id}.csv" , mode = "a" )
    fl.write( SIMU_HEAD )
    fl.close()

    if Sm is None:
        Sm = simmu()

    base_step = 0.
    if step_size is None:
        step_size = 0
    
    while Sm.curr_time < Sm.run_time:

        a = bool( step_size )
        b = ( Sm.curr_time >= step_size + base_step )
        c = bool( Sm.curr_time )
        
        tup = Sm()

        if not( a ) or ( a and ( b or not c ) ):
            record_iteration( simulation_id , tup )
            if verbose:
                print_iter( tup )
        
        if a and b:
            base_step += step_size


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
SIMU_HEAD = "iter_num,bod_id,x,y,vx,vy\n"
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

    fl = open( f"DATA/simu/simulacao_{simulation_id}.csv" , mode = "a" )
    iter_num , pos , vel = tup
    for i in range( 3 ):
        s = f"{iter_num:.5f},{i},{pos[i,0]},{pos[i,1]},{vel[i,0]},{vel[i,1]}\n"
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

        tup = Sm()
        
        if not( step_size ) or ( Sm.curr_time >= step_size + base_step ):
            record_iteration( simulation_id , tup )
            base_step += step_size

        if verbose:
            print_iter( tup )

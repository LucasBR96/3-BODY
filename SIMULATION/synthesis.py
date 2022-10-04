# python standard library ----------------------------------------------
from collections import namedtuple
import random as rd
import itertools as it
import time as tm

# third party modules    ----------------------------------------------
import numpy as np

# CONSTANTS ----------------------------------------------------------
RADIUS = 2
ITER_STEP = 0.025
RECORD_STEP = 0.1
NUM_ITERS = 2048
MIN_DISTANCE = 0.1
HEAD = "bod_id,iter_num,x,y,vx,vy\n"

body = namedtuple(
        "body", ["x", "y", "vx","vy"]
    )

current_simu = 0
simu_times   = {}
def simu_time_update( step_func ):

    def step_with_update( bods ):
        t = tm.time() 
        step_func( bods )
        dt = tm.time() - t
        simu_times[ current_simu ] = simu_times.get( current_simu , 0 ) + dt

    return step_with_update

def init_planets( rnd_pos = True , rnd_speed = True ):
    
    angles = [ -np.pi/6 , np.pi/2 , ( 7/6 )*np.pi ] 
    if rnd_pos:
        theta_1 = rd.random()*2*np.pi
        theta_2 = theta_1 + (2/3)*np.pi
        theta_3 = theta_1 + np.pi*( 1 + (2/3)*rd.random() )
        angles = [ theta_1 , theta_2 , theta_3 ]
    
    velocities = [ ( 0 , 0 ) , ( 0 , 0 ) , ( 0 , 0 ) ]
    if rnd_speed:
        direcs = [ 2*np.pi*rd.random() for _ in range( 3 ) ]
        velocities = [ ( np.cos( x ) , np.sin( x ) ) for x in direcs ]

    bodies = []
    for theta , vel in zip( angles , velocities ):
        x , y = RADIUS*np.cos( theta ) , RADIUS*np.sin( theta )
        vx , vy = vel
        bod = body( x , y , vx , vy )
        bodies.append( body )
    return bodies

def update_position( bod , acc ):

    ax , ay = acc
    x , y , vx , vy = bod

    bod.vx = vx + ax*ITER_STEP
    bod.vy = vy + ay*ITER_STEP
    bod.x  = x + vx*ITER_STEP + 0.5*ax*( ITER_STEP**2 )
    bod.y  = y + vy*ITER_STEP + 0.5*ay*( ITER_STEP**2 )

def distance_squared( body_1 , body_2 ):
    return ( body_1.x - body_2.x )**2 + ( body_1.y - body_2.y )**2

def get_acc( body_1 , body_2 ):
    
    d_sqrd = distance_squered( body_1 , body_2 )
    if d_sqrd <= MIN_DISTANCE**2:
        return 0
    return 1/d_sqrd

def get_acc_vector( body_1 , body_2 ):

    acc_mod = get_acc( body_1 , body_2 )
    if acc_mod == 0:
        return ( 0 , 0 )

    distance = np.sqrt( distance_squared( body_1 , body_2 ) )
    dx = body_2.x - body_1.x
    dy = body_2.y - body_1.y

    ax = acc_mod*( dx/distance )
    ay = acc_mod*( dy/distance )
    return ( ax , ay )

@simu_time_update
def iter_step( bodies ):
    
    acc = np.zeros( ( len( bodies ) , 2 ) )
    n = len( bodies )
    for i , j in it.combinations( range( n ) , 2 ):
        acc_vec = np.array( get_acc_vector( bodies[ i ] , bodies[ j ] ) )
        acc[ i ] += acc_vec
        acc[ j ] -= acc_vec
    
    for i , bod in enumerate( bodies ):
        update_position( bod , acc[ i ] )

def record_bod( bod, bod_id, simulation_id, iter_num ):

    fl = open( f"DATA/simu/simulacao_{simulation_id}" , mode = "a" )
    s = f"{bod_id},{iter_num},{bod.x},{bod.y},{bod.vx},{bod.vy}\n"
    fl.write( s )
    fl.close()

    print( s )

def simulate( simulation_id ):
    
    fl = open( f"DATA/simu/simulacao_{simulation_id}" , mode = "a" )
    fl.write( HEAD )
    fl.close()

    bods = init_planets( rnd_speed = False )
    for m in range( NUM_ITERS ):
        for z in range( RECORD_STEP//ITER_STEP ):
            iter_step( bods )

        for i , bod in enumerate( bods ):
            record_bod( bod , i, simulation_id, m ) 

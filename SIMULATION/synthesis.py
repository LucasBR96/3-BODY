# python standard library ----------------------------------------------
from collections import namedtuple
import random as rd
import itertools as it

# third party modules    ----------------------------------------------
import numpy as np

# CONSTANTS ----------------------------------------------------------
RADIUS = 2
ITER_STEP = 0.025
RECORD_STEP = 0.1
NUM_ITERS = 2048
MIN_DISTANCE = 0.1

body = namedtuple(
        "body", ["x", "y", "vx","vy"]
    )

def init_planets():

    theta_1 = rd.random()*2*np.pi
    theta_2 = theta_1 + np.pi/3
    theta_3 = theta_1 + np.pi*( 1 + rd.random()/3 )
    angles = [ theta_1 , theta_2 , theta_3 ]

    bodies = []
    for theta in angles:
        x , y = RADIUS*np.cos( theta ) , RADIUS*np.sin( theta )
        vel_theta = rd.random()*2*np.pi
        vx , vy = np.cos( vel_theta ) , np.sin( vel_theta )
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

def iter_step( bodies ):
    
    acc = np.zeros( ( len( bodies ) , 2 ) )
    n = len( bodies )
    for i , j in it.combinations( range( n ) , 2 ):
        acc_vec = np.array( get_acc_vector( bodies[ i ] , bodies[ j ] ) )
        acc[ i ] += acc_vec
        acc[ j ] -= acc_vec
    
    for i , bod in enumerate( bodies ):
        update_position( bod , acc[ i ] )



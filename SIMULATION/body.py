import numpy as np

class body:

    def __init__( self , pos , speed ):
        self.pos = pos
        self.speed = speed

    def move( self , acc , time ):

        pos , speed = self.pos , self.speed
        self.speed = speed + acc*time
        self.pos   = pos + speed*time + acc*( time**2 )/2
    
    def __str__( self ):
        
        x , y = self.pos
        vx , vy = self.speed
        return f"{x},{y},{vx},{vy}"
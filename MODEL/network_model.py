import torch as tc
import torch.utils as ut
import torch.nn as tnn

NUM_LAY = 10
LAY_SIZE = 256

class perceptron( tnn.Module ):

    def __init__( self , in_size = LAY_SIZE , out_size = LAY_SIZE ):

        super().__init__()
        self.lin = tnn.Linear( in_size , out_size )
        self.rel = tnn.ReLU( )
    
    def forward( self , X ):

        X = self.lin( X )
        return self.rel( X )
    

class stellar_model( tnn.Module ):

    def __init__( self ):
        super().__init__()

        self.head = perceptron( 13 )
        self.body = tnn.Sequential(
            *[ perceptron() for _ in range( NUM_LAY ) ]
        )
        self.tail = tnn.Linear( LAY_SIZE , 6 )
    
    def forward( self , X ):
        y = self.head( X )
        y = self.body( y )
        return self.tail( y )
    

if __name__ == "__main__":

    device = "cuda" if tc.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    X = tc.rand( 5 , 10 ).to( device )
    print( f"Before P: \n \n {X}")

    P = perceptron( 10 , 5 ).to( device )
    X = P( X )

    print( f"\nAfter P: \n \n {X}")
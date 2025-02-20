import scipy.io
import numpy as np
import jax.numpy as jnp

def get_dataset(filename):
    data = scipy.io.loadmat('data/'+filename)
    t, x, y, Vs, Ws = data["t"], data["x"], data["y"], data["V"], data["W"]
    t = t.astype(jnp.float32)
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    
    # X, Y, T = np.meshgrid(x,y,t, indexing='ij')
    X, Y = np.meshgrid(x, y, indexing='ij')
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    # T = T.reshape(-1, 1)  
    # coords = np.hstack((X, Y, T))

    for a,t_ in enumerate(t):
                h = np.ones((len(X)))*t_
                if a>0: coords = np.vstack((coords,np.hstack((X,Y,h.reshape(-1,1)))))
                else: coords = np.hstack((X,Y,h.reshape(-1,1)))
    V = Vs.reshape(-1,1)
    W = Ws.reshape(-1,1)

    return jnp.array(coords), t, X, Y, V, W
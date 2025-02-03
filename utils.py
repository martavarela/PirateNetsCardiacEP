import scipy.io
import numpy as np
import jax.numpy as jnp

def get_dataset(filename):
    data = scipy.io.loadmat('data/'+filename)
    t, x, y, z, Vs, Ws = data["t"], data["x"], data["y"], \
                                data["z"], data["V"], data["W"]
    
    for a,t_ in enumerate(t):
                h = np.ones((len(x)))*t_
                if a>0: coords = np.vstack((coords,np.hstack((x,y,z,h.reshape(-1,1)))))
                else: coords = np.hstack((x,y,z,h.reshape(-1,1)))
    V = Vs.reshape(-1,1)
    W = Ws.reshape(-1,1)

    return jnp.array(coords), t, x, y, z, V, W

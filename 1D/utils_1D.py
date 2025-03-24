import scipy.io
import numpy as np
import jax.numpy as jnp

def get_dataset(filename):
    data = scipy.io.loadmat('data/'+filename)
    t, x, V, W = data["t"], data["x"], data["V"], data["W"]
    t = t.astype(jnp.float32).reshape(-1,1)
    x = x.astype(jnp.float32).reshape(-1,1)
    
    T, X = np.meshgrid(t,x,indexing='ij')
    T = T.reshape(-1, 1)   
    X = X.reshape(-1, 1)
  
    coords = np.hstack((T, X))

    # scale time to [0,1]
    # t = (t-t[0])/(t[-1]-t[0])
    
    # X, Y = np.meshgrid(x, y)
    # for a,t_ in enumerate(t):
    #     h = np.ones((len(X)))*t_
    #     if a>0: coords = np.vstack((coords,np.hstack((h.reshape(-1,1),X,Y))))
    #     else: coords = np.hstack((h.reshape(-1,1),X,Y))

    return jnp.array(coords), t, x, V, W
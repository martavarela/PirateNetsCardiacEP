from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt

class AlievPanfilov2D(ForwardIVP):
    def __init__(self, config, V0, V, t_star, x_star, y_star, coords):
        super().__init__(config)

        self.config.weighting.use_causal = False
         ## PDE Parameters
        self.a = 0.01
        self.b = 0.15
        self.D = 0.1
        self.k = 8
        self.mu_1 = 0.2
        self.mu_2 = 0.3
        self.epsilon = 0.002

        self.t0 = t_star[0]
        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star
        self.V0 = V0
        self.V_ref = V
        self.coords = coords

        # Predictions over a grid
        self.V_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
        self.W_pred_fn = vmap(vmap(self.W_net, (None, None, 0, 0)), (None, 0, None, None))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0, 0)), (None, 0, None, None))

    def neural_net(self, params, t, x, y):
        txy = jnp.stack([t, x, y])
        print(f"Shape of x before FourierEmbs: {txy.shape}")
        _, outputs = self.state.apply_fn(params, txy)
        V = outputs[0]
        W = outputs[1]
        return V, W
    
    def u_net(self, params, t, x, y):
        V, _ = self.neural_net(params, t, x, y)
        return V

    def W_net(self, params, t, x, y):
        _, W = self.neural_net(params, t, x, y)
        return W

    def r_net(self, params, t, x, y):
        # code from vanilla PINNs to be adapted to PirateNets:
        # dv_dt = dde.grad.jacobian(y, x, i=0, j=3)
        # dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        # dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        # dv_dzz = dde.grad.hessian(y, x, component=0, i=2, j=2)
        # dw_dt = dde.grad.jacobian(y, x, i=1, j=3)
        ## Coupled PDE+ODE Equations
        # eq_a = dv_dt -  self.D*(dv_dxx + dv_dyy + dv_dzz) + self.k*V*(V-self.a)*(V-1) +W*V 
        # eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))

        V, W = self.neural_net(params, t, x, y)
        V_t = grad(self.u_net, argnums=1)(params, t, x, y) # derivative of V with respect to t (2nd argument)
        W_t = grad(self.W_net, argnums=1)(params, t, x, y)
        V_hessian = hessian(self.u_net, argnums=(1, 2, 3))(params, t, x, y)

        V_xx = V_hessian[1][1]
        V_yy = V_hessian[2][2]

        eq_V = V_t - self.D*(V_xx + V_yy) + self.k*V*(V-self.a)*(V-1) + W*V
        eq_W = W_t - (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))

        return eq_V, eq_W

    def rV_net(self, params, t, x, y):
        rV, _ = self.r_net(params, t, x, y)
        return rV
    
    def rW_net(self, params, t, x, y):
        _, rW = self.r_net(params, t, x, y)
        return rW

    @partial(jit, static_argnums=(0,)) # modifies following function by partially applying the jit function to it with a static 1st argument
    def res_and_w(self, params, batch):
        "Compute residuals and weights for causal training"
        # Sort time coordinates
        t_sorted = batch[:, 0].sort()
        rV_pred, rW_pred = \
            vmap(self.r_pred_fn, (None, 0, 0))(params, t_sorted, batch[:, 1], batch[:, 2])
        # Split residuals into chunks
        rV_pred = rV_pred.reshape(self.num_chunks, -1)
        lV = jnp.mean(rV_pred**2, axis=1)
        rW_pred = rW_pred.reshape(self.num_chunks, -1)
        lW = jnp.mean(rW_pred**2, axis=1)
        l = lV + lW
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):

        data_batch = batch["data"]
        batch = batch["res"]

        data_coords_batch, V_train = data_batch
        # Initial condition loss
        V_pred_ic = vmap(self.u_net, (None, None, 0, 0))(params, self.t0, self.x_star, self.y_star)
        ics_loss = jnp.mean((self.V0 - V_pred_ic) ** 2)
        print('The shape of V_pred_ic is:', V_pred_ic.shape)
    
        # No-flux boundary condition loss
        # du_dx_bound = grad(self.V_pred_fn, argnums=1)(params, t, x, y, z)
        # bc_loss = jnp.mean( (du_dx_bound - 0) ** 2 )

        V_pred_data = self.V_pred_fn\
            (params, data_coords_batch[:, 0], data_coords_batch[:, 1],\
              data_coords_batch[:, 2])
        print('data batch shape:', data_coords_batch.shape)
        print('data batch[:, 0] shape:', data_coords_batch[:, 0].shape)
        V_loss = jnp.linalg.norm(V_train - V_pred_data) # change way of calculating loss, should be the same anyway

        # Residual loss
        if self.config.weighting.use_causal == True:
            l, w = self.res_and_w(params, batch)
            res_loss = jnp.mean(l * w)
        else:
            rV_pred, rW_pred = \
            self.r_pred_fn(params, batch[:, 0], batch[:, 1], batch[:, 2])
            res_loss = jnp.mean((rV_pred) ** 2 + (rW_pred) ** 2)

        loss_dict = {"ics": ics_loss, "res": res_loss, "data": V_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        ics_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.u_net, params, self.t0, self.x_star
        )

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            # sort the time step for causal loss
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )
            res_ntk = res_ntk.reshape(self.num_chunks, -1)  # shape: (num_chunks, -1)
            res_ntk = jnp.mean(
                res_ntk, axis=1
            )  # average convergence rate over each chunk
            _, casual_weights = self.res_and_w(params, batch)
            res_ntk = res_ntk * casual_weights  # multiply by causal weights
        else:
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {"ics": ics_ntk, "res": res_ntk}

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, V_test):
        V_pred = self.V_pred_fn(params, self.t_star, self.x_star, self.y_star)
        V_test = jnp.reshape(V_test, V_pred.shape)
        error = jnp.linalg.norm(V_pred - V_test) / jnp.linalg.norm(V_test)
        return error
    

class Evaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.V_pred_fn(params, self.model.t_star, self.model.x_star, self.model.y_star, self.model.z_star)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, cmap="jet")
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
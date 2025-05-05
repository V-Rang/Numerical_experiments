'''
Febuary 5, 2025

Driver script that will simulate a disruption
and return the corresponding loss function. 
Loss function can be found in the documentation.

Created by: Jonathan Arnaud
'''

import main                         # main function that does disruption simulation
from scipy.integrate import simpson # routine to integrate RE current
import numpy as np
# from jax.config import config
# config.update("jax_disable_jit", True)
from config import config           # Loading config file to get parameters required to run simulation
import jax
import jax.numpy as jnp
import pickle
import equinox as eqx
import time
import Impurities as Imp
import Routines as RT
import Constants as Const
import REsolver as RE
import CurrentDecaySolver as CQ
import PowerBalanceSolver as PB
import diffrax as dfx
from diffrax import SaveAt
from jax import lax
from functools import partial
from jaxopt import Bisection
from jax.experimental import host_callback as hcb

jax.config.update("jax_enable_x64", True)

# Function to simulate disruption
'''
inputs ->  injection scheme
t_inject: array of timestamps for each injection
nD_inject: array of amount of deuterium to inject at each timestamp
nHe_inject: array of amount of helium to inject at each timestamp
nNe_inject: array of amount of neon to inject at each timestamp
nAr_inject: array of amount of argon to inject at each timestamp

outputs -> time integrated RE current [MA]
TimeIntegrated_I_RE: scalar quantity of time integrated RE current
'''


# @eqx.filter_jit
def DoDisruption(inp_arr, solver_1_times, solver_2_times):

    MyImp = Imp.Impurities()  # class containing impurity data
    # deuteriumRadInterp = MyImp.deuteriumRadInterp
    # deuteriumZbarInterp = MyImp.deuteriumZbarInterp
    # heliumRadInterp = MyImp.heliumRadInterp
    # heliumZbarInterp = MyImp.heliumZbarInterp
    # neonRadInterp = MyImp.neonRadInterp
    # neonZbarInterp = MyImp.neonZbarInterp
    # argonRadInterp = MyImp.argonRadInterp
    # argonZbarInterp = MyImp.argonZbarInterp

    deuteriumRadInterp = MyImp.Rad_deuterium
    deuteriumZbarInterp = MyImp.Zbar_deuterium
    heliumRadInterp = MyImp.Rad_helium
    heliumZbarInterp = MyImp.Zbar_helium
    neonRadInterp = MyImp.Rad_neon
    neonZbarInterp = MyImp.Zbar_neon
    argonRadInterp = MyImp.Rad_argon
    argonZbarInterp = MyImp.Zbar_argon

    n_e_gridvals = MyImp.n_e_grid
    T_e_gridvals = MyImp.Tgrid_deuterium

    t_inject = inp_arr[:2]   # timestamp for each injection
    nD_inject = inp_arr[2:4] # deuterium quantity to inject
    nHe_inject = inp_arr[4:6] # helium quantity to inject
    nNe_inject = inp_arr[6:8] # neon quantity to inject
    nAr_inject = inp_arr[8:] # argon quantity to inject

    dt_inject = 1e-4*np.ones(len(config.t_inject))

    N_inject = len(config.t_inject) # Number of injections

    # beginning of Solver = main.GATOR(config)

    # MyRoutines = RT.Routines()
    # MyCQ = CQ.CurrentDecay()
    # MyRE = RE.RunawayElectron()
    # MyPB = PB.PowerBalance()

    MyConst = Const.Constants()

    nD = 1e20
    nHe = 0
    nNe = 0
    nAr = 0

    B = 5.3    
    
    # N_inject = len(config.t_inject) # Number of injections
    # dt_inject = 1e-4*np.ones(len(config.t_inject))

    a = 2          # radius               [m  ]
    A = np.pi*a**2 # Cross sectional area [m^2]
     
    # n_e = 0.
    # T_e = 0.
    # p = 0.
    # I_RE = 0.
    # I_p = 0.

    BtildeOverB = 0.04
    S_ext = 2e6 # [W/m^3]

    I_p_thresh = 1e6      # CQ ends if plasma current goes below 1MA
    t_thresh = 1       # CQ ends if it lasts longer than one second

    # end of Solver = main.GATOR(config)
    
    # needed by ODEsystem:

    # beginning of Solver.solve()
    T_e = 15e3 # Initial plasma temperature [eV]
    # self = MyRoutines.Get_zbar_and_nefree(self) # update n_e

    # beginning of Get_zbar_and_nfree
    nD_zbar_func = nD/1e6
    # nD = compute_nD(plasma.nD)
    # print(type(nD))
    nHe = nHe/1e6
    nNe = nNe/1e6
    nAr = nAr/1e6

    '''
    Initialize convergence
    metric quantities
    '''
    error = np.ones(1)
    count = 0

    # Initialize charge states
    ZD_prev  = 1
    ZHe_prev = 1
    ZNe_prev = 3
    ZAr_prev = 5

    '''
    Begin loop and iterate between
    evaluating charge state and
    free electron density until
    converged
    '''
    # while jnp.all(error > 1.e-14) and count < 10000:
    # for i in range(1):

    # Evaluate free electron density
    n_ePrev = nD_zbar_func*ZD_prev + nHe*ZHe_prev + nNe*ZNe_prev + nAr*ZAr_prev

    # query = jnp.array([T_e, n_ePrev])
    # query_2d = query[None, :]  # shape (1,2)
    # Compute charge states
    # ZD  = deuteriumZbarInterp(query_2d)[0]
    # ZHe = heliumZbarInterp(query_2d)[0]
    # ZNe = neonZbarInterp(query_2d)[0]
    # ZAr = argonZbarInterp(query_2d)[0]

    # n_e_gridvals = MyImp.n_e_grid
    # T_e_gridvals = MyImp.Tgrid_deuterium

    # ZD = 1.
    # ZHe = 2.
    # ZNe = 9.99882
    # ZAr = 17.9471

    T_index = jnp.argmin(jnp.abs(T_e - T_e_gridvals))
    n_index = jnp.argmin(jnp.abs(n_ePrev - n_e_gridvals))
    
    ZD = deuteriumZbarInterp[T_index, n_index]
    ZHe = heliumZbarInterp[T_index, n_index]
    ZNe = neonZbarInterp[T_index, n_index]
    ZAr = argonZbarInterp[T_index, n_index]

    # print(ZD,":",ZHe, ":", ZNe, ":", ZAr) # 1.0 : 2.0 : 9.99882 : 17.9471
    # Update free electron density
    # and iteration metrics

    # print(nD_zbar_func, ":", ZD)
    # print(nHe, ":", ZHe)
    # print( nNe, ":", ZNe)
    # print( nAr, ":", ZAr )

    n_eNew =  nD_zbar_func*ZD + nHe*ZHe + nNe*ZNe + nAr*ZAr
    error = abs(n_eNew - n_ePrev) / n_ePrev
    count += 1

    # Update charge states
    ZD_prev  = ZD
    ZHe_prev = ZHe
    ZNe_prev = ZNe
    ZAr_prev = ZAr

    assert count < 10000, "Free electron update while impurities were injected did not converge!"
    

    # def iterative_update_fixed(error, count, num_iters=5):
    #     def body_fun(i, state):
    #         err, cnt = state    
    #         n_ePrev = nD_zbar_func*ZD_prev + nHe*ZHe_prev + nNe*ZNe_prev + nAr*ZAr_prev

    #         query = jnp.array([T_e, n_ePrev])
    #         query_2d = query[None, :]  # shape (1,2)

    #         # Compute charge states
    #         ZD  = deuteriumZbarInterp(query_2d)[0]
    #         ZHe = heliumZbarInterp(query_2d)[0]
    #         ZNe = neonZbarInterp(query_2d)[0]
    #         ZAr = argonZbarInterp(query_2d)[0]

    #         # ZD = 1.
    #         # ZHe = 2.
    #         # ZNe = 9.99882
    #         # ZAr = 17.9471

    #         # print(ZD,":",ZHe, ":", ZNe, ":", ZAr) # 1.0 : 2.0 : 9.99882 : 17.9471
    #         # Update free electron density
    #         # and iteration metrics
    #         n_eNew =  nD_zbar_func*ZD + nHe*ZHe + nNe*ZNe + nAr*ZAr
    #         error = abs(n_eNew - n_ePrev) / n_ePrev
    #         count += 1

    #         # Update charge states
    #         ZD_prev  = ZD
    #         ZHe_prev = ZHe
    #         ZNe_prev = ZNe
    #         ZAr_prev = ZAr

    #         return error, count

    #     final_state = lax.fori_loop(0, num_iters, body_fun, (error, count))
    #     return final_state

    # final_error, final_count = iterative_update_fixed(error, count, num_iters=1000)

    # after converge update parameters
    # in plasma object
    ZD = ZD
    ZHe = ZHe
    ZNe = ZNe
    ZAr = ZAr
    n_e = n_eNew*1e6 # converting back to [m^-3] 
    
    # jax.debug.print('ZD = {}, ZHe = {}, ZNe = {}, ZAr = {}', ZD, ZHe, ZNe, ZAr)

    nD_background = config.nD_background

    # def outer_fun(x, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp,
    #             neonZbarInterp, argonZbarInterp, nD, nHe, nNe, nAr, p, e_const, n_tot_ion):

    #     def inner_fun(y, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp, neonZbarInterp, argonZbarInterp,
    #                 nD, nHe, nNe, nAr):
            
    #         T_index = jnp.argmin(jnp.abs(x - T_e_gridvals))
    #         n_index = jnp.argmin(jnp.abs( (y - n_e_gridvals)/1e6 ))
            
    #         ZD = deuteriumZbarInterp[T_index, n_index]
    #         ZHe = heliumZbarInterp[T_index, n_index]
    #         ZNe = neonZbarInterp[T_index, n_index]
    #         ZAr = argonZbarInterp[T_index, n_index]

    #         return y - (nD*ZD + nHe*ZHe + nNe*ZNe + nAr*ZAr)

    #     inner_solver = Bisection(
    #         optimality_fun=inner_fun,
    #         lower=1e18,
    #         upper=1e23,     
    #         tol=1e-6,
    #         maxiter=50,
    #         check_bracket = False,
    #         implicit_diff_solve = None
    #         # jit = False
    #         )
        
    #     sol_inner = inner_solver.run(1e20, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp, neonZbarInterp, argonZbarInterp,
    #                 nD, nHe, nNe, nAr).params
    
    #     jax.debug.print("y = {}, x = {}", sol_inner, x)

    #     T_e_old = p/e_const/(sol_inner + n_tot_ion)
    #     func = x - (sol_inner + nD)/(sol_inner + n_tot_ion)*T_e_old

    #     return func


    # def pure_bisection(f, a, b, tol=1e-10, maxiter=20):
    #     """
    #     Find root of f(x)=0 in [a,b] by bisection.
    #     f  : callable x -> f(x)
    #     a  : lower bound
    #     b  : upper bound
    #     """
    #     # initial f(a), f(b)
    #     fa = f(a)
    #     fb = f(b)
    #     # state = (a, b, fa, fb, iteration)
    #     def cond_fn(state):
    #         a, b, fa, fb, i = state
    #         # continue while i<maxiter and interval width > tol
    #         return (i < maxiter) & ((b - a) > tol)

    #     def body_fn(state):
    #         a, b, fa, fb, i = state
    #         m = 0.5 * (a + b)
    #         fm = f(m)
    #         # if f(a)*f(m)>0 then root is in [m,b], else in [a,m]
    #         left_has_root = fa * fm > 0
    #         # update bounds
    #         a_new = jnp.where(left_has_root, m, a)
    #         fa_new = jnp.where(left_has_root, fm, fa)
    #         b_new = jnp.where(left_has_root, b, m)
    #         fb_new = jnp.where(left_has_root, fb, fm)
    #         return (a_new, b_new, fa_new, fb_new, i + 1)

    #     a_final, b_final, _, _, _ = lax.while_loop(
    #         cond_fn, body_fn, (a, b, fa, fb, 0)
    #     )

    #     return 0.5 * (a_final + b_final)

    def pure_bisection(f, a, b, tol=1e-15, maxiter=100):
        """
        Find a root of f(x)=0 on [a,b] with Brent's method (like scipy.brentq).
        Assumes f(a) and f(b) have opposite signs.
        """
        a = jnp.asarray(a, dtype=jnp.float64)
        b = jnp.asarray(b, dtype=jnp.float64)
        
        fa = f(a)
        fb = f(b)
        # c is the previous best bracket point
        c = a
        fc = fa
        # d and e record the size of the last and second‐last steps
        d = b - a
        e = d

        def cond_fn(state):
            a, b, c, fa, fb, fc, d, e, i = state
            return (i < maxiter) & (jnp.abs(b - a) > tol)

        def body_fn(state):
            a, b, c, fa, fb, fc, d, e, i = state

            # Ensure |f(a)| < |f(b)|
            swap = jnp.abs(fa) < jnp.abs(fb)
            
            a, b = jnp.where(swap, b, a), jnp.where(swap, a, b)
            fa, fb = jnp.where(swap, fb, fa), jnp.where(swap, fa, fb)

            # Now a is the “worse” end, b is the best
            c = c
            fc = fc

            # Decide whether to do interpolation (secant / inverse‐quadratic) or bisection
            # Inverse‐quadratic if fa, fb, fc all distinct, else secant
            use_iquad = (fa != fc) & (fb != fc)
            s_iq = ((a * fb * fc) / ((fa - fb) * (fa - fc))
                    + (b * fa * fc) / ((fb - fa) * (fb - fc))
                    + (c * fa * fb) / ((fc - fa) * (fc - fb)))
            s_sec = b - fb * (b - a) / (fb - fa)
            s = jnp.where(use_iquad, s_iq, s_sec)

            # Check whether the interpolation is acceptable; otherwise fall back to bisection
            cond1 = (s < (3*a + b)/4) | (s > b)
            cond2 = (e < tol) | (jnp.abs(s - b) >= jnp.abs(b - c)/2)
            do_bisect = cond1 | cond2
            s = jnp.where(do_bisect, 0.5 * (a + b), s)

            # Update bookkeeping for the next iteration
            e = d
            d = b - s

            fs = f(s)

            # Shift the triple (a,b,c) so that b is always the best bracket
            c, fc = a, fa
            cond_fb_fs = fb * fs < 0
            a = jnp.where(cond_fb_fs, b, s)
            fa = jnp.where(cond_fb_fs, fb, fs)
            b = jnp.where(cond_fb_fs, s, b)
            fb = jnp.where(cond_fb_fs, fs, fb)

            return (a, b, c, fa, fb, fc, d, e, i + 1)

        # Initial state (iteration counter i=0)
        state0 = (a, b, c, fa, fb, fc, d, e, 0)
        a_final, b_final, *_ = lax.while_loop(cond_fn, body_fn, state0)
        return 0.5 * (a_final + b_final)


    # def pure_bisection(f, a, b, tol=1e-15, maxiter=100):
    #     # 1) make sure floats
    #     a = jnp.array(a, dtype=jnp.float64)
    #     b = jnp.array(b, dtype=jnp.float64)
    #     fa = f(a); fb = f(b)
    #     # (optionally assert fa*fb<0)

    #     c = a; fc = fa
    #     d = b - a; e = d

    #     def cond_fn(state):
    #         a, b, c, fa, fb, fc, d, e, i = state
    #         return (i < maxiter) & (jnp.abs(b - a) > tol)

    #     def body_fn(state):
    #         a, b, c, fa, fb, fc, d, e, i = state

    #         # 2) reorder so |fb| < |fa|
    #         swap = jnp.abs(fa) < jnp.abs(fb)
    #         a, b = jnp.where(swap, b, a), jnp.where(swap, a, b)
    #         fa, fb = jnp.where(swap, fb, fa), jnp.where(swap, fa, fb)
    #         c, fc = jnp.where(swap, a, c), jnp.where(swap, fa, fc)

    #         # 3) attempt IQI vs. secant
    #         use_iqi = (fa != fc) & (fb != fc)
    #         s_iqi = (
    #             (a * fb * fc) / ((fa - fb) * (fa - fc))
    #         + (b * fa * fc) / ((fb - fa) * (fb - fc))
    #         + (c * fa * fb) / ((fc - fa) * (fc - fb))
    #         )
    #         s_sec = b - fb * (b - a) / (fb - fa)
    #         s = jnp.where(use_iqi, s_iqi, s_sec)

    #         # 4) fallback to bisection if out of range / not reducing fast enough
    #         cond1 = (s < (3*a + b)/4) | (s > b)
    #         cond2 = (e < tol) | (jnp.abs(s - b) >= jnp.abs(b - c)/2)
    #         do_bisect = cond1 | cond2
    #         s = jnp.where(do_bisect, 0.5 * (a + b), s)

    #         fs = f(s)
    #         d, e = b - a, d  # shift history

    #         # 5) update bracket [a,b]
    #         left = fb * fs < 0
    #         a = jnp.where(left, b, a)
    #         fa = jnp.where(left, fb, fa)
    #         b = jnp.where(left, s, b)
    #         fb = jnp.where(left, fs, fb)

    #         return (a, b, c, fa, fb, fc, d, e, i + 1)

    #     init_state = (a, b, c, fa, fb, fc, d, e, 0)
    #     a_final, b_final, *_ = lax.while_loop(cond_fn, body_fn, init_state)
    #     return 0.5 * (a_final + b_final)


    def inner_solver_custom_bisection(y, x, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp, neonZbarInterp, argonZbarInterp,
        nD, nHe, nNe, nAr):
            
        T_index = jnp.argmin(jnp.abs(x - T_e_gridvals))
        n_index = jnp.argmin(jnp.abs( (y - n_e_gridvals)/1e6 ))
        
        ZD = deuteriumZbarInterp[T_index, n_index]
        ZHe = heliumZbarInterp[T_index, n_index]
        ZNe = neonZbarInterp[T_index, n_index]
        ZAr = argonZbarInterp[T_index, n_index]

        # jax.debug.print("{} : {} : {} : {}", ZD, ZHe, ZNe, ZAr)


        return y - (nD*ZD + nHe*ZHe + nNe*ZNe + nAr*ZAr)

    def outer_solver_custom_bisection(x, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp,
        neonZbarInterp, argonZbarInterp, nD, nHe, nNe, nAr, p, e_const, n_tot_ion):
    
        sol_inner = pure_bisection(lambda y: inner_solver_custom_bisection(y, x, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp, neonZbarInterp, argonZbarInterp,
        nD, nHe, nNe, nAr), 1e18, 1e23)

        # jax.debug.print("y = {}, x = {}", sol_inner, x)

        T_e_old = p/e_const/(sol_inner + n_tot_ion)
        return x - (sol_inner + nD)/(sol_inner + n_tot_ion)*T_e_old
    

    # time_vals = []
    
    # def record_time(t, _):
    #     # `t` here is a concrete DeviceArray
    #     time_vals.append(float(t))
        
    def ODEsystem(t, y, args):
        # print(t)

        # time_vals.append(t)
        # t = hcb.id_tap(record_time, t)
        
        p, I_RE, I_p = y[0], y[1], y[2]

        # p = p
        # I_RE = I_RE
        # I_p = I_p
        # t = t  

        # self = self.MyRoutines.ComputeParameters(self)
        # self = self.MyPB.GetPowerBalance(self)
        # self = self.MyRE.GetFluidRE(self)
        # self = self.MyCQ.GetCurrentDecay(self)
        # out = jnp.array([self.dp_Over_dt, self.dI_RE_Over_dt, self.dI_p_Over_dt])

        # self = self.MyRoutines.ComputeParameters(self)
        I_Omega = I_p - I_RE # Ohmic current [A]
        j_RE    = I_RE      / A # RE     current density [A/m^2]
        j_Omega = I_Omega   / A # Ohmic  current density [A/m^2]
        j_p     = I_p       / A # Plasma current density [A/m^2]

        n_RE = j_RE/MyConst.e/MyConst.c # RE number density [m^-3]

        Heaviside_inject  = 0.5 * (1 + jnp.tanh( (t - t_inject)/dt_inject ) ) 

        # jax.debug.print("{}", t)

        # jax.debug.print("{}", t_inject)

        # jax.debug.print("{}", Heaviside_inject)

        # print(type(plasma.nD), ":", plasma.nD)
        # print(type(plasma.t_inject), ":", type(plasma.nD), ":", type(plasma.nHe), ":", type(plasma.nNe), ":", type(plasma.nAr))

        nD  = nD_background + jnp.sum(Heaviside_inject * nD_inject)
        nHe = jnp.sum(Heaviside_inject * nHe_inject)
        nNe = jnp.sum(Heaviside_inject * nNe_inject)
        nAr = jnp.sum(Heaviside_inject * nAr_inject)

        n_tot_ion = nD + nHe + nNe + nAr

        # jax.debug.print("{} : {} : {} : {} : {}", nD, nHe, nNe, nAr, n_tot_ion)


        # ***********************************************************
        # custom bisection:
        e_const = MyConst.e
    
        # def outer_solver_custom_bisection(x, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp,
        #     neonZbarInterp, argonZbarInterp, nD, nHe, nNe, nAr, p, e_const, n_tot_ion):

        # T_e = outer_solver_custom_bisection.run(10e3, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp,
        #     neonZbarInterp, argonZbarInterp, nD, nHe, nNe, nAr, p, e_const, n_tot_ion).params

        T_e = pure_bisection(lambda x: outer_solver_custom_bisection(x, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp,
            neonZbarInterp, argonZbarInterp, nD, nHe, nNe, nAr, p, e_const, n_tot_ion), 1, 15e3)
        # jax.debug.print("_____________________")
        n_e = nD*ZD + nHe*ZHe + nNe*ZNe + nAr*ZAr

        jax.debug.print("{} : {}", T_e, n_e)
        # jax.debug.print("{}", nNe)
        # jax.debug.print("{} : {} : {} : {} : {} : {} : {} : {} : {}", nD, nHe, nNe, nAr, p, e_const, n_tot_ion,T_e,n_e)

        # ***********************************************************

        # ***********************************************************
        # outer_solver = Bisection(
        #     optimality_fun = outer_fun,
        #     lower=1.,
        #     upper=15e3,
        #     tol=1e-12,
        #     maxiter=50,
        #     check_bracket = False,
        #     implicit_diff_solve = None
        # )

        # # e_const = MyConst.e

        # # T_e = outer_solver.run(10e3, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp,
        # #             neonZbarInterp, argonZbarInterp, nD, nHe, nNe, nAr, p, e_const, n_tot_ion).params

        # # jax.debug.print("T_e = {}", T_e)

        # def solve_Te(T_e_init, *args):
        #     return outer_solver.run(T_e_init, *args).params

        # e_const = MyConst.e

        # T_e = solve_Te(10e3, T_e_gridvals, n_e_gridvals, deuteriumZbarInterp, heliumZbarInterp,
        #             neonZbarInterp, argonZbarInterp, nD, nHe, nNe, nAr, p, e_const, n_tot_ion)

        # ***********************************************************

        # ***********************************************************
        # # T_e_init_guess = (1 + 15e3)/2
        # n_e_init_guess = 1e20
        # T_e_init_guess = 15e3

        # num_outer_loops = 50
        # num_inner_loops = 10    

        # for i in range(num_outer_loops):
        #     T_e = T_e_init_guess
        #     for j in range(num_inner_loops):
                
        #         # ZD = 1.
        #         # ZHe = 2.
        #         # ZNe = 9.9996
        #         # ZAr = 17.9841

        #         # query = jnp.array([T_e, n_e_init_guess/1e6])
        #         # query_2d = query[None, :]  # shape (1,2)
        #         # ZD  = deuteriumZbarInterp(query_2d)[0]
        #         # ZHe = heliumZbarInterp(query_2d)[0]
        #         # ZNe = neonZbarInterp(query_2d)[0]
        #         # ZAr = argonZbarInterp(query_2d)[0]
        
        #         T_index = jnp.argmin(jnp.abs(T_e - T_e_gridvals))
        #         n_index = jnp.argmin(jnp.abs( (n_e_init_guess - n_e_gridvals)/1e6 ))
                
        #         ZD = deuteriumZbarInterp[T_index, n_index]
        #         ZHe = heliumZbarInterp[T_index, n_index]
        #         ZNe = neonZbarInterp[T_index, n_index]
        #         ZAr = argonZbarInterp[T_index, n_index]

        #         n_e = (nD*ZD + nHe*ZHe + nNe*ZNe + nAr*ZAr)
            
        #         # jax.debug.print("T_e = {}, n_e = {}", T_e, n_e)

        #         n_e_init_guess = n_e

        #     T_e_old = p/MyConst.e/(n_e + n_tot_ion)
        #     T_e_init_guess = (n_e + nD)/(n_e + n_tot_ion)*T_e_old
            
        #     # jax.debug.print("T_e_comp = {}", T_e_init_guess)
        
        # T_e = T_e_init_guess
        # n_e = n_e_init_guess

        # jax.debug.print("T_e = {}, n_e = {}", T_e, n_e)
        # ***********************************************************


        # ***********************************************************
        # @partial(jax.jit, static_argnums=(2,3))
        # def solve_fixed_point(T_init, n_init, num_outer, num_inner):    
        #     def outer_body(carry, _):
        #         T_e, n_e = carry

        #         def inner_body(i, n_e):
        #             # return 0.1 * T_e + 0.9 * n_e        
        #             T_index = jnp.argmin(jnp.abs(T_e - T_e_gridvals))
        #             n_index = jnp.argmin(jnp.abs( (n_e - n_e_gridvals)/1e6 ))
                    
        #             ZD = deuteriumZbarInterp[T_index, n_index]
        #             ZHe = heliumZbarInterp[T_index, n_index]
        #             ZNe = neonZbarInterp[T_index, n_index]
        #             ZAr = argonZbarInterp[T_index, n_index]

        #             return nD*ZD + nHe*ZHe + nNe*ZNe + nAr*ZAr

        #         n_e = jax.lax.fori_loop(0, num_inner, inner_body, n_e)

        #         # T_e = 0.2 * n_e + 0.8 * T_e
        #         T_e_old = p/MyConst.e/(n_e + n_tot_ion)
        #         T_e = (n_e + nD)/(n_e + n_tot_ion)*T_e_old

        #         return (T_e, n_e), None

        #     (T_final, n_final), _ = jax.lax.scan(
        #         outer_body,
        #         (T_init, n_init),
        #         None,
        #         length=num_outer
        #     )

        #     return T_final, n_final

        # # Usage:
        # T0 = 15e3      
        # n0 = 1e20    
        # T_e, n_e = solve_fixed_point(T0, n0, num_outer= 20, num_inner= 10)
        # ***********************************************************

        # jax.debug.print("T_e = {}, n_e = {}", T_e, n_e)

        # return jnp.array([T_e, T_e, T_e])
        # jax.debug.print("ZD = {}, ZHe = {}, ZNe = {}, ZAr = {}", ZD, ZHe, ZNe, ZAr)

        Zeff = (nD*ZD + nHe*ZHe**2 + nNe*ZNe**2 + nAr*ZAr**2)/n_e # effective charge
        # print(type(plasma.Zeff))
        # if plasma.Zeff < 1:
        #     plasma.Zeff = 1
        Zeff = jnp.maximum(Zeff, 1.)

        Coulog0 = 14.9 - 0.5 * jnp.log(n_e/1e20) + jnp.log(T_e/1e3) # Coulog logarithm [Wesson tokamaks]
        tau_c = 4 * MyConst.pi * MyConst.ep0**2 * MyConst.m_e**2 * MyConst.c**3 / (MyConst.e**4 * n_e * Coulog0) # relativistic collision time [s]
        
        # jax.debug.print("pi = {}, ep0 = {}, m_e = {}, c = {}, e = {}, n_e = {}, Coulog0 = {}", MyConst.pi, MyConst.ep0, MyConst.m_e, MyConst.c, MyConst.e, n_e, Coulog0)

        eta = 1.65e-9 * Zeff * Coulog0 / (T_e/1e3)**1.5 # Spitzer resistivity including Zeff [Wesson eq 14.10.1]
        E_Vert = eta * j_Omega # parallel electric field
        E_c = MyConst.m_e * MyConst.c / (MyConst.e * tau_c) # Connor-Hastie electric field
        
        # jax.debug.print("tau_c = {}", tau_c)

        E_Vert_Over_E_c = E_Vert / E_c # electric field normalized to Connor-Hastie
        chi = 1.08704e12 * BtildeOverB**2 * MyConst.e**2 * T_e**2.5 * Zeff/(MyConst.m_e * Coulog0 * (nD*ZD**2 + nHe*ZHe**2 + nNe*ZNe**2 + nAr*ZAr**2)) # heat diffusivity [m^2/s]
        tau_E = a**2/chi # confinement time [s]
        tau_s = 6*MyConst.pi*MyConst.ep0*MyConst.m_e**3*MyConst.c**3/(MyConst.e**4*B**2)
        alpha = tau_c/tau_s

        # self = self.MyPB.GetPowerBalance(self)
        S_kappa = n_e*T_e/tau_E
        S_Ohmic = eta*j_Omega**2

        # jax.debug.print("n_e = {}, nD = {}, nHe = {}, nNe = {}, nAr = {}", n_e, nD, nHe, nNe, nAr)
        # query = jnp.array([T_e, n_e/1e6])
        # query_2d = query[None, :]  # shape (1,2)
        # S_rad = n_e/1e6*( nD*2e-30 + nHe*0. + nNe*0. + nAr*0. ) # raditiave power loss [W/m^3]
        # S_rad = n_e/1e6*( nD*deuteriumRadInterp(query_2d)[0] + nHe*heliumRadInterp(query_2d)[0] + nNe*neonRadInterp(query_2d)[0] + nAr* argonRadInterp(query_2d)[0] ) # raditiave power loss [W/m^3]

        T_index = jnp.argmin(jnp.abs(T_e - T_e_gridvals))
        n_index = jnp.argmin(jnp.abs( (n_e - n_e_gridvals)/1e6 ))
        
        # ZD = deuteriumRadInterp[T_index, n_index]
        # ZHe = heliumRadInterp[T_index, n_index]
        # ZNe = neonRadInterp[T_index, n_index]
        # ZAr = argonRadInterp[T_index, n_index]

        S_rad = n_e/1e6*( nD*deuteriumRadInterp[T_index, n_index] + nHe*heliumRadInterp[T_index, n_index] + nNe*neonRadInterp[T_index, n_index] + nAr* argonRadInterp[T_index, n_index] ) # raditiave power loss [W/m^3]

        # result = deuteriumRadInterp(query_2d)[0]  # returns an array of shape (1,)

        # jax.debug.print("result = {}", result)
        # jax.debug.print("S_rad = {}", S_rad)
        
        S_RE = E_c * j_RE

        dp_Over_dt = 2/3 * (
                            S_Ohmic +
                            S_ext   +
                            S_RE    -
                            S_rad   - 
                            S_kappa                                      
                            )


        # jax.debug.print("{} : {} : {} : {} : {} : {}",n_e, nD, nHe, nNe, nAr, S_rad)

        # jax.debug.print("{} : {} : {}", eta, j_Omega, S_Ohmic)

        # jax.debug.print("E_c = {}, j_RE = {}, S_RE = {}", E_c, j_RE, S_RE)

        # jax.debug.print("{} : {} : {} : {} : {} : {}", S_Ohmic, S_ext, S_RE, S_rad, S_kappa, dp_Over_dt)

        # self = self.MyRE.GetFluidRE(self)
        gamma_0 = 1/tau_c/Coulog0 * jnp.sqrt(MyConst.pi/(3 * (Zeff + 5) ) )
        gamma_avRP = gamma_0 * (E_Vert_Over_E_c - 1)
        gamma_av = gamma_avRP

        dI_RE_Over_dt = I_RE * gamma_av

        # self = self.MyCQ.GetCurrentDecay(self)
        dI_p_Over_dt = -2*MyConst.pi*E_Vert/MyConst.mu0

        # jax.debug.print("{} : {} : {}", dp_Over_dt, dI_RE_Over_dt, dI_p_Over_dt)

        out = jnp.array([dp_Over_dt, dI_RE_Over_dt, dI_p_Over_dt])

        # jax.debug.print("dp/dt = {}, dIre/dt = {}, dIp/dt = {}", dp_Over_dt, dI_RE_Over_dt, dI_p_Over_dt)
        # jax.debug.print("dp/dt = {}", dp_Over_dt)
        # out = jnp.array([-p, -I_RE, -I_p])

        return out


    # nD = 1e20
    
    # end of Routines.Get_zbar_and_nfree

    # back to Solver.solve()

    # print(T_e, MyConst.e, n_e, nD)
    # jax.debug.print("Te = {}, e = {}, n_e = {}, nD = {}", T_e, MyConst.e, n_e, nD)
    # print(nD) #1e20

    p = T_e * MyConst.e * (n_e + nD) # computing initial pressure
    # jax.debug.print("ZD = {}, ZHe = {}, ZNe = {}, ZAr = {}, n_e = {}, p = {}", ZD, ZHe, ZNe, ZAr, n_e, p) # same as scipy code.

    I_p = 15e6 # Initial plasma current     [A ]
    I_RE = 1                         # runaway electron current

    # print(p, ":", I_RE, ":", I_p) # 480660.00000000006 : 1 : 15000000.0
    
    init_condition = jnp.array([p, I_RE, I_p])

    # jax.debug.print("init = {}", init_condition)
    

    t_start, t_end = 0.0, t_inject[1]
    root_finder = dfx.VeryChord(rtol=1e-6, atol=1e-6, )
    # solver = dfx.Kvaerno5(root_finder = root_finder)
    solver = dfx.Tsit5()
    # solver = dfx.Radau(rtol=1e-6, atol=1e-6)

    # adjoint = dfx.ImplicitAdjoint()   # or BacksolveAdjoint(), etc.
    adjoint = dfx.BacksolveAdjoint()

    # stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-3)
    # stepsize_controller = dfx.PIDController(rtol=1e-6, atol = 1e-2)
    stepsize_controller = dfx.StepTo(ts = solver_1_times)

    t_a = time.time()

    sol = dfx.diffeqsolve(
        dfx.ODETerm(ODEsystem),
        solver=solver,
        t0=solver_1_times[0],
        t1=solver_1_times[-1],
        dt0 = None,
        y0=init_condition,
        stepsize_controller=stepsize_controller,
        # saveat = dfx.SaveAt(t1 = True),
        saveat = dfx.SaveAt(ts=solver_1_times),
        max_steps=1000000,
        throw = True,
        adjoint = adjoint
    )

    t_b = time.time()

    TimeForSimluation = t_b - t_a

    jax.debug.print("sol = {}, time = {}", sol.ys[-1], TimeForSimluation)

    # ts_host = np.array(sol.ts)
    
    sol_1 = np.array(sol.ys)

    jnp.save('jax_solver1_sol.npy', sol_1)

    return sol.ys[-1][0]

    # jax.debug.print("sol = {}", sol.ys[-1])

#   #***************************************************
    # init_condition_2 = jnp.array([
    #     sol.ys[-1][0],  
    #     sol.ys[-1][1],  
    #     sol.ys[-1][2]
    # ])

    # t_start, t_end = t_inject[1], t_thresh

    # sol1 = dfx.diffeqsolve(
    #     dfx.ODETerm(ODEsystem), 
    #     solver=solver,
    #     t0=t_start,     
    #     t1=t_end,
    #     dt0 = None,
    #     # dt0 = 0.0007071068057186922,
    #     y0=init_condition_2,
    #     # args = n_e,
    #     stepsize_controller=stepsize_controller,
    #     saveat=dfx.SaveAt(ts=solver_2_times), 
    #     max_steps=1000000,
    #     throw = True
    # )

    # # print(sol.ys.shape, ":", sol.ts.shape) #(337,3), (337, )

    # jax.debug.print("sol1 = {}", sol1.ys[-1])
    # t2 = time.time()

    # TimeForSimluation = t2-t1
    # sol_t    = jnp.concatenate([sol.ts,   sol1.ts   ])
    # sol_p    = jnp.concatenate([sol.ys[:,0], sol1.ys[:,0]])
    # sol_I_RE = jnp.concatenate([sol.ys[:,1], sol1.ys[:,1]])
    # sol_I_p  = jnp.concatenate([sol.ys[:,2], sol1.ys[:,2]])

    # Delta_t_CQ_max = 150.0e-3
    # Delta_t_CQ_min = 50.0e-3

    # t    = sol_t    # Time [s]
    # I_RE = sol_I_RE # RE current [A]
    # I_p  = sol_I_p  # Plasma current [A]
    # I_p_Norm = ( I_p - jnp.min(I_p) ) / (jnp.max(I_p) - jnp.min(I_p) )

    # # t_I_p_80 = jnp.where(I_p_Norm == 0.8)[0]
    # # if len(t_I_p_80) < 1:
    # #     Delta_t_CQ = Delta_t_CQ_max
    # #     w_CQ = I_p[-1]/1e6

    # Delta_t_CQ = Delta_t_CQ_max
    # w_CQ = I_p[-1]/1e6

    # Heaviside_CQ_min = jnp.heaviside((Delta_t_CQ_min-Delta_t_CQ),0.5)
    # Heaviside_CQ_max = jnp.heaviside((Delta_t_CQ-Delta_t_CQ_max),0.5)
    
    # L_CQ = Heaviside_CQ_min + Heaviside_CQ_max
    
    # L_I_RE = jnp.max(I_RE/1e6) # Maximum RE current in MA
    # w_I_RE = 1
    
    # Loss = w_CQ*L_CQ + w_I_RE*L_I_RE

    # # jax.debug.print("loss = {}", Loss)
    # return Loss


def disruption_scalar(inp_arr, solver1_times, solver2_times):
    return DoDisruption(inp_arr, solver1_times, solver2_times)

grad_fn = jax.grad(disruption_scalar, argnums=0)

disruption_jit     = jax.jit(disruption_scalar)
grad_disruption_jit = jax.jit(grad_fn)


# @eqx.filter_jit  # instead of using jax.jit here
def main():
    with open('with_injection_master_results_dpdt', 'rb') as file:
        scipy_results = pickle.load(file)

    solver_1_results = scipy_results['solver1_results']
    solver_2_results = scipy_results['solver2_results']

    solver_1_times = solver_1_results['solver1_time']
    solver_2_times = solver_2_results['solver2_time']

    t_inject = jnp.array([0, 0.04])
    nD_inject  = jnp.array([0, 0])
    nHe_inject = jnp.array([0, 0])
    # nNe_inject = jnp.array([0, 0])
    nNe_inject = jnp.array([1e19, 5e20])
    nAr_inject = jnp.array([0, 0])

    inp_arr = jnp.concat((t_inject, nD_inject, nHe_inject, nNe_inject, nAr_inject))
    
    output = None
    grads = None

    out = DoDisruption(inp_arr, solver_1_times, solver_2_times)

    # output = disruption_jit(inp_arr, solver_1_times, solver_2_times)
    
    # grads  = grad_disruption_jit(inp_arr, solver_1_times, solver_2_times)

    return None, None

    # return output, grads


if __name__ == "__main__":
    value, gradient = main()
    print(value,'\n', gradient)

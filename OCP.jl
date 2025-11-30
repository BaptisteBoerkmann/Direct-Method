using JuMP, Ipopt, Plots, MINPACK, LaTeXStrings, Printf, MAT



include("runge_kutta.jl")
include("docp.jl")


# Model Parameters
# Constants
c1 = 0.0025;     # k_on
c2 = 2500;       # TF_tot
c3 = 10;         # k_off
c4 = 20000;      # k_m
c5 = 0.05;       # k_m,deg,b
c6 = 1500;       # K_m
c7 = 0.8;        # k_rib,max
c8 = 0.1;        # μ_max
c9 = 2;          # k_u,trans
c10 = 0.3;       # delta
c11 = 1250;      # K_CU
c12 = 4000;      # beta
c13 = 0.001;     # gamma
c14 = 37;        # alpha
c15 = 10;        # Y_X/S
c16 = 250;       # U_max
c17 = 5;         # K_S
c18 = 1000000;   # K_IU
c19 = 2500;      # K_CI
c20 = 0.2;       # delta_1
c21 = 100;       # X(0)
c22 = 20;        # S(0)
c23 = 40;        # s_in


# Time variables
N = 3000;        # Amount of Time steps discretization size
t0 = 0;          # Initial time
tf = 30;         # Final Time


# Initial State
x1_0 = 1;        # Transcriptional Reporter
x2_0 = 1;        # mRNA
x3_0 = 1;        # U, unfolded proteins
x4_0 = 1;        # H, Hac1p transcription factor
x5_0 = 1;        # C, unfolded proteins buffer?
x6_0 = 0;        # FA, folded amylase
x7_0 = 20;       # S, substrate
x8_0 = 100;      # X(0)
x9_0 = 0;        # Objective for chemostat case



# Initial condition
init_batch = [x1_0, x2_0, x3_0, x4_0, x5_0, x6_0, x7_0];
init_chemo = [x1_0, x2_0, x3_0, x4_0, x5_0, x6_0, x7_0, x8_0, x9_0];


# Declare constrained variables
x_lb_b = [0, 0, 0, 0, 0, 0, 0];     # lower bound for x
x_lb_c = [0, 0, 0, 0, 0, 0, 0, 0, 0];
u_lb_b = [0];       # Control lower bound
u_ub_b = [100];     # Control upper bound
u_lb_c = [0,0];       # Control lower bound
u_ub_c = [100,0.01];     # Control upper bound


# System dynamics


function dyna_batch(x, u, i)
        μ = (1 - x[3]/(x[3] + c16)) * c8 * x[7]/(x[7] + c17)
        psi = x[3]/(x[3] + c18 + (c11 + c18 * x[5])/(c19 * (c11 + x[3])))

        dx1 =  c1 * u[1] * (c2 - x[1]) - c3 * x[1]
        dx2 = c4 * x[1]/(c6 + x[1]) - (c5 + μ) * x[2]
        dx3 = μ * c9 * x[2] *c7/c8   - c10 * (x[5] * x[3]/(x[3] + c11)) - μ * x[3]
        dx4 = μ * c12 * psi * c7/c8 - (c5 + μ) * x[4]
        dx5 =  μ * (c13 + c14 * x[4]) * c7/c8 - μ * x[5]
        dx6 = c20 * (x[5] * x[3]/(x[3] + c11)) * (c21 + c15 * (c22 - x[7]))
        dx7 = - μ * (c21 + c15 * (c22 - x[7]))/c15

        [ dx1, dx2, dx3, dx4, dx5, dx6, dx7]
end

function dyna_chemo(x, u, i)
    μ = (1 - x[3]/(x[3] + c16)) * c8 * x[7]/(x[7] + c17)
    psi = x[3]/(x[3] + c18 + (c11 + c18 * x[5])/(c19 * (c11 + x[3])))

    dx1 =  c1 * u[1] * (c2 - x[1]) - c3 * x[1]
    dx2 = c4 * x[1]/(c6 + x[1]) - (c5 + μ) * x[2]
    dx3 = μ * c9 * x[2] *c7/c8   - c10 * (x[5] * x[3]/(x[3] + c11)) - μ * x[3]
    dx4 = μ * c12 * psi * c7/c8 - (c5 + μ) * x[4]
    dx5 =  μ * (c13 + c14 * x[4]) * c7/c8 - μ * x[5]
    dx6 = c20 * (x[5] * x[3]/(x[3] + c11)) * x[8] -u[2]*x[6]
    dx7 = - μ * x[8]/c15 - u[2]*(x[7]-c23)
    dx8 =  μ * x[8] -u[2]*x[8]
    dx9 = u[2]*x[6]

    [ dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9]
end


# Optimization --------------------------------------------------------------


docp_batch = new_docp(7, 1, dyna_batch, N, (t0, tf),
                objective_index=6,
                initial_state=init_batch, state_lb=x_lb_b,
                control_lb=u_lb_b, control_ub=u_ub_b)
sol_batch = _solve!(docp_batch, (t0, tf), N, print_level=5)

println("-" ^ 40)
println("Objective value = ", sol_batch.obj)
println("-" ^ 40)

sol_batch


docp_chemo = new_docp(9, 2, dyna_chemo, N, (t0, tf),
                objective_index=9,
                initial_state=init_chemo, state_lb=x_lb_c,
                control_lb=u_lb_c, control_ub=u_ub_c)
sol_chemo = _solve!(docp_chemo, (t0, tf), N, print_level=5)

println("-" ^ 40)
println("Objective value = ", sol_chemo.obj)
println("-" ^ 40)

sol_chemo
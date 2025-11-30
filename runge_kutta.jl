# Define Runge-Kutta Butcher tableau data structure
struct rk_method
    name::Symbol
    s::Integer
    a::Matrix{<:Real}
    b::Vector{<:Real}
    c::Vector{<:Real}
end;

# Define Gauss-Legendre method (of order 2s) with s=2
# https://en.wikipedia.org/wiki/Gauss-Legendre_method
rk_data = rk_method(
    :gauss2,
    2,
    [0.25 (0.25-sqrt(3) / 6); (0.25+sqrt(3) / 6) 0.25],
    [0.5, 0.5],
    [(0.5 - sqrt(3) / 6), (0.5 + sqrt(3) / 6)],
);


# must be called after defining state and control
function autonomous_rk!(docp::GenericModel, # rk::rk_method,
                        dynamics_f::Function,
                        state_dim::Integer, time_steps::Integer, time_range)
    f = dynamics_f
    n = state_dim
    N = time_steps

    x = docp.obj_dict[:x]
    u = docp.obj_dict[:u]

    t1, t2 = time_range
    Δt = (t2 - t1) / (time_steps - 1)

    # always the same scheme for now
    rk = rk_data
    # define k arrays as optimization variables
    @variable(docp, k[1:rk.s, 1:N, 1:n])

    # define RK scheme as nonlienar optimization constraint
    @constraints(docp, begin
        # k[j,i] = f( x[i] + Δt Σ_s A[j,s]k[s,i] )
        rk_inner[j=1:rk.s, i=1:N], k[j,i,:] == f(x[i,:] + Δt *
                                                 sum(rk.a[j,s] * k[s,i,:] for s in 1:rk.s),
                                                 u[i,:],
                                                 i * Δt)
        # x[i+1] = x[i] + Δt Σ_j b[j]k[j,i]
        rk_scheme[i = 1:N-1], x[i+1,:] == x[i,:] +
                                          Δt * sum(rk.b[j] * k[j,i,:] for j in 1:rk.s)
    end)

    docp
end

export autonomous_rk!

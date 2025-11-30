const Bound = Union{Nothing, Vector}
const TimeRange{T<:AbstractFloat} = Union{AbstractRange{T}, AbstractVector{T}}

# Solution data structure
struct DOCPSolution
    t::Vector{Real}
    x::Matrix{Real}
    λ::Matrix{Real}
    u::Matrix{Real}
    obj::Real
end;

TimeRange(t0::T, tf::T, N::Ti) where {T<:AbstractFloat, Ti<:Integer} = t0:(tf-t0)/(N-1):tf
TimeRange(t0::Ti, tf::Ti, N::Ti) where {Ti<:Integer} = TimeRange(Float64(t0),
                                                                 Float64(tf), N)
time_bounds(t::TimeRange) = (t[1], t[end])
time_bounds(sol::DOCPSolution) = time_bounds(sol.t)
Base.length(sol::DOCPSolution) = length(sol.t)


function new_docp(state_dim::Integer, control_dim::Integer, dynamics_f::Function,
        time_steps::Integer, time_range; # , rk::rk_data=rk_method(:gauss2);
        objective_sense::Symbol=:max, objective_index::Integer=0,
        initial_state::Bound=nothing, initial_control::Bound=nothing,
        state_lb::Bound=nothing, state_ub::Bound=nothing,
        control_lb::Bound=nothing, control_ub::Bound=nothing)
    docp = JuMP.Model(Ipopt.Optimizer)

    n = state_dim
    m = control_dim
    N = time_steps



    # define state
    if isnothing(state_lb)
        if isnothing(state_ub)
            @variable(docp, x[1:N, 1:n])
        else
            @variable(docp, x[1:N, i = 1:n] <= state_ub[i])
        end
    else
        if isnothing(state_ub)
            @variable(docp, x[1:N, i = 1:n] >= state_lb[i])
        else
            @variable(docp, state_lb[i] <= x[1:N, i = 1:n] <= state_ub[i])
        end
    end
    if !isnothing(initial_state)
        set_initial_state!(docp, initial_state)
    end  

    # Fixed control
    if false
        fixed=[fill(0,1500);fill(100,1501)];
        #@variable(docp, fixed[j]<= u[j=1:N, i = 1:m] <= fixed[j])
        @variable(docp, u[1:N, 1:m])  # declare it normally

        # Then fix it to the desired values:
        for i in 1:m
            for j in 1:N
            fix(u[j, i], fixed[j]; force=true)
            end
        end
    else    
        # define control
        if isnothing(control_lb)
            if isnothing(control_ub)
                @variable(docp, u[1:N, 1:m])
            else
                @variable(docp, u[1:N, i = 1:m] <= control_ub[i])
            end
        else
            if isnothing(control_ub)
                @variable(docp, u[1:N, i = 1:m] >= control_lb[i])
            else
                @variable(docp, control_lb[i] <= u[1:N, i = 1:m] <= control_ub[i])
            end
        end
    end
    if !isnothing(initial_control)
        set_initial_control!(docp, initial_control)
    end
    

    # define RK scheme
    autonomous_rk!(docp, dynamics_f, state_dim, time_steps, time_range)

    # define objective
    if objective_sense == :max
        if objective_index == 0
            @objective(docp, Max, x[end, end])
        elseif 1 <= objective_index <= state_dim
            @objective(docp, Max, x[end, objective_index])
        else
            error("Objective index must belong to the state dimension bounds")
        end
    elseif objective_sense == :min
        if objective_index == 0
            @objective(docp, Min, x[end, end])
        elseif 1 <= objective_index <= state_dim
            @objective(docp, Min, x[end, objective_index])
        else
            error("Objective index must belong to the state dimension bounds")
        end
    else
        error("`objective_sense` must be :max or :min")
    end

    docp
end



function set_initial_state!(docp::GenericModel, x0::Vector{<:Real})
    @constraint(docp, initial_state, docp.obj_dict[:x][1,:] == x0[:])
end



function _solve!(docp::GenericModel, time_range, time_steps::Integer;
        tol=1e-6, const_tol=1e-8, max_iter=5000, print_level=5,
        mu_strategy="adaptive")
    # set optimizer options
    set_optimizer_attribute(docp, "print_level", print_level)
    set_optimizer_attribute(docp, "tol", tol)
    set_optimizer_attribute(docp, "constr_viol_tol", const_tol)
    set_optimizer_attribute(docp, "max_iter", max_iter)
    set_optimizer_attribute(docp, "mu_strategy", mu_strategy)

    # run solver
    optimize!(docp)
    if termination_status(docp) == MOI.OPTIMAL
        println("Optimal solution found", "\n")
    elseif termination_status(docp) == MOI.LOCALLY_SOLVED
        println("Local solution found", "\n")
    elseif termination_status(docp) == MOI.TIME_LIMIT && has_values(docp)
        println("Solution is suboptimal due to a time limit," *
                " but a primal solution is available", "\n")
    elseif termination_status(docp) == MOI.ITERATION_LIMIT && has_values(docp)
        println("Solution is suboptimal due to a iteration limit," *
                " but a primal solution is available", "\n")
    elseif termination_status(docp) == MOI.ALMOST_LOCALLY_SOLVED
        println("Solution converged to a stationary point," *
                " but a local optimal solution is available", "\n")
    elseif termination_status(docp) == MOI.LOCALLY_INFEASIBLE
        println("The algorithm converged to an infeasible point" *
                " or otherwise completed its search without finding" *
                " a feasible solution, without guarantees that no" *
                " feasible solution exists.")
    else
        error("The model was not solved correctly. Solver status: ",
                termination_status(docp), "\n")
    end

    # retrieve solution
    t0, tf = time_range
    t = TimeRange(t0, tf, time_steps)
    x = value.(docp.obj_dict[:x])
    u = value.(docp.obj_dict[:u])
    λ = dual.(docp.obj_dict[:rk_scheme])
    λ = -reduce(vcat, transpose.(λ))
    λ = vcat(λ, λ[end,:]')
    return DOCPSolution(t, x, λ, u, objective_value(docp))
end


export new_docp, _solve!

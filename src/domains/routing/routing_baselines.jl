function expected_hungarian!(server::RoutingAllocation, routing_sim::RoutingSimulator,
                              rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    available_drones = String[]
    for (drone_nm, dp) in server.agent_prop_set
        if server.agent_prop_set[drone_nm].at_depot == true
            push!(available_drones, drone_nm)
        end
    end

    available_pkgs = collect(keys(routing_sim.active_packages))

    n_drones = length(available_drones)
    n_pkgs = length(available_pkgs)

    if n_drones == 0 || n_pkgs == 0
        return
    end

    cost_matrix = ones(n_drones, n_pkgs)

    # Loop over drones X packages and assign cost matrix entries
    for (i, drone_nm) in enumerate(available_drones)
        for (j, pkg_nm) in enumerate(available_pkgs)

            depot_loc = server.agent_set[drone_nm].depot_loc
            pkg_loc = routing_sim.active_packages[pkg_nm].delivery

            if Distances.evaluate(EuclideanLatLongMetric(), convert_to_vector(depot_loc), convert_to_vector(pkg_loc)) <= routing_sim.distance_thresh

                # Compute the cost based on windows
                drone_pkg_window = server.agent_task_windows[(drone_nm, pkg_nm)]

                travel_time = drone_pkg_window[3] - drone_pkg_window[2] # SUCCESS - FINISH
                meanval = travel_time
                stdval = meanval/routing_sim.tt_est_std_scale
                tt_dist = Distributions.Epanechnikov(meanval, stdval)
                succ_prob = Distributions.cdf(tt_dist, drone_pkg_window[2])

                cost_matrix[i, j] = -succ_prob*routing_sim.delivery_reward

            end # if distance <= thresh
        end # for enumerate(available_pkgs)
    end # for enumerate(available_drones)


    if minimum(cost_matrix) == 1.0
        @warn "All matchings are failures"
        return
    end

    assignment, cost = hungarian(cost_matrix)

    for (i, assgn) in enumerate(assignment)

        if assgn != 0 && cost_matrix[i,assgn] < 1.0

            drone_nm = available_drones[i]
            pkg_nm = available_pkgs[assgn]

            server.agent_task_allocation[drone_nm] = (name=pkg_nm, time=Inf)

            routing_sim.true_delivery_return[(drone_nm, pkg_nm)] = sample_true_delivery_return_time(server.agent_task_windows[(drone_nm, pkg_nm)],
                                                                                                    server.current_time,
                                                                                                    routing_sim.tt_est_std_scale,
                                                                                                    rng)
            server.agent_prop_set[drone_nm].at_depot = false
            server.agent_prop_set[drone_nm].current_package = pkg_nm

            new_busy_package = routing_sim.active_packages[pkg_nm]
            routing_sim.busy_packages[pkg_nm] = new_busy_package
            delete!(routing_sim.active_packages, pkg_nm)
            routing_sim.num_active_packages -= 1

        else
            @info "Drone $(available_drones[i]) goes unassigned this round"
        end # assgn != 0
    end # for enumerate(assignment)

    # @infiltrate
end # end function


function earliest_due_date!(server::RoutingAllocation, routing_sim::RoutingSimulator,
                            rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # First collect all available drones
    available_drones = Dict{Int,Vector{String}}()

    # Split out available drones in terms of depot
    # Since drones in same depot will use priority ordering
    for (drone_nm, dp) in server.agent_prop_set

        drone = server.agent_set[drone_nm]
        if dp.at_depot == true
            if ~(haskey(available_drones, drone.depot_number))
                available_drones[drone.depot_number] = [drone_nm]
            else
                push!(available_drones[drone.depot_number], drone_nm)
            end
        end
    end # for (drone_nm, dp)

    all_assigned_pkgs = Set{String}()

    # Assign all available drones
    for (depot_number, depot_drones) in available_drones

        depot_loc = server.agent_set[depot_drones[1]].depot_loc

        pkgs_in_range = Set{String}()
        for (pkg_nm, pp) in routing_sim.active_packages
            if Distances.evaluate(EuclideanLatLongMetric(), convert_to_vector(depot_loc), convert_to_vector(pp.delivery)) <= routing_sim.distance_thresh
                push!(pkgs_in_range, pkg_nm)
            end
        end # for (pkg_nm, pp)

        for drone_nm in depot_drones

            # Iterate over interaction events and add first one
            # that is not already assigned
            ie_idx = 0
            for (idx, ie) in enumerate(server.agent_prop_set[drone_nm].interaction_events)
                if ~(ie.task_name in all_assigned_pkgs) && (ie.task_name in pkgs_in_range) &&
                    ~(haskey(routing_sim.busy_packages, ie.task_name))
                    ie_idx = idx
                end
            end

            # Now assign and sample true deliv-return times
            if ie_idx > 0
                pkg_nm = server.agent_prop_set[drone_nm].interaction_events[ie_idx].task_name

                if ~(haskey(routing_sim.busy_packages, pkg_nm))

                    push!(all_assigned_pkgs, pkg_nm)

                    server.agent_task_allocation[drone_nm] = (name=pkg_nm, time=Inf)

                    routing_sim.true_delivery_return[(drone_nm, pkg_nm)] = sample_true_delivery_return_time(server.agent_task_windows[(drone_nm, pkg_nm)],
                                                                                                            server.current_time,
                                                                                                            routing_sim.tt_est_std_scale,
                                                                                                            rng)
                    server.agent_prop_set[drone_nm].at_depot = false

                    new_busy_package = routing_sim.active_packages[pkg_nm]
                    routing_sim.busy_packages[pkg_nm] = new_busy_package
                    delete!(routing_sim.active_packages, pkg_nm)
                    routing_sim.num_active_packages -= 1
                end
            end # if ie_idx != -1
        end # for drone_nm in depot_drones
    end # for (dn, dd) in available_drones
end # end function


## Implementations for POMDPs.jl
@with_kw struct RoutingDroneState
    drone_idx::Int64
    pkg_assignment::Int64 # -1 at return, 0 at depot, and > 0 for on the way
    current_time::Int64
    true_delivery_return_tuple::Tuple{Float64,Float64}
end

"""
Only MCTS for routing
"""
@with_kw mutable struct RoutingMCTSMDP <: POMDPs.MDP{RoutingDroneState,Int64}
    routing_sim::RoutingSimulator
    server::RoutingAllocation
    horizon::Int64
    drone_pkg_assignment::Vector{Int64}
    true_delivery_return::Vector{Tuple{Float64,Float64}}
end

function update_routing_mcts_fullstate!(mdp::RoutingMCTSMDP, rng::AbstractRNG)

    drone_pkg_assignment = Int64[]
    true_delivery_return = Tuple{Float64,Float64}[]

    for (d, drone_nm) in enumerate(mdp.server.agent_ordering)

        drone_props = mdp.server.agent_prop_set[drone_nm]

        if ~(isempty(drone_props.current_package))
            p = parse(Int64, SubString(drone_props.current_package, 4))
            del_return = sample_true_delivery_return_time(mdp.server.agent_task_windows[(drone_nm, drone_props.current_package)],
                                                          mdp.server.current_time,
                                                          mdp.routing_sim.tt_est_std_scale,
                                                          rng)
            push!(true_delivery_return, del_return)
            push!(drone_pkg_assignment, p)
        else
            if drone_props.at_depot == true
                push!(drone_pkg_assignment, 0)
                push!(true_delivery_return, (0.0, 0.0))
            else
                # Drone is returning - should still have an assignment in agent_task_allocation
                # We SAMPLE that for the true del return time
                push!(drone_pkg_assignment, -1)

                del_return = nothing
                for (k, v) in mdp.server.agent_task_allocation
                    if k == drone_nm
                        pkg_nm, _ = v
                        del_return = sample_true_delivery_return_time(mdp.server.agent_task_windows[(drone_nm, pkg_nm)],
                                                                      mdp.server.current_time,
                                                                      mdp.routing_sim.tt_est_std_scale,
                                                                      rng)
                        break
                    end
                end

                # We MUST get a del_return by construction
                @assert ~(isnothing(del_return))

                push!(true_delivery_return, del_return)
            end
        end
    end

    mdp.drone_pkg_assignment = drone_pkg_assignment
    mdp.true_delivery_return = true_delivery_return
end


function get_current_routing_mcts_state(mdp::RoutingMCTSMDP, idx::Int64)
    return RoutingDroneState(idx, mdp.drone_pkg_assignment[idx], convert(Int64, mdp.server.current_time), mdp.true_delivery_return[idx])
end

POMDPs.discount(mdp::RoutingMCTSMDP) = 1.0
POMDPs.isterminal(mdp::RoutingMCTSMDP, s::RoutingDroneState) = (s.current_time == mdp.horizon)

function POMDPs.actions(mdp::RoutingMCTSMDP, s::RoutingDroneState)

    drone_nm = mdp.server.agent_ordering[s.drone_idx]
    in_range_pkgs = [0]

    if s.pkg_assignment == 0 # At depot

        pkgs_to_exclude = Set{Int64}()

        # Iterate over drones up to that point and exclude packages
        for i = 1:s.drone_idx-1
            if mdp.drone_pkg_assignment[i] > 0
                push!(pkgs_to_exclude, mdp.drone_pkg_assignment[i])
            end
        end

        depot_loc = mdp.server.agent_set[drone_nm].depot_loc
        for (pkg_nm, pp) in mdp.routing_sim.active_packages
            pkg_id = parse(Int64, SubString(pkg_nm, 4)) # Extracting number from name
            if Distances.evaluate(EuclideanLatLongMetric(), convert_to_vector(depot_loc), convert_to_vector(pp.delivery)) <= mdp.routing_sim.distance_thresh &&
                ~(pkg_id in pkgs_to_exclude)
                push!(in_range_pkgs, pkg_id)
            end
        end

    end

    return in_range_pkgs
end

function POMDPs.gen(mdp::RoutingMCTSMDP, s::RoutingDroneState, a::Int64, rng::AbstractRNG)

    new_pkg_assignment = s.pkg_assignment
    new_true_del_return_tuple = s.true_delivery_return_tuple
    next_time = s.current_time + 1
    on_time_pkg = 0


     if s.pkg_assignment == 0

         if a > 0
            drone_nm = mdp.server.agent_ordering[s.drone_idx]
            pkg_nm = string("pkg", a)
            new_true_del_return_tuple = sample_true_delivery_return_time(mdp.server.agent_task_windows[(drone_nm, pkg_nm)],
                                                                          mdp.server.current_time,
                                                                          mdp.routing_sim.tt_est_std_scale,
                                                                          rng)
            new_pkg_assignment = a
        end
    else
        @assert s.true_delivery_return_tuple != (0.0, 0.0)
        del_time, ret_time = s.true_delivery_return_tuple

        if s.pkg_assignment == -1
            if ret_time > s.current_time && ret_time < next_time # Back to base; otherwise change nothing
                new_pkg_assignment = 0
                new_true_del_return_tuple = (0.0, 0.0)
            end
        else
            # On the way
            pkg_nm = string("pkg", s.pkg_assignment)
            if del_time > s.current_time && del_time < next_time # Delivered in sim

                if haskey(mdp.routing_sim.busy_packages, pkg_nm)
                    tw = mdp.routing_sim.busy_packages[pkg_nm].time_window[2]
                elseif haskey(mdp.routing_sim.active_packages, pkg_nm)
                    tw = mdp.routing_sim.active_packages[pkg_nm].time_window[2]
                else
                    throw(ErrorException("Neither busy nor active has pkg!"))
                end

                if del_time <= tw
                    on_time_pkg = 1
                end
                new_pkg_assignment = -1 # On the way back
            end
        end # s.pkg_assignment == -1
    end # s.pkg_assignment == 0
    return (sp=RoutingDroneState(s.drone_idx, new_pkg_assignment, next_time, new_true_del_return_tuple), r = on_time_pkg)
end # end function


function update_server_with_mdp_action!(mdp::RoutingMCTSMDP, action::Vector{Int64}, rng::AbstractRNG)

    for (agent_action, drone_nm) in zip(action, mdp.server.agent_ordering)
        if agent_action > 0 # Do nothing otherwise
            pkg_nm = string("pkg", agent_action)
            mdp.server.agent_task_allocation[drone_nm] = (name=pkg_nm, time=Inf)

            mdp.routing_sim.true_delivery_return[(drone_nm, pkg_nm)] =
                        sample_true_delivery_return_time(mdp.server.agent_task_windows[(drone_nm, pkg_nm)],
                                                         mdp.server.current_time,
                                                         mdp.routing_sim.tt_est_std_scale,
                                                         rng)
            mdp.server.agent_prop_set[drone_nm].at_depot = false
            mdp.server.agent_prop_set[drone_nm].current_package = pkg_nm


            new_busy_package = mdp.routing_sim.active_packages[pkg_nm]
            mdp.routing_sim.busy_packages[pkg_nm] = new_busy_package
            delete!(mdp.routing_sim.active_packages, pkg_nm)
            mdp.routing_sim.num_active_packages -= 1
        end
    end
end

struct NearestPkgPolicy <: Policy
    mdp::RoutingMCTSMDP
end


function POMDPs.action(pol::NearestPkgPolicy, s::RoutingDroneState)

    mdp = pol.mdp
    action = 0

    if s.pkg_assignment == 0 # At depot, assign to nearest available
        pkgs_to_exclude = Set{Int64}()

        # Iterate over drones up to that point and exclude packages
        for i = 1:s.drone_idx-1
            if mdp.drone_pkg_assignment[i] > 0
                push!(pkgs_to_exclude, mdp.drone_pkg_assignment[i])
            end
        end

        drone_nm = mdp.server.agent_ordering[s.drone_idx]
        depot_loc = mdp.server.agent_set[drone_nm].depot_loc
        min_pkg = 0
        min_dist = mdp.routing_sim.distance_thresh

        for (pkg_nm, pp) in mdp.routing_sim.active_packages
            pkg_id = parse(Int64, SubString(pkg_nm, 4)) # Extracting number from name
            dist = Distances.evaluate(EuclideanLatLongMetric(), convert_to_vector(depot_loc), convert_to_vector(pp.delivery))
            if dist <= min_dist && ~(pkg_id in pkgs_to_exclude)
                min_dist = dist
                min_pkg = pkg_id
            end
        end

        if min_pkg != 0
            action = min_pkg
        end
    end

    return action
end

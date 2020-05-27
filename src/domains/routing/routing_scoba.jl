
delivery_util(reward::Float64, ie::InteractionEvent) = reward

"""
Look up the CDF of the Epanechnikov distribution for the arrival time.
"""
function delivery_success_prob(std_scale::Float64, ref_time::Float64, ie::InteractionEvent)

    travel_time = ie.timestamps[SUCCESS] - ie.timestamps[FINISH]

    # TODO: Need something more principled than this distribution
    meanval = travel_time
    stdval = travel_time/std_scale
    tt_dist = Distributions.Epanechnikov(meanval, stdval)

    # Compute the CDF of the finish time
    succ_prob = Distributions.cdf(tt_dist, ie.timestamps[FINISH])

    return succ_prob
end

"""
Valid tasks are those within the distance threshold from the depot.
"""
function scoba_routing!(server::RoutingAllocation, routing_sim::RoutingSimulator,
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

    # Diagnostics to send to CBA
    task_util_allocation = Dict{String,SCoBASolver.TaskUtil}()
    all_considered_tasks = Dict{String,Set{String}}()
    assignment_util = 0.0

    util_val_fn(ie) = delivery_util(routing_sim.delivery_reward, ie)
    success_prob(ref_time, ie) = delivery_success_prob(routing_sim.tt_est_std_scale,
                                ref_time, ie)

    # @infiltrate

    # Assign all available drones
    for (depot_number, depot_drones) in available_drones

        # Just get the location of any of them
        depot_loc = server.agent_set[depot_drones[1]].depot_loc

        # Iterate over active packages and check those that are in range
        pkgs_in_range = Set{String}()
        for (pkg_nm, pp) in routing_sim.active_packages
            if Distances.evaluate(EuclideanLatLongMetric(), convert_to_vector(depot_loc), convert_to_vector(pp.delivery)) <= routing_sim.distance_thresh
                push!(pkgs_in_range, pkg_nm)
            end
        end # for (pkg_nm, pp)

        depot_assigned_pkgs = Set{String}()

        # For drones from same depot, priority ordering
        for drone_nm in depot_drones

            # @info "Assigning available drones $(drone_nm)"

            # Exclude already assigned packages FROM DEPOT
            pkgs_to_consider = setdiff(pkgs_in_range, depot_assigned_pkgs)

            # @info "$(length(pkgs_to_consider)) packages considered!"

            # Run tree gen function!
            generate_search_tree!(server, drone_nm, pkgs_to_consider, success_prob, util_val_fn, 0.0)
            tree = server.agent_prop_set[drone_nm].tree

            # Get the attempt index and util if applicable
            if ~(isempty(tree))
                dec_idx = get_next_attempt_idx(tree)

                if dec_idx != -1
                    dec_node = tree.nodes[dec_idx]

                    # Update depot_considered tasks
                    push!(depot_assigned_pkgs, dec_node.task_name)

                    # Update things that will be sent to coordinator
                    assignment_util += dec_node.util
                    task_util_allocation[drone_nm] = (task=dec_node.task_name, util=dec_node.util)
                    all_considered_tasks[drone_nm] = pkgs_to_consider
                end

            end # if ~(isempty(tree))
        end # for drone in depot_drones
    end # for (depot_number, depot_drones)

    # Run coordinate_assignment to resolve conflicts
    if ~(isempty(task_util_allocation))

        scoba_alg = SCoBAAlgorithm(allocation=server)
        
        true_task_util_allocation = coordinate_allocation!(scoba_alg, task_util_allocation,
                                                        all_considered_tasks, assignment_util,
                                                        success_prob, util_val_fn, 0.0)

        # NOTE: might have redundancies in true_task_util_assignment - ignore!
        for (drone_nm, pkg_util) in true_task_util_allocation

            @assert haskey(server.agent_task_allocation, drone_nm) == false
            pkg_nm = pkg_util.task

            if ~(haskey(routing_sim.busy_packages, pkg_nm))

                server.agent_task_allocation[drone_nm] = (name=pkg_nm, time=Inf)

                # NOTE: set true delivery return time once assigned
                routing_sim.true_delivery_return[(drone_nm, pkg_nm)] = sample_true_delivery_return_time(server.agent_task_windows[(drone_nm, pkg_nm)],
                                                                                                        server.current_time,
                                                                                                        routing_sim.tt_est_std_scale,
                                                                                                        rng)

                server.agent_prop_set[drone_nm].at_depot = false
                server.agent_prop_set[drone_nm].current_package = pkg_nm

                # TODO: Deleting from active pkgs here for convenience
                new_busy_package = routing_sim.active_packages[pkg_nm]
                routing_sim.busy_packages[pkg_nm] = new_busy_package
                delete!(routing_sim.active_packages, pkg_nm)
                routing_sim.num_active_packages -= 1
            end
        end # for (drone, pkg_util)

    end # if ~isempty task_util_allocation
end # function

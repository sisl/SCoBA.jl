"""
Lookup the nearest neighbors and teh estimate matrix to get a rough estimate.
"""
function get_travel_time_estimate(halton_nn_tree::BallTree, loc1::LatLonCoords, loc2::LatLonCoords,
                                  estimate_matrix::Matrix{Float64}, time_scale::Float64)

    # Get the nearest neighbor for each location
    # IMP - If they are the same, return 0.0
    loc1_vect = convert_to_vector(loc1)
    loc1_idxs, _ = knn(halton_nn_tree, loc1_vect, 1)
    loc1_idx = loc1_idxs[1]

    loc2_vect = convert_to_vector(loc2)
    loc2_idxs, _ = knn(halton_nn_tree, loc2_vect, 1)
    loc2_idx = loc2_idxs[1]

    # Returns 0.0 if same!
    tt_est = estimate_matrix[loc1_idx, loc2_idx]

    # Use direct distance divided by speed if 0.0
    if tt_est == 0.0
        tt_est = Distances.evaluate(EuclideanLatLongMetric(), convert_to_vector(loc1), convert_to_vector(loc2)) / 0.00777 # Avg. drone speed
    end

    return tt_est/time_scale
end


function generate_package_request(lat_dist::Distribution, lon_dist::Distribution, current_time::Float64,
                                  tw_duration::Float64, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    delivery = (lat = rand(rng, lat_dist), lon = rand(rng, lon_dist))

    # Randomness in walking time and in duration
    start_time = current_time + rand(rng, Distributions.Uniform(tw_duration/2.0, tw_duration))
    duration = rand(rng, Distributions.Uniform(tw_duration/2.0, tw_duration))
    time_window = (start_time, start_time + duration)

    return Package(delivery, time_window)
end



function sample_true_delivery_return_time(drone_pkg_window::SVector{3,Float64}, current_time::Float64,
                                         std_scale::Float64, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    travel_time = drone_pkg_window[3] - drone_pkg_window[2]

    meanval = travel_time
    stdval = travel_time/std_scale
    tt_dist = Distributions.Epanechnikov(meanval, stdval)

    true_delivery_time = current_time + rand(rng, tt_dist)
    true_delivery_time = (true_delivery_time <= drone_pkg_window[1]) ? drone_pkg_window[1] : true_delivery_time

    true_return_time = ceil(true_delivery_time) + rand(rng, tt_dist)

    return (true_delivery_time, true_return_time)
end # end function



# For simulator - Set downtime based on distance between pickup and dropoff - but no uncertainty there
function setup_routing_sim(params_fn::String, halton_nn_tree, estimate_matrix::Matrix{Float64};
                           num_init_requests::Int64=5, new_request_prob::Float64=0.75, delivery_reward::Float64=1000.0,
                           time_window_duration::Float64=30.0, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Parse the parameters for the relevant city
    city_params = parse_city_params(params_fn)
    lat_dist = Distributions.Uniform(city_params.lat_start, city_params.lat_end)
    lon_dist = Distributions.Uniform(city_params.lon_start, city_params.lon_end)

    # Update the number of active packages etc.
    active_packages = Dict{String,Package}()

    for n = 1:num_init_requests
        new_package_name = string("pkg",n)

        new_package = generate_package_request(lat_dist, lon_dist, 0.0, time_window_duration, rng)
        active_packages[new_package_name] = new_package
    end

    return RoutingSimulator(current_time=0.0, city_params=city_params, new_request_prob=new_request_prob, delivery_reward=delivery_reward,
                            time_window_duration=time_window_duration, halton_nn_tree=halton_nn_tree, estimate_matrix=estimate_matrix,
                            active_packages=active_packages, num_total_packages=num_init_requests, num_active_packages=num_init_requests)
end

## Need something that maintains the TRUE pickup and dropoff times but only uses them when called upon
# 1. Generate new request based on probability
# 2. If new time is > TRUE pickup, simulate pickup (same for dropoff)
# 3. Assign drones as free or active; Update current pos of drone if picked / dropped off
function update_routing_sim!(sim::RoutingSimulator, server::RoutingAllocation,
                            rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # First update the simulator and server time
    old_time = sim.current_time
    sim.current_time += 1
    server.current_time += 1

    curr_drone_locs_cols = Tuple{LatLonCoords,Symbol}[]
    curr_sites_locs_cols = Tuple{LatLonCoords,Symbol}[]

    # Simulate any true pickups and dropoffs
    keys_to_del = Set{Tuple{String,String}}()
    busy_packages_to_del = Set{String}()
    done_packages_to_del = Set{String}()
    for (k,v) in server.agent_task_allocation
        drone_nm = k
        package_nm, _ = v

        # Lookup true delivery and return times
        delivery, return_time = sim.true_delivery_return[(drone_nm, package_nm)]

        # For pickup, assign package to drone and mark package inactive
        if delivery > old_time && delivery <= sim.current_time

            server.agent_prop_set[drone_nm].current_package = ""
            server.agent_prop_set[drone_nm].at_depot = false # For good measure

            # Need to check if it reached in time!
            if delivery <= sim.busy_packages[package_nm].time_window[2]
                sim.delivered_packages += 1
                @info "$(drone_nm) has delivered $(package_nm)!"

                # Drone is green and at package location
                push!(curr_drone_locs_cols, (sim.busy_packages[package_nm].delivery, :green))

            else
                sim.late_packages += 1
                @info "$(package_nm) was not delivered in time!"

                # Drone is green and at package location
                push!(curr_drone_locs_cols, (sim.busy_packages[package_nm].delivery, :red))

            end

            push!(busy_packages_to_del, package_nm)
            sim.done_packages[package_nm] = sim.busy_packages[package_nm]

            # Increment loss with difference from start of window
            sim.sum_of_delivery_time += delivery - server.agent_task_windows[(drone_nm, package_nm)][1]

        # For dropoff, free up drone and setup deletion of (drone,package) keys
        elseif return_time > old_time && return_time <= sim.current_time

            # Drone at depot again
            server.agent_prop_set[drone_nm].at_depot = true
            push!(keys_to_del, (drone_nm, package_nm))
            @info "$(drone_nm) back at depot!"
            push!(done_packages_to_del, package_nm)

            # Drone is blue and at depot
            push!(curr_drone_locs_cols, (server.agent_set[drone_nm].depot_loc, :blue))

        else
            ## NOTE: Super hacky, only for plotting
            if haskey(sim.busy_packages, package_nm) == false && haskey(sim.done_packages, package_nm) == false
                continue
            end
            depot_loc = server.agent_set[drone_nm].depot_loc

            if sim.current_time < delivery
                pp_loc = sim.busy_packages[package_nm].delivery
                maxdiff = max(delivery, return_time - delivery)
                interp_factor = (delivery - sim.current_time)/maxdiff
                new_lat = depot_loc.lat + (1.0 - interp_factor)*(pp_loc.lat - depot_loc.lat)
                new_lon = depot_loc.lon + (1.0 - interp_factor)*(pp_loc.lon - depot_loc.lon)
            else
                pp_loc = sim.done_packages[package_nm].delivery
                interp_factor = (return_time - sim.current_time)/(return_time - delivery)
                new_lat = depot_loc.lat + (interp_factor)*(pp_loc.lat - depot_loc.lat)
                new_lon = depot_loc.lon + (interp_factor)*(pp_loc.lon - depot_loc.lon)
                # @infiltrate
            end
            # @infiltrate

            push!(curr_drone_locs_cols, ((lat=new_lat, lon=new_lon), :blue))

        end # if delivery >
    end # for (k,v)

    # Delete bookkeeping keys for dropped off package
    for k in keys_to_del
        delete!(sim.true_delivery_return, k)
        delete!(server.agent_task_windows, k)
        delete!(server.agent_task_allocation, k[1]) # Only delete the agent key
    end

    for r in busy_packages_to_del
        delete!(sim.busy_packages, r)
    end # for r

    for r in done_packages_to_del
        delete!(sim.done_packages, r)
    end

    # Now deal with deliveries that were never attempted?
    packages_to_del = Set{String}()
    for (package_nm, rp) in sim.active_packages
        if rp.time_window[2] <= sim.current_time
            @info "$(package_nm) was not even attempted!"
            sim.late_packages += 1
            push!(packages_to_del, package_nm)

            # Add package location with red
            push!(curr_sites_locs_cols, (rp.delivery, :red))
        end
    end

    for r in packages_to_del
        delete!(sim.active_packages, r)
        sim.num_active_packages -= 1
    end # for r


    # Finally, generate new packages
    if rand(rng) <= sim.new_request_prob
        lat_dist = Distributions.Uniform(sim.city_params.lat_start, sim.city_params.lat_end)
        lon_dist = Distributions.Uniform(sim.city_params.lon_start, sim.city_params.lon_end)
        new_package = generate_package_request(lat_dist, lon_dist, sim.current_time,
                                                sim.time_window_duration, rng)
        sim.num_total_packages += 1
        sim.num_active_packages += 1
        new_package_nm = string("pkg",sim.num_total_packages)
        @info "$(new_package_nm) added!"
        sim.active_packages[new_package_nm] = new_package
    end # if rand(rng)

    # Grey packages
    for (pkg_nm, pp) in sim.active_packages
        push!(curr_sites_locs_cols, (pp.delivery, :grey))
    end
    for (pkg_nm, pp) in sim.busy_packages
        push!(curr_sites_locs_cols, (pp.delivery, :grey))
    end

    sim.curr_drone_site_locs = CurrDroneSiteLocs(curr_drone_locs_cols, curr_sites_locs_cols)
end # function


function update_time_windows!(sim::RoutingSimulator, server::RoutingAllocation)

    @assert sim.current_time == server.current_time

    # We look at all active packages and update their time windows w.r.t the drones
    for (drone_nm,drone_props) in server.agent_prop_set

        interaction_events = server.agent_prop_set[drone_nm].interaction_events
        drone = server.agent_set[drone_nm]

        for (package_nm,rp) in sim.active_packages
            if ~(haskey(server.agent_task_windows, (drone_nm, package_nm)))
                # Compute downtime in terms of dropoff time
                return_time = get_travel_time_estimate(sim.halton_nn_tree, rp.delivery,
                                                        drone.depot_loc, sim.estimate_matrix, sim.time_scale)

                # Insert interaction event timestamps
                ie_timestamps = Dict{MODE,Float64}(START=>rp.time_window[1], FINISH=>rp.time_window[2],
                                                   SUCCESS=>ceil(rp.time_window[2])+return_time)
                push!(interaction_events, InteractionEvent(drone_nm, package_nm, ie_timestamps))

                # Update time windows on the assignment server side
                server.agent_task_windows[(drone_nm, package_nm)] = [rp.time_window[1], rp.time_window[2], ceil(rp.time_window[2])+return_time]
            end # if ~(haskey)
        end # for (package,rp)

        # Sort the temporal event list
        sort!(interaction_events, by=x->x.timestamps[SUCCESS])
        server.agent_prop_set[drone_nm].interaction_events = interaction_events
    end

end # function update_time_windows

function parse_routing_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--trials", "-n"
            arg_type = Int64
            default = 100
        "--timesteps", "-t"
            arg_type = Int64
            default = 500
        "--n_drones", "-d"
            arg_type = Int64
        "--n_depots", "-e"
            arg_type = Int64
        "--new_request_prob", "-p"
            arg_type = Float64
        "--time_window", "-w"
            arg_type = Float64
        "baseline"
            help = "Which method to evaluate (dispatch/hungarian/css)"
            required = true
        "out_file_name"
            required = true
    end

    return parse_args(s)
end # end function parse routing

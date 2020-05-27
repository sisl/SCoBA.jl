using Statistics
using Random
using Distributions
using JSON
using POMDPs
using MCTS
using NearestNeighbors
using GeometryTypes
using JLD2

using SCoBA
using SCoBA.SCoBADomains
using SCoBA.SCoBADomains: LatLonCoords
using SCoBA.SCoBASolver

# Hardcode the depot locations to get good coverage
const DEPOT1 = LatLonCoords((lat = 37.774524, lon = -122.473656))
const DEPOT2 = LatLonCoords((lat = 37.751751, lon = -122.410654))
const DEPOT3 = LatLonCoords((lat = 37.718779, lon = -122.462401))
const DEPOT4 = LatLonCoords((lat = 37.789290, lon = -122.426797))
const DEPOT5 = LatLonCoords((lat = 37.739611, lon = -122.492203))
const DEPOTLOCS = [DEPOT1, DEPOT2, DEPOT3, DEPOT4, DEPOT5]
const TRAVELTIME_EST = "./param_files/sf_halton_tt_estimates_scoba.jld2"
const PARAMS_FN = "./param_files/sf_bb_params.toml"
const rng = MersenneTwister(1345)
const mcts_params = (trials=100, explore=0.1, depth=20)

function main(n_depots::Int64, n_drones::Int64, n_requests::Int64, outfile::String,
              n_trials::Int64=10, method::String="scoba")

    # To store the computation times
    comp_times = Float64[]

    # Load up travel time estimates
    @load TRAVELTIME_EST  scoba_halton_tree travel_time_estimates

    city_params = SCoBADomains.parse_city_params(PARAMS_FN)
    lat_dist = Uniform(city_params.lat_start, city_params.lat_end)
    lon_dist = Uniform(city_params.lon_start, city_params.lon_end)

    # Iterate over trials
    for t = 1:n_trials+1

        # Define set of drones
        drones_per_depot = div(n_drones, n_depots)
        drone_ordering = String[]
        drone_set = Dict{String,Drone}()

        # Distribute drones equally across depots
        n = 1
        for nd = 1:n_depots
            for dd = 1:drones_per_depot

                drone_nm = string("dn", n)
                n += 1
                new_drone = Drone(depot_number=nd, depot_loc=DEPOTLOCS[nd])

                push!(drone_ordering, drone_nm)
                drone_set[drone_nm] = new_drone
            end
        end # for in n_depots

        # Boilerplate code since drones don't have any intrinsic properties here
        drone_prop_set = Dict{String,DroneProperties}()
        for drone_nm in drone_ordering
            drone_prop_set[drone_nm] = DroneProperties()
        end

        drone_server = RoutingAllocation(agent_set=drone_set, agent_prop_set=drone_prop_set,
                                         agent_ordering=drone_ordering,
                                         max_tasks_to_consider=20,
                                         conflict_threshold=20)

        sim = setup_routing_sim(PARAMS_FN, scoba_halton_tree, travel_time_estimates,
                                num_init_requests=n_requests,
                                new_request_prob=1.0, rng=rng)
        
        if method == "scoba"

            update_time_windows!(sim, drone_server)
            t = @elapsed scoba_routing!(drone_server, sim, rng)

            push!(comp_times, t)
        
        elseif method == "hungarian"

            update_time_windows!(sim, drone_server)
            t = @elapsed expected_hungarian!(drone_server, sim, rng)

            push!(comp_times, t)

        elseif method == "mcts"

            mdp = RoutingMCTSMDP(sim, drone_server, 100, Int64[], Tuple{Float64,Float64}[])
            nearestpol = NearestPkgPolicy(mdp)

            mcts = MCTSSolver(n_iterations=mcts_params.trials, depth=mcts_params.depth, rng=rng,
                              exploration_constant=mcts_params.explore,
                              reuse_tree=false,
                              estimate_value = RolloutEstimator(nearestpol))
            mcts_policy = POMDPs.solve(mcts, mdp)

            update_time_windows!(sim, drone_server)
            update_routing_mcts_fullstate!(mdp, rng)

            curr_action = Int64[]

            sum_t = 0.0

            for idx = 1:n_drones

                drone_state = get_current_routing_mcts_state(mdp, idx)
                t = @elapsed drone_action = POMDPs.action(mcts_policy, drone_state)
                mdp.drone_pkg_assignment[idx] = drone_action
                push!(curr_action, drone_action)
                sum_t += t
            end

            push!(comp_times, sum_t)
        end
    end # t = 1:n_Trials + 1

    meanval =  mean(comp_times[2:end]) # Ignore first time
    @show meanval

    timing_results = Dict("mean"=>meanval,
                          "sem"=>std(comp_times[2:end])/sqrt(n_trials),
                          "median"=>median(comp_times[2:end]))

    open(outfile, "w") do f
        JSON.print(f, timing_results, 2)
    end
end # main

# Example call to main - for 5 depots, 15 drones, 50 requests, and 50 trials
# main(5, 15, 50, "routing_time_trial50_dr15dep5req50.json", 50)
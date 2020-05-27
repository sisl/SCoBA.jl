using Statistics
using Random
using Distributions
using StaticArrays
using JSON
using POMDPs
using MCTS
using Logging
using JLD2
using POMDPs
using POMDPModels
using POMDPPolicies



using SCoBA
using SCoBA.SCoBADomains
using SCoBA.SCoBADomains: LatLonCoords
using SCoBA.SCoBASolver

# For warnings about exceeding conflict thresholds
global_logger(SimpleLogger(stderr, Logging.Warn))

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



function main()

    # Get command line input from script
    parsed_args = parse_routing_commandline()

    # Set up bookkeeping datastructures for stats
    trials = parsed_args["trials"]
    results_dict = Dict()
    results_dict["trials"] = trials

    late_pkgs = Int64[]
    delivered_pkgs = Int64[]
    total_pkgs = Int64[]

    # Print the filename for tracking from the high-level bash script
    println(parsed_args["out_file_name"])

    @load TRAVELTIME_EST scoba_halton_tree travel_time_estimates

    # Define set of drones
    drones_per_depot = div(parsed_args["n_drones"], parsed_args["n_depots"])
    drone_ordering = String[]
    drone_set = Dict{String,Drone}()

    n = 1
    for nd = 1:parsed_args["n_depots"]
        for dd = 1:drones_per_depot

            drone_nm = string("dn", n)
            n += 1
            new_drone = Drone(depot_number=nd, depot_loc=DEPOTLOCS[nd])

            push!(drone_ordering, drone_nm)
            drone_set[drone_nm] = new_drone
        end
    end # for in n_depots


    # NOTE: Set number of initial requests slightly higher than drones
    num_init_requests = convert(Int64, round(1.5 * parsed_args["n_drones"]))


    if parsed_args["baseline"] == "mcts" # Special things for MDP

        for i = 1:trials

            println("Trial $(i)")

            # Reset drone prop set
            drone_prop_set = Dict{String,DroneProperties}()
            for drone_nm in drone_ordering
                drone_prop_set[drone_nm] = DroneProperties()
            end

            drone_server = RoutingAllocation(agent_set=drone_set, agent_prop_set=drone_prop_set,
                                             agent_ordering=drone_ordering,
                                             max_tasks_to_consider=20,
                                             conflict_threshold=10)

            # Reset sim
            sim = setup_routing_sim(PARAMS_FN, scoba_halton_tree, travel_time_estimates,
                                    num_init_requests=num_init_requests,
                                    new_request_prob=parsed_args["new_request_prob"],
                                    time_window_duration=parsed_args["time_window"],
                                    rng=rng)

            # Setup MDP and MCTS solver
            mdp = RoutingMCTSMDP(sim, drone_server, parsed_args["timesteps"], Int64[], Tuple{Float64,Float64}[])
            nearestpol = NearestPkgPolicy(mdp)

            mcts = MCTSSolver(n_iterations=mcts_params.trials, depth=mcts_params.depth, rng=rng,
                              exploration_constant=mcts_params.explore,
                              reuse_tree=false,
                              estimate_value = RolloutEstimator(nearestpol))
            mcts_policy = POMDPs.solve(mcts, mdp)

            for t = 1:parsed_args["timesteps"]

                update_time_windows!(sim, drone_server)
                update_routing_mcts_fullstate!(mdp, rng)
                
                # Get joint action from individual agent concatenation
                curr_action = Int64[]
                for idx = 1:parsed_args["n_drones"]

                    drone_state = get_current_routing_mcts_state(mdp, idx)
                    drone_action = POMDPs.action(mcts_policy, drone_state)
                    mdp.drone_pkg_assignment[idx] = drone_action
                    push!(curr_action, drone_action)
                end

                update_server_with_mdp_action!(mdp, curr_action, rng)
                update_routing_sim!(mdp.routing_sim, mdp.server, rng)

            end

            push!(late_pkgs, sim.late_packages)
            push!(delivered_pkgs, sim.delivered_packages)
            push!(total_pkgs, sim.num_total_packages)
        end # for i = 1:trials

    else
        # One of the non-MDP methods
        if parsed_args["baseline"] == "edd"
            baseline_fn = earliest_due_date!
        elseif parsed_args["baseline"] == "hungarian"
            baseline_fn = expected_hungarian!
        elseif parsed_args["baseline"] == "scoba"
            baseline_fn = scoba_routing!
        else
            throw(ErrorException("Incorrect baseline argument $(parsed_args["baseline"])!"))
        end

        for i = 1:trials

            println("Trial $(i)")

            # Reset drone prop set
            drone_prop_set = Dict{String,DroneProperties}()
            for drone_nm in drone_ordering
                drone_prop_set[drone_nm] = DroneProperties()
            end

            drone_server = RoutingAllocation(agent_set=drone_set, agent_prop_set=drone_prop_set, agent_ordering=drone_ordering)

            # Reset sim
            sim = setup_routing_sim(PARAMS_FN, scoba_halton_tree, travel_time_estimates,
                                    num_init_requests=num_init_requests,
                                    new_request_prob=parsed_args["new_request_prob"],
                                    time_window_duration=parsed_args["time_window"],
                                    rng=rng)

            for t = 1:parsed_args["timesteps"]
                update_time_windows!(sim, drone_server)
                baseline_fn(drone_server, sim, rng)
                update_routing_sim!(sim, drone_server, rng)
            end

            push!(late_pkgs, sim.late_packages)
            push!(delivered_pkgs, sim.delivered_packages)
            push!(total_pkgs, sim.num_total_packages)
        end # for i = 1:trials
    end # Outer if

    results_dict["late"] = late_pkgs
    results_dict["delivered"] = delivered_pkgs
    results_dict["total"] = total_pkgs

    # Print to output file
    open(parsed_args["out_file_name"], "w") do f
        JSON.print(f, results_dict, 2)
    end
end # main

main()

# Example call to routing script
# You can also run julia1.4 scripts/benchmark_routing.jl --help to get details of the command line params
# julia1.4 scripts/benchmark_routing.jl -n 10 -t 100 -d 15 -e 5 -p 0.5 -w 30 scoba r_dr15_dep5_probpt5_win30_scoba.json
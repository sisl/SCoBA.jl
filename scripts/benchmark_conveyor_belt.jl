using Statistics
using Random
using Distributions
using StaticArrays
using JSON
using POMDPs
using MCTS
using Logging
using POMDPs
using JLD2
using SCoBA
using SCoBA.SCoBADomains
using SCoBA.SCoBASolver

global_logger(SimpleLogger(stderr, Logging.Error))


const rng = MersenneTwister(5678)
const mcts_params = (disc=0.02, trials=100, explore=0.1, depth=20)
const qlearning_params = (disc=0.05, max_boxes=3)

function main()

    # Get command line input from script
    parsed_args = parse_conveyor_commandline()
   
    # Set up bookkeeping datastructures for stats
    trials = parsed_args["trials"]
    results_dict = Dict()
    results_dict["trials"] = trials
    lost_boxes = Int64[]
    deposited_boxes = Int64[]

    # Print the filename for tracking from the high-level bash script
    println(parsed_args["out_file_name"])

    # Iterate over different randomized trials
    for i = 1:trials

        println("Trial $(i)")
        grasp_probs = [parse(Float64, p) for p in split(parsed_args["grasp_probs"], ",")]
        server, cbelt = setup_conveyor_scenario(grasp_probs=grasp_probs,
                                            belt_speed=parsed_args["belt_speed"],
                                            new_box_prob=parsed_args["new_box_prob"])
        
        # Check which method was specified and benchmark accordingly
        if parsed_args["baseline"] == "mcts"

            # Create full MMDP version of problem
            mdp = create_convbelt_mdp(cbelt, server, mcts_params.disc, convert(Int64, parsed_args["timesteps"]))

            # Heuristic EDD Rollout Policy for MCTS Solver
            eddpol = EDDPolicy(mdp)
            mcts = MCTSSolver(n_iterations=mcts_params.trials, depth=mcts_params.depth,
                              rng=rng, exploration_constant=mcts_params.explore,
                              reuse_tree=false,
                              estimate_value = RolloutEstimator(eddpol))
            
            mcts_policy = POMDPs.solve(mcts, mdp)

            # Simulate for timesteps and collect stats
            for t = 1:parsed_args["timesteps"]

                # Look at current belt and update time window information online
                update_time_windows!(mdp.cbelt, mdp.server)

                # Current (s,a) for MCTS
                curr_state = get_current_belt_mdp_state(mdp, rng)
                curr_action = POMDPs.action(mcts_policy, curr_state)
                
                # Execute action on problem
                update_server_with_mdp_action!(mdp, curr_action)

                # Simulate belt and arms for one step
                propagate_belt!(mdp.cbelt, mdp.server)
                propagate_arms!(mdp.server, mdp.cbelt, rng)
                mirror_gen_step!(mdp.cbelt, rng)
            end # t = 1:timesteps
            server = mdp.server
            cbelt = mdp.cbelt

        elseif parsed_args["baseline"] == "qlearning"

            # Load up QLearning policy trained offline
            # See scripts/qlearning_conveyor_belt.jl for how it was trained
            @load parsed_args["qpol_name"] policy

            # Set up single arm MDP for Independent QLearning (IQL) to simulate on
            mdp = create_convbelt_singlearm_mdp(cbelt, server,
                                        qlearning_params.disc, qlearning_params.max_boxes)

            for t = 1:parsed_args["timesteps"]

                update_time_windows!(mdp.cbelt, mdp.server)
                vec_action = Int64[]

                # IQL; hence get individual actions separately and then concatenate
                for ag_idx = 1:length(mdp.server.agent_ordering)
                    state = get_singlearm_mdp_state(mdp, ag_idx)
                    a = POMDPs.action(policy, state)
                    push!(vec_action, a)
                end

                # Execute action on problem
                update_server_with_mdp_action!(mdp, vec_action)

                propagate_belt!(mdp.cbelt, mdp.server)
                propagate_arms!(mdp.server, mdp.cbelt, rng)
                mirror_gen_step!(mdp.cbelt, rng)

            end # t = 1:timesteps 
            server = mdp.server
            cbelt = mdp.cbelt

        else
            
            # One of the non-MDP methods
            if parsed_args["baseline"] == "edd"
                baseline_fn = earliest_due_date!
            elseif parsed_args["baseline"] == "hungarian"
                baseline_fn = expected_hungarian!
            elseif parsed_args["baseline"] == "scoba"
                baseline_fn = scoba_conveyor!
            else
                throw(ErrorException("Incorrect baseline argument $(parsed_args["baseline"])!"))
            end    

            for t = 1:parsed_args["timesteps"]

                update_time_windows!(cbelt, server)

                # Apply the method, whatever it is
                baseline_fn(server, cbelt)

                propagate_belt!(cbelt, server)
                propagate_arms!(server, cbelt, rng)

                mirror_gen_step!(cbelt, rng)
            end # t = 1:timesteps

        end

        # Update terminal counts if boxes grabbed but not deposited
        count_terminal_grabbed_boxes!(cbelt, server)

        push!(lost_boxes, cbelt.lost_boxes)
        push!(deposited_boxes, cbelt.deposited_boxes)
    end # for i=1:trials

    results_dict["lost"] = lost_boxes
    results_dict["deposited"] = deposited_boxes

    # Print to output file
    open(parsed_args["out_file_name"], "w") do f
        JSON.print(f, results_dict, 2)
    end

end # main

# Run the function and fire away!
main()


# Example call to script
# You can also run julia1.4 scripts/benchmark_conveyor_belt.jl --help to get details of the command line params
# julia1.4 scripts/benchmark_conveyor_belt.jl -g 0.6,0.6,0.6 -n 10 -t 500 scoba cbelt_grasp_pt6_scoba.json
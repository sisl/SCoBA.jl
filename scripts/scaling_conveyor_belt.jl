using Statistics
using Random
using Distributions
using JSON
using POMDPs
using POMDPPolicies
using MCTS
using SCoBA
using SCoBA.SCoBADomains
using SCoBA.SCoBASolver


# Vary the number of objects as START:SKIP:STOP
const STOP = 200
const SKIP = 40
const START = 40

# For random box generation reproducibility
rng = MersenneTwister(1234)

function main(outfile::String, baseline::String="scoba", n_trials::Int64=100)

    results = Dict()

    for n_boxes = START:SKIP:STOP

        @show n_boxes

        comp_times = Float64[]
        num_nodes = Int64[]

        # Do one more trial as we will ignore the first (Julia)
        for t = 1:n_trials+1

            server, cbelt = setup_conveyor_scenario()

            
            if baseline == "scoba"

                # Add a batch of boxes
                for _ = 1:n_boxes
                    # choose a point in the first arm's workspace
                    start_point = rand(rng, Uniform(0.05, 0.35))
                    add_box_to_belt!(cbelt, start_point)
                end # 1:n_boxes

                # Default steps to run SCoBA
                update_time_windows!(cbelt, server)

                pickup_util(ie) = 1.0
                success_prob_fn_first(rt, sf) = cumul_grasp_prob(server.agent_set["arm1"], rt, sf)

                # Measure the time to run SCoBA tree generation
                time = @elapsed generate_search_tree!(server, "arm1",
                                                    server.agent_prop_set["arm1"].workspace_objs,
                                                    success_prob_fn_first, pickup_util,
                                                    server.agent_set["arm1"].downtime)

                # Update the statistics
                push!(comp_times, time)
                push!(num_nodes, length(server.agent_prop_set["arm1"].tree.nodes))

            elseif baseline == "hungarian"

                for _ = 1:n_boxes
                    # choose a point in the first arm's workspace
                    start_point = rand(rng, Uniform(0.05, 0.95))
                    add_box_to_belt!(cbelt, start_point)
                end # 1:n_boxes

                # Default steps to run SCoBA
                update_time_windows!(cbelt, server)

                time = @elapsed expected_hungarian!(server, cbelt)
                push!(comp_times, time)
            end

        end # for t in n_trials

        meantime = mean(comp_times[2:end])
        meannodes = Inf
        if ~isempty(num_nodes)
            meannodes = mean(num_nodes[2:end])
        end
        results[n_boxes] = Dict("meantime"=>meantime, "semtime"=>std(comp_times[2:end])/sqrt(n_trials), "medtime"=>median(comp_times[2:end]),
                               "meannodes"=>meannodes)
    end

    # Write to JSON outfile
    open(outfile, "w") do f
        JSON.print(f, results, 2)
    end
end # end main

# Example call to main
# main("conveyor_belt_scaling_scoba.json", "scoba", 100)
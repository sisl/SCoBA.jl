## Conflict-Based Allocation
const TaskUtil = NamedTuple{(:task, :util), Tuple{String,Float64}}

@with_kw mutable struct SCoBAHighLevelNode
    task_allocation::Dict{String,TaskUtil}      = Dict{String,TaskUtil}()    # Agent name |--> task name
    considered_tasks::Dict{String,Set{String}}  = Dict{String,Set{String}}()  # Agent name |--> tasks to consider
    util::Float64                               = 0.0
    id::Int64                                   = 0
end
Base.isless(hln1::SCoBAHighLevelNode, hln2::SCoBAHighLevelNode) = hln1.util < hln2.util

@with_kw mutable struct SCoBAAlgorithm
    allocation::GenericAllocation
    heap::MutableBinaryMaxHeap{SCoBAHighLevelNode}    = MutableBinaryMaxHeap{SCoBAHighLevelNode}()
    num_total_conflicts::Int64                        = 0
end

# Inspired by CBS but with allocation instead of path search
function coordinate_allocation!(solver::SCoBAAlgorithm, initial_task_allocation::Dict{String,TaskUtil},
                                initial_considered_tasks::Dict{String,Set{String}}, init_util::Float64,
                                success_prob_fn::Function, util_val_fn::Function,
                                downtime::Float64)

    # Create start node with initial info
    start = SCoBAHighLevelNode(task_allocation=initial_task_allocation,
                                considered_tasks=initial_considered_tasks,
                                util=init_util)

    push!(solver.heap, start)
    id = 1

    # Iterative case: Pop top of heap
    # Check if any conflict in allocation
    # If no, then return solution. Otherwise, generate child nodes that re-assign tasks accordingly
    while ~(isempty(solver.heap))

        P = pop!(solver.heap)

        # Check for conflicting allocations
        task_to_agent = Dict{String,Vector{String}}()
        for (agent, task_util) in P.task_allocation

            task = task_util.task

            if ~(haskey(task_to_agent, task))
                task_to_agent[task] = [agent]
            else
                push!(task_to_agent[task], agent)
            end
        end # for (a,t) in task_alloc

        # Now look for tasks assigned to multiple agents
        num_conflicts = 0

        tasks_to_ignore = Set{String}(collect(keys(task_to_agent)))

        for (task, alloc_agents) in task_to_agent

            # If alloc_agents has more than one, then we need to generate constraints etc.
            if length(alloc_agents) > 1

                # For each of the assigned agents, generate a new high-level node
                # where the over-assigned task is removed from each of the other assigned agents
                for agt in alloc_agents

                    new_node = deepcopy(P)
                    new_node.id = id

                    # Create a set of all agents other than agt
                    other_agents_set = Set{String}(alloc_agents)
                    setdiff!(other_agents_set, [agt])

                    # For each of other agents, remove task from considered_tasks and recompute low level tree
                    for other_agt in other_agents_set

                        # We KNOW that other_agt will not be assigned to task again
                        # Reduce util and remove task allocation
                        new_node.util -= new_node.task_allocation[other_agt].util
                        delete!(new_node.task_allocation, other_agt)

                        # Generate new tree for other agent
                        setdiff!(new_node.considered_tasks[other_agt], tasks_to_ignore)

                        generate_search_tree!(solver.allocation, other_agt,
                                              new_node.considered_tasks[other_agt],
                                              success_prob_fn, util_val_fn, downtime)

                        tree = solver.allocation.agent_prop_set[other_agt].tree
                        dec_idx = get_next_attempt_idx(tree)

                        # If agent is assigned to a new task, update hlnode
                        if dec_idx != -1
                            dec_node = tree.nodes[dec_idx]

                            # Update util and allocation of new node
                            new_node.util += dec_node.util
                            new_node.task_allocation[other_agt] = (task=dec_node.task_name, util=dec_node.util)

                        end # if dec_idx
                    end # for other_agt

                    push!(solver.heap, new_node)
                    id += 1

                end # for agt in alloc_agents

                num_conflicts += 1
                solver.num_total_conflicts += 1

                if solver.num_total_conflicts > solver.allocation.conflict_threshold
                    @warn "High level conflict thresh exceeded!"
                    return P.task_allocation
                end

            end # if length > 1
        end # for (t,a) in task_to_agent

        if num_conflicts == 0 # This should be guaranteed to happen?
            @debug "No more conflicts!"
            return P.task_allocation
        end
    end # while ~isempty(heap)

    # Return empty allocation?
    @info "No coordinated allocation!"
    return Dict{String,TaskUtil}()

end # function coordinate_allocation

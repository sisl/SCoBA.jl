function cumul_grasp_prob(arm::Arm, ref_time::Float64, ie::InteractionEvent)

    true_ref_time = (ref_time < ie.timestamps[START]) ? ie.timestamps[START] : ref_time
    grasp_time = convert(Int64, floor(ie.timestamps[FINISH] - arm.downtime/2.0 - true_ref_time))

    return cumprob(arm.grasp_model, grasp_time)
end

# Utility of a successful pickup is just 1
pickup_util(ie::InteractionEvent) = 1.0


"""
For conveyor belt, the CG is a directed chain, so we do not need any higher level coordination.
A later arm only needs to consider the previous arm's policy tree to identify objects
that are either inevitable or potentially likely to enter its own workspace.
"""
function identify_inevitable_adversarial_entries(tree::SearchTree, workspace_objs::Set{String}, decision_idx::Int64)

    if decision_idx == -1
        return workspace_objs, ""
    end

    # Now only explore a subtree if there IS a decision node that will be taken
    # N.B - This will be the chosen option in most cases
    @assert typeof(tree.nodes[decision_idx]) == SCoBASolver.DecisionNode

    # Attempted box may fail - adversarial entry
    adversarial_entry = tree.nodes[decision_idx].task_name

    possibly_considered = Set{String}()

    # Traverse the subtree and note which objects are possibly considered
    # Then add the remainder to entries
    subtree_fringe = Deque{Int64}()
    push!(subtree_fringe, decision_idx)

    while !isempty(subtree_fringe)

        stf_top = popfirst!(subtree_fringe)

        if typeof(tree.nodes[stf_top]) == SCoBASolver.DecisionNode &&
            tree.nodes[stf_top].attempt == true
            push!(possibly_considered, tree.nodes[stf_top].task_name)
        end

        if haskey(tree.child_ids, stf_top)
            for stfc in tree.child_ids[stf_top]
                push!(subtree_fringe, stfc)
            end
        end
    end

    # Get all the objs not considered during subtree search
    not_considered = setdiff(workspace_objs, possibly_considered)

    return not_considered, adversarial_entry
end

"""
Run SCoBA on the conveyor belt domain.
"""
function scoba_conveyor!(server::ArmAllocationServer, cbelt::ConveyorBelt)

    first_arm = server.agent_ordering[1]

    if server.agent_prop_set[first_arm].obj_grabbed == ""
        # Create wrapper for success probability

        @debug first_arm
        @debug server.agent_prop_set[first_arm].workspace_objs

        success_prob_fn_first(rt, ie) = cumul_grasp_prob(server.agent_set[first_arm], rt, ie)
        generate_search_tree!(server, first_arm, server.agent_prop_set[first_arm].workspace_objs,
                             success_prob_fn_first, pickup_util,server.agent_set[first_arm].downtime)

        tree = server.agent_prop_set[first_arm].tree
        entries = Set{String}()
        adv_entry = ""
        if ~(isempty(tree))
            dec_idx = get_next_attempt_idx(tree)

            if dec_idx != -1
                dec_node = tree.nodes[dec_idx]
                @debug dec_node
                grasp_coords = get_box_point(cbelt, dec_node.task_name, dec_node.timeval)
                @assert grasp_coords[1] != Inf

                server.agent_task_allocation[first_arm] = (name=dec_node.task_name, time=dec_node.timeval)
            end

            entries, adv_entry = identify_inevitable_adversarial_entries(tree,
                                    server.agent_prop_set[first_arm].workspace_objs, dec_idx)
        end
    else
        # Need to populate entries and adv_entry
        entries = server.agent_prop_set[first_arm].workspace_objs
        adv_entry = ""
    end

    for arm_name in server.agent_ordering[2:end]

        if server.agent_prop_set[arm_name].obj_grabbed != ""
            continue
        end

        @debug arm_name
        # Objs to consider is the union of inevitable entries and workspace objs
        objs_to_consider = union(entries, server.agent_prop_set[arm_name].workspace_objs)
        @debug objs_to_consider

        # If the above yields nothing, THEN just consider adversarial_entry
        if isempty(objs_to_consider)
            if adv_entry != ""
                objs_to_consider = Set{String}()
                push!(objs_to_consider, adv_entry)
            end
        end

        if !isempty(objs_to_consider)

            success_prob_fn_rest(rt, ie) = cumul_grasp_prob(server.agent_set[arm_name], rt, ie)
            generate_search_tree!(server, arm_name, objs_to_consider,
                                 success_prob_fn_rest, pickup_util,
                                 server.agent_set[arm_name].downtime)
            tree = server.agent_prop_set[arm_name].tree
            dec_idx = get_next_attempt_idx(tree)

            if dec_idx != -1
                dec_node = tree.nodes[dec_idx]
                @debug dec_node
                grasp_coords = get_box_point(cbelt, dec_node.task_name, dec_node.timeval)
                @assert grasp_coords[1] != Inf

                server.agent_task_allocation[arm_name] = (name=dec_node.task_name, time=dec_node.timeval)
            end

            entries, adv_entry = identify_inevitable_adversarial_entries(tree,
                            server.agent_prop_set[arm_name].workspace_objs, dec_idx)
        end

        # If empty, then all relevant things would anyway be empty
    end # end for arm_name

    @debug server.agent_task_allocation
end

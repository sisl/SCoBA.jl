"""
Helper function that inserts a new decision node in the search tree,
corresponding to the new time window event that is observed.
success_prob_fn takes two arguments: (i) reference time (ii) interaction event and returns a single floating
point probability of success.
"""
function insert_decision_node!(tree::SearchTree, agent_name::String, ie::InteractionEvent,
                                parent_id::Int, ref_time::Float64, utilval::Float64, success_prob_fn::Function)

    # Start with end index to add nodes
    idx = length(tree.nodes)

    # If ref_time is less than the start of the event, set true ref time to event start
    true_ref_time = (ref_time < ie.timestamps[START]) ? ie.timestamps[START] : ref_time
    @assert true_ref_time <= ie.timestamps[FINISH] "ref_time is $(ref_time) and ie is $(ie.timestamps)"

    # N.B. Decision nodes should be initialized with 0 util
    # Decision node corresponding to attempt
    idx += 1
    attempt_idx = idx
    dnode1 = DecisionNode(agent_name=agent_name, task_name=ie.task_name, attempt=true, timeval=true_ref_time,
                          util=0.0, idx=attempt_idx, utilset=false)
    push!(tree.nodes, dnode1)
    tree.parent_id[attempt_idx] = parent_id

    # Decision node corresponding to not attempt
    # Add node idx to decision leaf
    idx += 1
    no_attempt_idx = idx
    dnode2 = DecisionNode(agent_name=agent_name, task_name=ie.task_name, attempt=false, timeval=true_ref_time,
                          util=0.0, idx=no_attempt_idx, utilset=false)
    push!(tree.nodes, dnode2)
    tree.parent_id[no_attempt_idx] = parent_id
    new_decision_leaf = no_attempt_idx

    # set children of parent id to decision node idxs
    tree.child_ids[parent_id] = [attempt_idx, no_attempt_idx]

    # Use function to get probability of success
    # Arguments of success_prob are the ref_time and the time window
    success_prob = success_prob_fn(ref_time, ie)

    # For now, set util to -Inf because leaf will set it
    idx += 1
    fail_idx = idx
    onode1 = OutcomeNode(agent_name=agent_name, task_name=ie.task_name, outcome=FINISH,
                         timeval=ie.timestamps[FINISH], probability=1.0 - success_prob, util=0.0, idx=fail_idx, utilset=false)
    push!(tree.nodes, onode1)
    tree.parent_id[fail_idx] = attempt_idx

    idx += 1
    succ_idx = idx
    onode2 = OutcomeNode(agent_name=agent_name, task_name=ie.task_name, outcome=SUCCESS,
                         timeval=ie.timestamps[SUCCESS], probability=success_prob, util=utilval, idx=succ_idx, utilset=false)
    push!(tree.nodes, onode2)
    tree.parent_id[succ_idx] = attempt_idx

    # Set child IDs of attempt_idx and new outcome leaves to succ and fail
    tree.child_ids[attempt_idx] = [succ_idx, fail_idx]
    new_outcome_leaves = [succ_idx, fail_idx]

    return new_outcome_leaves, new_decision_leaf
end


"""
Generate the allocation policy tree for a given agent and set of tasks to consider.
The util_val_fn returns the utility of completing an interaction event successfully.
"""
function generate_search_tree!(server::GenericAllocation{A,AP}, agent_name::String,
                               tasks_to_consider::Set{String}, success_prob_fn::Function, util_val_fn::Function,
                               downtime::Float64) where {A,AP}

    # Must be at least one task to consider
    if isempty(tasks_to_consider)
        empty!(server.agent_prop_set[agent_name].tree)
        return
    end

    # Assume that updated time windows have been generated
    # and that agent is currently not occupied with task already
    # Have outcome leaf idxs and outcome decision idxs (when NOT ATTEMPTED)
    # If a new event subsumes all outcome leaf idxs it should NOT be added to not-attempted OUTCOME DECISION IDX
    tree = server.agent_prop_set[agent_name].tree
    interaction_events = server.agent_prop_set[agent_name].interaction_events

    outcome_leaf_idxs = Set{Int}()
    decision_leaf_idxs = Set{Int}()

    first_ie_stamp = Inf

    considered_ie = [ie for ie in interaction_events if ie.task_name in tasks_to_consider]

    num_to_consider = (length(considered_ie) > server.max_tasks_to_consider) ? server.max_tasks_to_consider : length(considered_ie)

    # IMP - The first interaction event timestamp may be shared by multiple
    # events; in that case, should consider all at root
    for ie in considered_ie[1:num_to_consider]

        ie_stamp = ie.timestamps[START]

        if ie_stamp < first_ie_stamp

            utilval = util_val_fn(ie)
            (new_outcome_leaves, new_decision_leaf) = insert_decision_node!(tree, agent_name, ie,
                                                                            0, server.current_time,
                                                                            utilval, success_prob_fn)
            for idx in new_outcome_leaves
                push!(outcome_leaf_idxs, idx)
            end
            push!(decision_leaf_idxs, new_decision_leaf)

            first_ie_stamp = ie_stamp

        else

            # Now iterate through outcome leaf nodes and compare
            new_outcome_leaf_idxs = Set{Int}()
            new_decision_leaf_idxs = Set{Int}()
            outcome_leaves_to_rm = Set{Int}()
            decision_leaves_to_rm = Set{Int}()

            for ol_idx in outcome_leaf_idxs
                onode = tree.nodes[ol_idx]
                @assert typeof(onode) == OutcomeNode # sanity check

                if onode.timeval <= ie.timestamps[START]
                    # Dominated - continue
                    continue
                end

                if onode.timeval >= ie.timestamps[FINISH] - downtime/2.0
                    continue
                end

                # Last case - new IE begins before outcome and ends after outcome
                # now add decision node to try and its two outcomes
                utilval = util_val_fn(ie)
                (temp_outcome_idxs, temp_dec_idx) = insert_decision_node!(tree, agent_name, ie,
                                                                          ol_idx, onode.timeval,
                                                                          utilval, success_prob_fn)
                push!(outcome_leaves_to_rm, ol_idx)

                for to_idx in temp_outcome_idxs
                    push!(new_outcome_leaf_idxs, to_idx)
                end
                push!(new_decision_leaf_idxs, temp_dec_idx)
            end

            # IMP - Not attempting is DIFFERENT from failing immediately, because it affects the dominance principle
            # Current no-attempt decision nodes MUST have had their attempt counterparts added already
            # So if new event is dominated, then it should not follow no-attempt leaf nodes, EVEN if otherwise valid
            @assert isempty(new_decision_leaf_idxs) == isempty(new_outcome_leaf_idxs)

            # Otherwise, loop through decision_leaf_idxs and do the same thing
            for dec_idx in decision_leaf_idxs
                dnode = tree.nodes[dec_idx]

                @assert dnode.attempt == false

                # We know that event is non-dominated by outcomes
                # and we did not jump forward in time, so do not need to do earlier checks
                if dnode.timeval >= ie.timestamps[FINISH] - downtime/2.0
                    continue
                end

                utilval = util_val_fn(ie)
                (temp_outcome_idxs, temp_dec_idx) = insert_decision_node!(tree, agent_name, ie,
                                                                        dec_idx, dnode.timeval,
                                                                        utilval, success_prob_fn)
                push!(decision_leaves_to_rm, dec_idx)

                for to_idx in temp_outcome_idxs
                    push!(new_outcome_leaf_idxs, to_idx)
                end
                push!(new_decision_leaf_idxs, temp_dec_idx)
            end

            # Finally, remove old leaves and add new ones
            setdiff!(outcome_leaf_idxs, outcome_leaves_to_rm)
            union!(outcome_leaf_idxs, new_outcome_leaf_idxs)

            setdiff!(decision_leaf_idxs, decision_leaves_to_rm)
            union!(decision_leaf_idxs, new_decision_leaf_idxs)
        end
    end

    # Now go up from leaves, assigning utilities to parent nodes
    # First assign utilities to actual leaves based on outcomes
    # Then, add parents to the current fringe
    rev_fringe_idxs = Set{Int64}()

    for ol_idx in outcome_leaf_idxs

        # Add to util of parent node
        ol_node = tree.nodes[ol_idx]
        ol_node.utilset = true

        push!(rev_fringe_idxs, ol_idx)
    end

    # Do the same for decision leaves, but now
    # Since there are only positive or zero utilities
    # can just do max with self
    for dec_idx in decision_leaf_idxs

        # this must be a non-attempt decision so util zero
        dec_node = tree.nodes[dec_idx]
        @assert dec_node.attempt == false
        dec_node.util = 0.0 # For good measure
        dec_node.utilset = true
        push!(rev_fringe_idxs, dec_idx)
    end

    # IMP - REV FRINGE IDXS can have their own children in rev_fringe_idxs too!
    # So keep adding and removing until empty
    # N.B - This is for setting PARENT UTIL
    while !isempty(rev_fringe_idxs)

        to_rm = Set{Int64}()
        to_add = Set{Int64}()


        for rfi in rev_fringe_idxs

            # If child of root, should remove (unless still needed)
            if tree.parent_id[rfi] == 0
                push!(to_rm, rfi)
                continue
            end

            node = tree.nodes[rfi]
            @assert node.utilset == true "Node in rev fringe does NOT have util set",node

            # If any sibling not set, continue WHILE keeping in rev_fringe_idxs
            par_id = tree.parent_id[rfi]
            par_node = tree.nodes[par_id]

            siblingset = true
            for child in tree.child_ids[par_id]
                if ~(child in rev_fringe_idxs)
                    siblingset = false
                    break
                end
            end
            if siblingset == false
                continue
            end


            # Do what is necessary for parent and add to rfi
            # Parent is decision node
            if typeof(node) == OutcomeNode
                par_node.util = par_node.util + node.probability*node.util
            else
                # Node type is decision
                # If parent is decision, just compare child decisions and choose max
                # If parent is outcome, add to exp. util if outcome success
                temp_node_util = node.util
                if typeof(par_node) == OutcomeNode && par_node.outcome == SUCCESS
                    temp_node_util = 1.0 + node.util
                end
                par_node.util = (temp_node_util > par_node.util) ? temp_node_util : par_node.util

            end

            push!(to_add, par_id)
            push!(to_rm, rfi)
        end

        for par_id in to_add
            tree.nodes[par_id].utilset = true
        end

        # First remove and then add
        union!(rev_fringe_idxs, to_add)
        setdiff!(rev_fringe_idxs, to_rm)
    end
end


# When there are multiple simultaneous objects, then it is not enough
# that the 'first' object is not attempted
function get_next_attempt_idx(tree::SearchTree)

    if isempty(tree)
        return -1
    end

    # Find the first node at which ATTEMPTING is the decision
    # Until that point, put all objs in entries
    dec_fringe = Deque{Int64}()
    push!(dec_fringe, 0)

    subtree_root_dec = -1

    while !isempty(dec_fringe)

        dec_top = popfirst!(dec_fringe)

        top_children = tree.child_ids[dec_top]

        @assert length(top_children) == 2 # TODO : Check this is valid
        attempt_idx = (tree.nodes[top_children[1]].attempt == true) ? top_children[1] : top_children[2]
        no_attempt_idx = top_children[1] + top_children[2] - attempt_idx

        if tree.nodes[no_attempt_idx].util > tree.nodes[attempt_idx].util
            push!(dec_fringe, no_attempt_idx)
        else
            subtree_root_dec = attempt_idx
        end
    end

    return subtree_root_dec
end

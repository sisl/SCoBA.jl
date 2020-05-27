"""
Implements the Hungarian baseline from the paper.
"""
function expected_hungarian!(server::ArmAllocationServer, cbelt::ConveyorBelt)

    # Empty the current object-arm assignment record
    empty!(server.agent_task_allocation)

    # Set up matrix for hungarian
    # If out of reach, assign infinite cost
    # First, get vectors for active arms and boxes
    active_arm_names = Vector{String}(undef, 0)
    for arm_name in collect(keys(server.agent_set))
        if server.agent_prop_set[arm_name].obj_grabbed == "" # Not currently assigned
            push!(active_arm_names, arm_name)
        end
    end

    # Vector of box names and current x-coordinates
    box_xpos_vector = Vector{Tuple{String, Float64}}(undef, 0)
    for (name, route) in cbelt.box_routes
        push!(box_xpos_vector, (name, route[1].coordinates[1]))
    end

    # Only choose the n rightmost boxes if number of boxes > arms
    narms = length(active_arm_names)
    nboxes = length(box_xpos_vector)

    if narms == 0 || nboxes == 0
        @info "Either no active arms or no active boxes!"
        return
    end

    # Sort in descending order of position
    sort!(box_xpos_vector, by = x->x[2], rev=true)
    @debug box_xpos_vector

    # Iterate over box_xpos_vector until every arm can grasp at least one box
    boxes_to_consider = String[]
    arms_to_cover = Set(active_arm_names)
    arms_covered = Set{String}()

    for boxpos in box_xpos_vector

        # Consider box since going in descending order of position - most urgent first
        push!(boxes_to_consider, boxpos[1])

        # Look at what arms can potentially grasp box
        for arm_name in active_arm_names
            if boxpos[2] < server.agent_set[arm_name].reach[2]
                push!(arms_covered, arm_name)
            end
        end

        # We're done if we have considered at least narms boxes and all arms can grasp at least one
        if length(boxes_to_consider) >= narms && arms_covered == arms_to_cover
            break
        end
    end

    # Now create cost matrix of boxes vs arms
    # Each entry will have prob of failure
    # Don't need tp dict - will attempt first possible grasp
    nboxes = length(boxes_to_consider)
    cost_matrix = ones(narms, nboxes)


    # From the boxes to consider, compute the grasp failure prob for each active arm
    # Ignore (i.e., keep at 1.) if not reachable
    for (i, arm_name) in enumerate(active_arm_names)

        this_arm = server.agent_set[arm_name]

        # Iterate over boxes and consider time windows with given arm
        for (j, box_name) in enumerate(boxes_to_consider)

            if haskey(server.agent_task_windows, (arm_name, box_name))
                window = server.agent_task_windows[(arm_name, box_name)]

                # Assign weight to probability of grasp failure
                ref_time = max(server.current_time + 1.0, window[1])
                grasp_time = convert(Int64, floor(window[2] - server.agent_set[arm_name].downtime/2.0 - ref_time))
                fail_prob = 1.0 - cumprob(this_arm.grasp_model, grasp_time)

                cost_matrix[i, j] = fail_prob
            end
        end
    end

    @debug cost_matrix

    if minimum(cost_matrix) == 1.0
        @warn "All matchings are failures"
        return
    end

    # Run Hungarian algorithm
    assignment, cost = hungarian(cost_matrix)
    @debug assignment

    # From assignment, updated arm_obj_assignment with the timestamp
    # and the corresponding grasp location, using get_box_point
    for (i, assgn) in enumerate(assignment)

        if assgn != 0

            # Then arm reduced_row_orig[i] is assigned to box reduced_col_orig[assign]
            true_arm_name = active_arm_names[i]
            true_box_name = boxes_to_consider[assgn]
            @debug (true_arm_name, true_box_name)

            # Only assign an arm to object if it actually has a valid time window
            if haskey(server.agent_task_windows, (true_arm_name, true_box_name))
                window = server.agent_task_windows[(true_arm_name, true_box_name)]

                ref_time = max(server.current_time + 1.0, window[1])

                server.agent_task_allocation[true_arm_name] = (name = true_box_name, time =ref_time)
            else
                @debug "Arm $(true_arm_name) goes unassigned this time"
            end
        end
    end

    @debug server.agent_task_allocation
end


"""
Implements the EDD baseline. Each arm is assigned to the rightmost object in its current workspace.
"""
function earliest_due_date!(server::ArmAllocationServer, cbelt::ConveyorBelt)

    empty!(server.agent_task_allocation)

    active_arm_names = Vector{String}(undef, 0)
    for arm_name in collect(keys(server.agent_set))
        if server.agent_prop_set[arm_name].obj_grabbed == "" # Not currently assigned
            push!(active_arm_names, arm_name)
        end
    end

    # Vector of box names and next x-coordinates for each box
    box_xpos_vector = Vector{Tuple{String, Float64}}(undef, 0)
    for (name, route) in cbelt.box_routes
        if length(route) >= 2
            push!(box_xpos_vector, (name, route[2].coordinates[1]))
        end
    end

    narms = length(active_arm_names)
    nboxes = length(box_xpos_vector)

    if narms == 0 || nboxes == 0
        @info "Either no active arms or no active boxes!"
        return
    end

    sort!(box_xpos_vector, by = x->x[2], rev=true)

    boxes_covered = Set{String}()

    # For each arm, look at rightmost object in its workspace that it can possibly grasp
    for arm_name in active_arm_names
        for (box_name, pos) in box_xpos_vector

            # If below condition is not satisfied by any box, arm stays unassigned
            if ~(box_name in boxes_covered) && pos < server.agent_set[arm_name].reach[2]

                window = server.agent_task_windows[(arm_name, box_name)] # Should not give err by construction
                ref_time = max(server.current_time + 1.0, window[1])

                grasp_coords = get_box_point(cbelt, box_name, ref_time)
                server.agent_task_allocation[arm_name] = (name = box_name, time = ref_time)
                push!(boxes_covered, box_name)
            end
        end
    end

    @debug server.agent_task_allocation
end



## POMDPs.MDP definition for MCTS and Q-Learning
const BeltState = Vector{Int64}
@with_kw struct ArmState
    obj_grabbed::Bool   = false
end
struct ConvBeltSystemState
    belt_disc::BeltState
    arm_states::Vector{ArmState}
end

"""
Simple discrete-time MDP formulation of the conveyor belt problem.
"""
@with_kw mutable struct ConvBeltMDP <: POMDPs.MDP{ConvBeltSystemState,Vector{Int64}}
    cbelt::ConveyorBelt
    server::ArmAllocationServer
    belt_disc_factor::Float64 # Ideally 1/bdf should give an integer
    n_belt_slots::Int64
    speed_skip_factor::Int64
    per_arm_slots::Int64
    horizon::Int64
    max_per_slot::Int64
    states::Vector{ConvBeltSystemState}
end

"""
Instantiates the MDP by discretizing the belt according to the chosen resolution
"""
function create_convbelt_mdp(cbelt::ConveyorBelt, server::ArmAllocationServer,
                             disc_factor::Float64, horizon::Int64,
                             max_per_slot::Int64=typemax(Int64))

    speed_skip_factor = max(1, convert(Int64, floor(cbelt.curr_speed/disc_factor)))
    arm_reach = server.agent_set[server.agent_ordering[1]].reach # Assuming same for all
    per_arm_slots = convert(Int64, ceil((arm_reach[2]-arm_reach[1])/disc_factor))

    n_belt_slots = convert(Int64, ceil(1.0/disc_factor))

    # Create states IF explicit
    statevect = ConvBeltSystemState[] # Empty by default
    if max_per_slot < typemax(Int64)
        belt_disc_vals = [0:max_per_slot for i = 1:n_belt_slots]
        belt_disc_vector_vals = [collect(Iterators.flatten(a)) for a in Iterators.product(belt_disc_vals...)]
        arm_state_vals = [(false,true) for i = 1:length(server.agent_ordering)]
        arm_state_vector_vals = [collect(Iterators.flatten(a)) for a in Iterators.product(arm_state_vals...)]
        statevect = [ConvBeltSystemState(bdv, [ArmState(a) for a in asv]) for bdv in belt_disc_vector_vals
                    for asv in arm_state_vector_vals]
    end

    return ConvBeltMDP(cbelt, server, disc_factor, n_belt_slots,
                           speed_skip_factor, per_arm_slots, horizon,
                           max_per_slot, statevect)
end

"""
Returns the ConvBeltSystemState struct that represents the current state of the MDP.
"""
function get_current_belt_mdp_state(mdp::ConvBeltMDP, rng::AbstractRNG=Random.GLOBAL_RNG)

    # Populate belt
    belt_disc = zeros(Int64, mdp.n_belt_slots)

    # Obtain box current position, find discrete idx on belt and increment
    for (box_name, route) in mdp.cbelt.box_routes
        curr_pos = route[1].coordinates[1]
        box_belt_idx = convert(Int64, ceil(curr_pos/mdp.belt_disc_factor))

        # NOTE: Truncating here. This is ALL we have to do
        belt_disc[box_belt_idx] = min(belt_disc[box_belt_idx] + 1, mdp.max_per_slot)
    end

    # Now get arms
    arm_states = Vector{ArmState}(undef, length(mdp.server.agent_ordering))

    for (ia, arm_name) in enumerate(mdp.server.agent_ordering)

        arm_props = mdp.server.agent_prop_set[arm_name]

        if arm_props.obj_grabbed != ""
            arm_states[ia] = ArmState(true)
        else
            arm_states[ia] = ArmState(false)
        end
    end

    # Finally return state with current time
    return ConvBeltSystemState(belt_disc, arm_states)
end


POMDPs.discount(mdp::ConvBeltMDP) = 1.0
POMDPs.isterminal(mdp::ConvBeltMDP, s::ConvBeltSystemState) = (mdp.server.current_time >= mdp.horizon) # TODO: No notion of terminal state

function POMDPs.actions(mdp::ConvBeltMDP, s::ConvBeltSystemState)

    arm_actions = UnitRange{Int64}[]
    for as in s.arm_states
        if as.obj_grabbed == true
            push!(arm_actions, 0:0) # No-op
        else
            push!(arm_actions, 0:mdp.per_arm_slots) # 0 for home, 1:belt_slots for rest
        end
    end

    return [collect(Iterators.flatten(a)) for a in Iterators.product(arm_actions...)]
end

function POMDPs.gen(mdp::ConvBeltMDP, s::ConvBeltSystemState, a::Vector{Int64}, rng::AbstractRNG)

    # First set new arm_states
    new_arm_states = ArmState[]
    box_positions_grabbed = Int64[]

    # go along belt disc and copy to next spot based on speed skip factor
    new_belt_disc = zeros(Int64, mdp.n_belt_slots)
    for idx = mdp.n_belt_slots-mdp.speed_skip_factor:-1:1
        new_belt_disc[idx + mdp.speed_skip_factor] = s.belt_disc[idx]
    end


    for (as, aa, arm_name) in zip(s.arm_states, a, mdp.server.agent_ordering)

        # Can be 0 only if obj was grabbed in s, so deposit to bin and free up
        if aa == 0
            push!(new_arm_states, ArmState(false))
        else
            # Now test out grab
            arm_facts = mdp.server.agent_set[arm_name]
            arm_slot = convert(Int64, floor(arm_facts.reach[1]/mdp.belt_disc_factor)) + aa

            # Grab if box there and rand(rng)....
            if (rand(rng) < arm_facts.grasp_model.bernoulli_prob &&
                new_belt_disc[arm_slot] >= 1)
                # Successful grab!!
                push!(box_positions_grabbed, arm_slot)
                push!(new_arm_states, ArmState(true))
            else
                # Unsuccessful or no box there?
                push!(new_arm_states, ArmState(false))
            end
        end # aa == -1
    end

    # decrement box counts for lifted slots
    for bp in box_positions_grabbed
        new_belt_disc[bp] -= 1
    end

    return (sp=ConvBeltSystemState(new_belt_disc, new_arm_states), r = length(box_positions_grabbed))
end

"""
The rollout policy for the conveyor belt uses the EDD strategy.
"""
struct EDDPolicy <: Policy
    mdp::ConvBeltMDP
end

function POMDPs.action(pol::EDDPolicy, s::ConvBeltSystemState)

    mdp = pol.mdp

    joint_action = Int64[]
    for (as, arm_name) in zip(s.arm_states, mdp.server.agent_ordering)
        if as.obj_grabbed == true
            push!(joint_action, 0) # No-op
        else
            # Find the furthest along slot that will still be valid at next step
            arm_facts = mdp.server.agent_set[arm_name]
            last_arm_slot = convert(Int64, ceil(arm_facts.reach[2]/mdp.belt_disc_factor))
            first_arm_slot = convert(Int64, floor(arm_facts.reach[1]/mdp.belt_disc_factor)) + 1

            aa = 0 # Default value
            for idx = last_arm_slot-mdp.speed_skip_factor:-1:first_arm_slot
                if s.belt_disc[idx + mdp.speed_skip_factor] >= 1
                    aa = idx + mdp.speed_skip_factor - first_arm_slot
                    break
                end
            end

            push!(joint_action, aa)
        end
    end

    return joint_action
end


## Other requirements for QLearning
POMDPs.initialstate(mdp::ConvBeltMDP, rng::AbstractRNG) = get_current_belt_mdp_state(mdp, rng)

# Not state-dependent actions
function POMDPs.actions(mdp::ConvBeltMDP)
    arm_ac_range = [0:mdp.per_arm_slots for i = 1:length(mdp.server.agent_ordering)]
    return [collect(Iterators.flatten(a)) for a in Iterators.product(arm_ac_range...)]
end


function POMDPs.actionindex(mdp::ConvBeltMDP, action::Vector{Int64})

    interm_idxs = [ (a+1) for a in action]
    ac_tup = Tuple(1:mdp.per_arm_slots+1 for i = 1:length(mdp.server.agent_ordering))
    return LinearIndices(ac_tup)[interm_idxs...]

end


POMDPs.states(mdp::ConvBeltMDP) = mdp.states

Base.isequal(s1::ConvBeltSystemState, s2::ConvBeltSystemState) = (s1.belt_disc == s2.belt_disc) &&
                            (s1.arm_states == s2.arm_states)

POMDPs.stateindex(mdp::ConvBeltMDP, s::ConvBeltSystemState) = findfirst(isequal(s), mdp.states)



"""
Q-Learning for the conveyor belt uses IQL for each agent.
"""
struct SingleArmState
    belt_disc::Vector{Int64}
    obj_grabbed::Bool
end


@with_kw mutable struct ConvBeltSingleArmMDP <: POMDPs.MDP{SingleArmState,Int64}
    cbelt::ConveyorBelt
    server::ArmAllocationServer
    belt_disc_factor::Float64 # Ideally 1/bdf should give an integer
    speed_skip_factor::Int64
    per_arm_slots::Int64
    max_per_slot::Int64
    states::Vector{SingleArmState}
end


function create_convbelt_singlearm_mdp(cbelt::ConveyorBelt, server::ArmAllocationServer,
                                       disc_factor::Float64,
                                       max_per_slot::Int64=typemax(Int64))

    speed_skip_factor = max(1, convert(Int64, floor(cbelt.curr_speed/disc_factor)))
    arm_reach = server.agent_set[server.agent_ordering[1]].reach # Assuming same for all
    per_arm_slots = convert(Int64, ceil((arm_reach[2]-arm_reach[1])/disc_factor))


    # Create states IF explicit
    statevect = ConvBeltSystemState[] # Empty by default
    if max_per_slot < typemax(Int64)
        belt_disc_vals = [0:max_per_slot for i = 1:per_arm_slots+1]
        belt_disc_vector_vals = [collect(Iterators.flatten(a)) for a in Iterators.product(belt_disc_vals...)]
        arm_state_vals = [false, true]
        statevect = [SingleArmState(bdv, asv) for bdv in belt_disc_vector_vals for asv in arm_state_vals]
    end

    return ConvBeltSingleArmMDP(cbelt, server, disc_factor,
                                speed_skip_factor, per_arm_slots,
                                max_per_slot, statevect)
end

POMDPs.initialstate(mdp::ConvBeltSingleArmMDP, rng::AbstractRNG) = SingleArmState([rand(rng, 1:mdp.max_per_slot) for i in 1:mdp.per_arm_slots+1], false)

POMDPs.discount(mdp::ConvBeltSingleArmMDP) = 1.0


# Need a gen for the MDP and then need a true outer gen conveyor_mcts?
# that is run before calling propagate belt and propagate arm
# It looks at the action obtained from MCTS and implements it in the actual outer env
function POMDPs.gen(mdp::ConvBeltSingleArmMDP, s::SingleArmState, a::Int64, rng::AbstractRNG)


    # go along belt disc and copy to next spot based on speed skip factor
    new_belt_disc = zeros(Int64, mdp.per_arm_slots+1)
    for idx = mdp.per_arm_slots+1-mdp.speed_skip_factor:-1:1
        new_belt_disc[idx + mdp.speed_skip_factor] = s.belt_disc[idx]
    end

    obj_grabbed = false
    r = 0
    if s.obj_grabbed == false # Only then _could_ it be true

        any_arm = mdp.server.agent_set[mdp.server.agent_ordering[1]]

        if rand(rng) < any_arm.grasp_model.bernoulli_prob && new_belt_disc[a+1] >= 1
            r = 1
            obj_grabbed = true
            new_belt_disc[a+1] -= 1
        end
    end

    return (sp=SingleArmState(new_belt_disc, obj_grabbed), r = r)
end

# Not state-dependent actions
POMDPs.actions(mdp::ConvBeltSingleArmMDP) = collect(1:mdp.per_arm_slots)
POMDPs.actionindex(mdp::ConvBeltSingleArmMDP, action::Int64) = action
POMDPs.states(mdp::ConvBeltSingleArmMDP) = mdp.states

Base.isequal(s1::SingleArmState, s2::SingleArmState) = (s1.belt_disc == s2.belt_disc) && (s1.obj_grabbed == s2.obj_grabbed)

POMDPs.stateindex(mdp::ConvBeltSingleArmMDP, s::SingleArmState) = findfirst(isequal(s), mdp.states)


function get_singlearm_mdp_state(mdp::ConvBeltSingleArmMDP, agent_idx::Int64)

    per_arm_disc = zeros(Int64, mdp.per_arm_slots+1)

    arm_facts = mdp.server.agent_set[mdp.server.agent_ordering[agent_idx]]

    start_point = arm_facts.reach[1] - mdp.belt_disc_factor

    for (box_name, route) in mdp.cbelt.box_routes
        curr_pos = route[1].coordinates[1]

        if curr_pos > start_point && curr_pos < arm_facts.reach[2]
            arm_slot_idx = convert(Int64, ceil((curr_pos-start_point)/mdp.belt_disc_factor))
            per_arm_disc[arm_slot_idx] = min(per_arm_disc[arm_slot_idx] + 1, mdp.max_per_slot)
        end
    end

    arm_props = mdp.server.agent_prop_set[mdp.server.agent_ordering[agent_idx]]
    obj_grabbed = ~(isempty(arm_props.obj_grabbed))

    return SingleArmState(per_arm_disc, obj_grabbed)
end

function update_server_with_mdp_action!(mdp::Union{ConvBeltMDP,ConvBeltSingleArmMDP},
                                       action::Vector{Int64})

    next_time = convert(Int64, mdp.server.current_time + 1)

    # Iterate over action and
    for (agent_action, arm_name) in zip(action, mdp.server.agent_ordering)
        arm_facts = mdp.server.agent_set[arm_name]
        arm_props = mdp.server.agent_prop_set[arm_name]

        if agent_action >= 1 && arm_props.obj_grabbed == ""
            arm_slot = convert(Int64, floor(arm_facts.reach[1]/mdp.belt_disc_factor)) + agent_action

            # Iterate over box routes and choose slot
            any_box = false
            arm_assgn = (name="", time=Inf)
            for (box_name, route) in mdp.cbelt.box_routes
                if length(route) >= 2
                    next_pos = route[2].coordinates[1]
                    box_belt_idx = convert(Int64, ceil(next_pos/mdp.belt_disc_factor))
                    if box_belt_idx == arm_slot && next_pos >= arm_facts.reach[1] && next_pos <= arm_facts.reach[2]
                        any_box = true
                        arm_assgn = (name = box_name, time = next_time)
                        break
                    end # box_belt_idx
                end # length(route)
            end

            if any_box
                mdp.server.agent_task_allocation[arm_name] = arm_assgn
            end
        end
    end
end

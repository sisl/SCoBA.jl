function count_terminal_grabbed_boxes!(cbelt::ConveyorBelt, server::ArmAllocationServer)

    for (arm_name, arm_props) in server.agent_prop_set
        if arm_props.obj_grabbed != ""
            cbelt.deposited_boxes += 1
        end
    end
end


### Conveyor Belt functions ###

"""
Simulate the next time-step for the conveyor belt. New boxes are added by the mirror generator
in the top-level script.
"""
function propagate_belt!(cbelt::ConveyorBelt, server::ArmAllocationServer)

    diagnostic_info = ""

    to_delete = Vector{String}(undef,0)

    # First push forward boxes according to route
    for (box_name, route) in cbelt.box_routes

        # Remove first entry of route
        popfirst!(route)

        # If at end of belt, remove
        if length(route) == 0 || route[1].coordinates[1] > cbelt.belt_end
            @info box_name," was not picked up!"
            diagnostic_info = string(box_name," lost!")
            push!(to_delete, box_name)
            cbelt.lost_boxes += 1

            # Delete from last arm bookkeeping
            delete!(server.agent_task_windows, (server.agent_ordering[end], box_name))
            delete!(server.agent_prop_set[server.agent_ordering[end]].workspace_objs, box_name)

            if haskey(server.agent_task_allocation, server.agent_ordering[end]) &&
                server.agent_task_allocation[server.agent_ordering[end]].name == box_name
                delete!(server.agent_task_allocation, server.agent_ordering[end])
            end
        end
    end

    # Now delete boxes to delete
    for box_to_del in to_delete
        remove_box!(cbelt, box_to_del)
    end

    # Update time
    cbelt.current_time += 1.0

    # IMP: New boxes added in top-level script

    return diagnostic_info
end


function remove_box!(cbelt::ConveyorBelt, box_name::String)
    delete!(cbelt.box_routes, box_name)
    cbelt.num_active_boxes -= 1
end

"""
Prior to the next allocation, update the arm-object task windows.
"""
function update_time_windows!(cbelt::ConveyorBelt, server::ArmAllocationServer)

    @assert cbelt.current_time == server.current_time

    # For root arm, all objs before start are also to be considered
    for (box_name, route) in cbelt.box_routes

        first_route_tp = route[1]
        if first_route_tp.coordinates[1] <= server.agent_set[server.agent_ordering[1]].reach[1]
            push!(server.agent_prop_set[server.agent_ordering[1]].workspace_objs, box_name)
        end
    end

    for (arm_name, arm) in server.agent_set

        agent_prop_set = server.agent_prop_set[arm_name]
        interaction_events = agent_prop_set.interaction_events

        # Get rid of stale interaction_events
        # AND update the start time for ongoing interaction events
        interaction_events = Vector{InteractionEvent}(undef, 0)
        for ie in agent_prop_set.interaction_events
            if ie.timestamps[FINISH] >= cbelt.current_time+1.0
                if ie.timestamps[START] < cbelt.current_time+1.0
                    ie.timestamps[START] = cbelt.current_time+1.0
                end
                push!(interaction_events, ie)
            end
        end

        # Iterate over boxes and insert in dict if valid
        # TODO - Change to handle update to existing routes
        for (box_name, route) in cbelt.box_routes

            # First check if boxes in or out of workspace already
            first_route_tp = route[1]
            @assert first_route_tp.timestamp >= server.current_time+1.0

            if first_route_tp.coordinates[1] >= arm.reach[2]
                # Out of workspace
                if box_name in agent_prop_set.workspace_objs
                    pop!(agent_prop_set.workspace_objs, box_name)
                    delete!(server.agent_task_windows, (arm_name, box_name))
                end
            else
                # Insert in workspace_objs IF the current position is within the workspace
                if first_route_tp.coordinates[1] >= arm.reach[1] && first_route_tp.coordinates[1] < arm.reach[2]
                    push!(agent_prop_set.workspace_objs, box_name)
                end

                # Regardless, update the interaction events
                if haskey(server.agent_task_windows, (arm_name, box_name)) == false

                    # now compute time windows
                    min_time = Inf
                    max_time = -Inf

                    for tpoint in route

                        tm = tpoint.timestamp
                        coords = tpoint.coordinates

                        if coords[1] >= arm.reach[1] && coords[1] <= arm.reach[2]
                            # Within window
                            if tm < min_time
                                min_time = tm
                            end
                            if tm > max_time
                                max_time = tm
                            end
                        end
                    end

                    # Assign windows and update temporal event list
                    if min_time != Inf && max_time != -Inf

                        server.agent_task_windows[(arm_name, box_name)] = [min_time, max_time+arm.downtime/2.0, max_time + arm.downtime]
                        ie_timestamps = Dict(START=>min_time, FINISH=>max_time+arm.downtime/2.0, SUCCESS=>max_time + arm.downtime)
                        push!(interaction_events, InteractionEvent(arm_name, box_name, ie_timestamps))

                    end
                end
            end
        end

        # Re-sort the temporal event list
        sort!(interaction_events, by=x->x.timestamps[START])
        agent_prop_set.interaction_events = interaction_events

    end
end

"""
Given the starting x-coordinate of the box and belt properties (speed etc), map out the
trajectory of the box over the planning horizon.
"""
function generate_route_from_init_point(cbelt::ConveyorBelt, box_start_time::Float64, box_start_point::Float64)

    box_route = Vector{TimePoint}(undef, 0)

    while box_start_point < cbelt.belt_end

        push!(box_route, TimePoint(box_start_time, [box_start_point, 0.5]))

        box_start_point += cbelt.curr_speed
        box_start_time += 1.0
    end

    return box_route
end

function add_box_to_belt!(cbelt::ConveyorBelt, box_start_point::Float64)

    cbelt.num_active_boxes += 1
    cbelt.num_total_boxes += 1
    new_box_name = string("box",cbelt.num_total_boxes)

    box_route = generate_route_from_init_point(cbelt, cbelt.current_time+1.0, box_start_point)
    cbelt.box_routes[new_box_name] = box_route

end


"""
Get box x-coordinate that is closest to the reference time.
"""
function get_box_point(cbelt::ConveyorBelt, box_name::String, ref_time::Float64)


    box_route = cbelt.box_routes[box_name]

    for box_tp in box_route
        if abs(box_tp.timestamp - ref_time) < 0.5
            return box_tp.coordinates
        end
    end

    # Shouldn't reach this point
    return [Inf, Inf]
end


"""
Sets up the mirrored feasible task sequence generator by reflecting the
simulator about the Y-axis.
"""
function setup_mirror_gen!(cbelt::ConveyorBelt)

    # Copy over everything while mirroring the x-axes of the positions
    n_arms = length(cbelt.mirror_gen.arm_set)

    mirror_arm_set = Vector{Arm}(undef, n_arms)
    cbelt.mirror_gen.arm_stages = Vector{GENSTAGE}(undef, n_arms)

    for (i, arm) in enumerate(cbelt.mirror_gen.arm_set)

        new_arm = Arm(base_pos = [-arm.base_pos[1], arm.base_pos[2]],
                      bin_pos = [-arm.bin_pos[1], arm.bin_pos[2]],
                      reach = [-arm.reach[1], -arm.reach[2]],
                      orig_pos = [-arm.orig_pos[1], arm.orig_pos[2]],
                      grasp_model = arm.grasp_model,
                      downtime = arm.downtime
                      )

        mirror_arm_set[i] = new_arm
        cbelt.mirror_gen.arm_stages[i] = PREDROP
    end

    cbelt.mirror_gen.arm_set = mirror_arm_set
end

"""
Simulate one more timestep of the mirrored feasible sequence generator.
"""
function mirror_gen_step!(cbelt::ConveyorBelt, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    mirror_gen = cbelt.mirror_gen

    # First propagate boxes forward with speed and add to belt the new boxes
    keys_to_delete = Set{Int64}()

    for (i, pos) in mirror_gen.box_pos

        new_pos = pos + cbelt.curr_speed
        if new_pos > 0.0
            push!(keys_to_delete, i)
            add_box_to_belt!(cbelt, -pos)
        else
            mirror_gen.box_pos[i] = new_pos
        end
    end

    for kd in keys_to_delete
        delete!(mirror_gen.box_pos, kd)
    end

    # Now propagate arms and generate boxes if PREDROP
    for (i, arm) in enumerate(mirror_gen.arm_set)

        stage = mirror_gen.arm_stages[i]

        if stage == PREDROP::GENSTAGE

            # Drop box with NEW BOX PROB
            if rand(rng) <= cbelt.new_box_prob

                # Generate a new box position at random
                dist = Distributions.Uniform(arm.reach[2], arm.reach[1])
                new_pos = rand(rng, dist)

                mirror_gen.gen_box_id += 1
                mirror_gen.box_pos[mirror_gen.gen_box_id] = new_pos

                new_stage = POSTDROP::GENSTAGE

            else
                new_stage = stage
            end

        elseif stage == POSTDROP::GENSTAGE
            new_stage = ATBIN::GENSTAGE
        else
            new_stage = PREDROP
        end

        mirror_gen.arm_stages[i] = new_stage
    end
    @debug mirror_gen.arm_stages
    @debug mirror_gen.box_pos
end



"""
A simplified model of the trajectory planning and execution problem.
The arm can move from any place in its workspace to another in unit time.
"""
function propagate_arms!(server::ArmAllocationServer, cbelt::ConveyorBelt, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    diagnostic_info = ""

    # Update time
    server.current_time += 1.0

    @assert isequal(server.current_time,cbelt.current_time) "Server-Belt time mismatch!"

    # iterate through arms
    for (arm_name, arm) in server.agent_set

        arm_props = server.agent_prop_set[arm_name]

        if haskey(server.agent_task_allocation, arm_name)

            arm_assgn = server.agent_task_allocation[arm_name]

            # If box not in box routes, it is a stale entry that has already been picked up or lost - delete
            if ~(haskey(cbelt.box_routes, arm_assgn.name))
                delete!(server.agent_task_allocation, arm_name)
            else
                # Attempt a grab IF the noted time of the attempt is now
                if abs(server.current_time - arm_assgn.time) < 1.0

                    # Take arm to object position at reference point
                    arm_props.endeff_pos = get_box_point(cbelt, arm_assgn.name, arm_assgn.time)

                    @debug "$(arm_name) about to attempt $(arm_assgn.name)"
                    if rand(rng) < arm.grasp_model.bernoulli_prob  # Successful grab

                        @info arm_name, "has grabbed ",arm_assgn.name,"!"
                        arm_props.obj_grabbed = arm_assgn.name

                        # remove from arm-obj mutual assignment, workspace_objs, and windows
                        delete!(server.agent_task_allocation, arm_name)
                        delete!(server.agent_task_windows, (arm_name, arm_assgn.name))
                        delete!(arm_props.workspace_objs, arm_assgn.name)

                        # Remove box from belt
                        remove_box!(cbelt, arm_assgn.name)
                    else    # Unsuccessful grab

                        @info arm_name, "has missed ",arm_assgn.name,"!"
                        diagnostic_info = string(arm_name," misses ",arm_assgn.name,";",diagnostic_info)
                    end
                else
                    # Just go to rest pos
                    arm_props.endeff_pos = arm.orig_pos
                end # end abs()
            end # cbelt has box key
        else
            # First check if going to deposit object
            if arm_props.obj_grabbed != ""

                arm_props.endeff_pos = arm.bin_pos
                @info arm_name, " is depositing", arm_props.obj_grabbed
                arm_props.obj_grabbed = ""

                cbelt.deposited_boxes += 1

            else

                @info arm_name, "is at origin!"
                arm_props.endeff_pos = arm.orig_pos

            end
        end
    end

    return diagnostic_info
end

"""
Sets up the conveyor belt and arm allocation server with three hardcoded arms.
The other problem parameters are taken as named arguments and have defaults.
"""
function setup_conveyor_scenario(;grasp_probs::Vector{Float64}=[0.75, 0.75, 0.75],
                                 belt_speed::Float64=0.07,
                                 new_box_prob::Float64=0.75)

    # Set up the three arms
    arm1 = Arm(base_pos=[0.2, 0.75], bin_pos=[0.4, 0.75], reach=[0.06, 0.36],
               orig_pos=[0.2, 0.65], grasp_model=DiscreteGPM(grasp_probs[1]), downtime=2.0)
    arm1_props = ArmProperties(endeff_pos = arm1.orig_pos)

    arm2 = Arm(base_pos=[0.5, 0.25], bin_pos=[0.3, 0.25], reach=[0.36, 0.66],
               orig_pos=[0.5, 0.35], grasp_model=DiscreteGPM(grasp_probs[2]), downtime=2.0)
    arm2_props = ArmProperties(endeff_pos = arm2.orig_pos)

    arm3 = Arm(base_pos=[0.8, 0.75], bin_pos=[0.6, 0.75], reach=[0.66, 0.96],
               orig_pos=[0.8, 0.65], grasp_model=DiscreteGPM(grasp_probs[3]), downtime=2.0)
    arm3_props = ArmProperties(endeff_pos = arm3.orig_pos)

    arm_set = Dict("arm1"=>arm1, "arm2"=>arm2, "arm3"=>arm3)
    agent_prop_set = Dict("arm1"=>arm1_props, "arm2"=>arm2_props, "arm3"=>arm3_props)
    arm_ordering = ["arm1", "arm2", "arm3"]

    # Set up belt
    cbelt = ConveyorBelt(curr_speed=belt_speed, belt_start=0., belt_end=1., belt_width=0.2,
                      new_box_prob=new_box_prob, mirror_gen = MirrorGenerator(arm_set = [arm1, arm2, arm3]))
    cbelt.current_time = 0.0
    setup_mirror_gen!(cbelt)

    # Set up assignment server
    server = ArmAllocationServer(current_time=0.0, agent_set=arm_set,
                              agent_prop_set=agent_prop_set, agent_ordering = ["arm1", "arm2", "arm3"])

    return (server, cbelt)
end # function

function parse_conveyor_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--trials", "-n"
            arg_type = Int64
            default = 100
        "--timesteps", "-t"
            arg_type = Int64
            default = 500
        "--belt_speed", "-s"
            arg_type = Float64
            default = 0.07
        "--new_box_prob", "-p"
            arg_type = Float64
            default = 0.75
        "--grasp_probs", "-g"
            arg_type = String
            default = "0.75,0.75,0.75"
        "--qpol_name", "-q"
            arg_type = String
            default = ""
        "baseline"
            help = "Which method to evaluate (dispatch/hungarian/mcts/qlearning/scoba)"
            required = true
        "out_file_name"
            required = true
    end

    return parse_args(s)
end

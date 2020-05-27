@with_kw struct TimePoint
    timestamp::Float64                  = 0.0
    coordinates::SVector{2,Float64}     = [0.0, 0.0]
end

"""
A Bernoulli model for grasp success, given the number of attempts made.
"""
struct DiscreteGPM
    bernoulli_prob::Float64
end
cumprob(dgpm::DiscreteGPM, steps::Int64) = 1.0 - (1 - dgpm.bernoulli_prob)^steps

"""
Immutable arm parameters.
"""
@with_kw struct Arm
    base_pos::SVector{2,Float64}
    bin_pos::SVector{2,Float64}
    reach::SVector{2,Float64}
    orig_pos::SVector{2,Float64}
    grasp_model::DiscreteGPM
    downtime::Float64
end

"""
Mutable arm properties.
"""
@with_kw mutable struct ArmProperties
    endeff_pos::SVector{2,Float64}                  = [0.0, 0.0]
    obj_grabbed::String                             = ""
    workspace_objs::Set{String}                     = Set{String}()
    interaction_events::Vector{InteractionEvent}    = Vector{InteractionEvent}(undef, 0)
    tree::SearchTree                                = SearchTree()
end


"""
Sets up the reflected generator for feasible object sequences.
"""
@enum GENSTAGE PREDROP=1 POSTDROP=2 ATBIN=3
@with_kw mutable struct MirrorGenerator
    arm_set::Vector{Arm}                    = Vector{Arm}(undef, 0)
    gen_box_id::Int64                       = 0
    box_pos::Dict{Int64,Float64}            = Dict{Int64,Float64}()
    arm_stages::Vector{GENSTAGE}            = Vector{GENSTAGE}(undef, 0)
end

"""
The environment object for the domain.
"""
@with_kw mutable struct ConveyorBelt <: MRTAEnvironment
    current_time::Float64                       = 0.0
    num_active_boxes::Int64                     = 0
    num_total_boxes::Int64                      = 0
    curr_speed::Float64                         = 0.6
    box_routes::Dict{String,Vector{TimePoint}}  = Dict{String,Vector{TimePoint}}()
    belt_start::Float64                         = 0.0
    belt_end::Float64                           = 1.0
    belt_width::Float64                         = 0.1
    new_box_prob::Float64                       = 0.75
    lost_boxes::Int64                           = 0
    deposited_boxes::Int64                      = 0
    mirror_gen::MirrorGenerator                 = MirrorGenerator()
end
const ArmAllocationServer = GenericAllocation{Arm,ArmProperties}

## Define the datastructures that are only used by the generic SCoBA solver

"""
Every task has three modes in terms of its time window - start, finish success
"""
@enum MODE START=1 FINISH=2 SUCCESS=3


"""
An event between an agent and a task, and the planned time for the same.
N.B. Using integer IDs instead of strings would probably be more efficient.
"""
struct InteractionEvent
    agent_name::String
    task_name::String
    timestamps::Dict{MODE,Float64}
end

"""
Represents a decision node of the policy tree
"""
@with_kw mutable struct DecisionNode
    agent_name::String  = String("")
    task_name::String   = String("")
    attempt::Bool       = false
    timeval::Float64    = 0.0
    util::Float64       = -Inf
    idx::Int64          = 0
    utilset::Bool       = false
end

"""
Represents an outcome node of the policy tree
"""
@with_kw mutable struct OutcomeNode
    agent_name::String      = String("")
    task_name::String       = String("")
    outcome::MODE           = START
    timeval::Float64        = 0.0
    probability::Float64    = 0.0
    util::Float64           = -Inf
    idx::Int64              = 0
    utilset::Bool           = false
end

"""
The policy tree structure is a list of decision or outcome nodes.
"""
@with_kw mutable struct SearchTree
    nodes::Vector{Union{DecisionNode,OutcomeNode}}  = Vector{Union{DecisionNode,OutcomeNode}}(undef, 0)
    child_ids::Dict{Int64,Vector{Int64}}            = Dict{Int64,Vector{Int64}}()
    parent_id::Dict{Int64,Int64}                    = Dict{Int64,Int64}()
end

function Base.empty!(st::SearchTree)
    empty!(st.nodes)
    empty!(st.child_ids)
    empty!(st.parent_id)
end
Base.isempty(st::SearchTree) = (isempty(st.nodes) || isempty(st.child_ids))

# A for Agent Type (immutable), AP for AgentProperties (mutable)
@with_kw mutable struct GenericAllocation{A,AP}
    current_time::Float64       = 0.0
    agent_set::Dict{String,A}   = Dict{String,A}()
    agent_prop_set::Dict{String,AP} = Dict{String,AP}()
    agent_task_windows::Dict{Tuple{String,String}, SVector{3,Float64}}  = Dict{Tuple{String,String}, SVector{3,Float64}}()
    agent_task_allocation::Dict{String, NamedTuple{(:name, :time), Tuple{String, Float64}}} = Dict{String, NamedTuple{(:name, :time), Tuple{String, Float64}}}()
    agent_ordering::Vector{String}  = String[]
    max_tasks_to_consider::Int64    = 100000
    conflict_threshold::Int64        = 100
end

abstract type MRTAEnvironment end

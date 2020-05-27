module SCoBASolver

using StaticArrays
using Parameters
using DataStructures


# Types etc
export
    MODE,
    START,
    FINISH,
    SUCCESS,
    InteractionEvent,
    SearchTree,
    GenericAllocation,
    MRTAEnvironment,
    DecisionNode,
    OutcomeNode,
    TaskUtil

# Solver stuff
export
    generate_search_tree!,
    get_next_attempt_idx,
    SCoBAAlgorithm,
    coordinate_allocation!



include("types.jl")
include("scoba_tree_search.jl")
include("scoba_conflict_resolution.jl")

end

module SCoBADomains

using Parameters
using StaticArrays
using Random
using Hungarian
using Distributions
using DataStructures
using ArgParse
using POMDPs
using Distances
using NearestNeighbors
using TOML

using Plots
using GeometryTypes

using SCoBA.SCoBASolver

# Conveyor belt types and simulator
export
    Arm,
    ArmProperties,
    ConveyorBelt,
    ArmAllocationServer,
    count_terminal_grabbed_boxes!,
    propagate_belt!,
    update_time_windows!,
    setup_mirror_gen!,
    mirror_gen_step!,
    add_box_to_belt!,
    propagate_arms!,
    setup_conveyor_scenario,
    parse_conveyor_commandline

# Conveyor belt algorithms - SCoBA and baseline stuff
export
    cumul_grasp_prob,
    scoba_conveyor!,
    expected_hungarian!,
    earliest_due_date!,
    ConvBeltSystemState,
    ConvBeltMDP,
    create_convbelt_mdp,
    get_current_belt_mdp_state,
    EDDPolicy,
    ConvBeltSingleArmMDP,
    create_convbelt_singlearm_mdp,
    get_singlearm_mdp_state,
    update_server_with_mdp_action!

# Routing types and simulator
export
    parse_city_params,
    Drone,
    DroneProperties,
    Package,
    RoutingAllocation,
    RoutingSimulator,
    NearestPkgPolicy,
    get_travel_time_estimate,
    sample_true_delivery_return_time,
    setup_routing_sim,
    update_routing_sim!,
    parse_routing_commandline,
    EuclideanLatLongMetric,
    LatLonCoords

# Routing algorithms - SCoBA and baselines
export
    scoba_routing!,
    RoutingMCTSMDP,
    update_routing_mcts_fullstate!,
    get_current_routing_mcts_state


include("conveyor_belt/conveyor_belt_types.jl")
include("conveyor_belt/conveyor_belt_simulator.jl")
include("conveyor_belt/conveyor_belt_scoba.jl")
include("conveyor_belt/conveyor_belt_baselines.jl")

include("routing/routing_types.jl")
include("routing/routing_simulator.jl")
include("routing/routing_scoba.jl")
include("routing/routing_baselines.jl")

end

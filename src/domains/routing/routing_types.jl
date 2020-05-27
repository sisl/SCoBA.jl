## Various helper functions for working with latitude and longitude
const Location2D = SVector{2, Float64}
const LatLonCoords = NamedTuple{(:lat, :lon), Tuple{Float64,Float64}}
LatLonCoords() = (lat = 0.0, lon = 0.0)

"""
Return the vector form of a latitude-longitude point [lat, lon]
"""
function convert_to_vector(c::LatLonCoords)
    return Location2D(c.lat, c.lon)
end

struct EuclideanLatLongMetric <: Metric
end

function Distances.evaluate(::EuclideanLatLongMetric,
                            coords1::Location2D,
                            coords2::Location2D)
    deglen = 110.25
    x = coords1[1]- coords2[1]
    y = (coords1[2] - coords2[2])*cos(coords2[1])
    return deglen*sqrt(x^2 + y^2)
end

@with_kw struct CityParams
    lat_start::Float64
    lat_end::Float64
    lon_start::Float64
    lon_end::Float64
end

function parse_city_params(param_file::String)

    params_dict = TOML.parsefile(param_file)

    return CityParams(lat_start = params_dict["LATSTART"],
                      lat_end = params_dict["LATEND"],
                      lon_start = params_dict["LONSTART"],
                      lon_end = params_dict["LONEND"])
end

@with_kw struct Drone
    avg_speed::Float64      = 0.00777
    depot_number::Int64
    depot_loc::LatLonCoords
end

@with_kw mutable struct DroneProperties
    at_depot::Bool                                  = true
    current_package::String                         = ""
    interaction_events::Vector{InteractionEvent}    = Vector{InteractionEvent}(undef, 0)
    tree::SearchTree                                = SearchTree()
end

const RoutingAllocation = GenericAllocation{Drone,DroneProperties}

struct Package
    delivery::LatLonCoords
    time_window::Tuple{Float64,Float64}
end


"""
Bookkeeping data structure only useful for plotting.
"""
@with_kw struct CurrDroneSiteLocs
    curr_drone_locs_cols::Vector{Tuple{LatLonCoords,Symbol}}    = Tuple{LatLonCoords,Symbol}[]
    curr_sites_locs_cols::Vector{Tuple{LatLonCoords,Symbol}}    = Tuple{LatLonCoords,Symbol}[]
end

@with_kw mutable struct RoutingSimulator <: MRTAEnvironment
    current_time::Float64
    city_params::CityParams
    new_request_prob::Float64
    time_window_duration::Float64
    halton_nn_tree::BallTree
    estimate_matrix::Matrix{Float64}
    delivery_reward::Float64
    active_packages::Dict{String,Package}   = Dict{String,Package}()
    busy_packages::Dict{String,Package}     = Dict{String,Package}()
    done_packages::Dict{String,Package}     = Dict{String,Package}()
    num_total_packages::Int64               = 0
    num_active_packages::Int64              = 0
    late_packages::Int64                    = 0
    delivered_packages::Int64               = 0
    sum_of_delivery_time::Float64           = 0.0
    time_scale::Float64                     = 60.0 # Dealing in 1-minute chunks
    true_delivery_return::Dict{Tuple{String,String},Tuple{Float64,Float64}} = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
    tt_est_std_scale::Float64               = 3.0
    distance_thresh::Float64                = 5.0
    curr_drone_site_locs::CurrDroneSiteLocs = CurrDroneSiteLocs()
end

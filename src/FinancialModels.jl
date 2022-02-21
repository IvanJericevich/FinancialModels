module FinancialModels
include("common.jl")
include("JacobLeal/JacobLeal.jl")
include("api.jl")
export Options, JacobLealParameters, Simulate
end
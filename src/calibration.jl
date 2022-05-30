using FinancialModels, ARCHModels, Polynomials, JLD
using HypothesisTests: ADFTest, ApproximateTwoSampleKSTest
using Statistics: mean, quantile, std
using StatsBase: kurtosis
using GLM: lm, coef
include("NMTA.jl")
struct Moments # Moments of log-returns
    μ::Float64 # Mean
    σ::Float64 # Standard deviation
    κ::Float64 # Kurtosis
    ks::Float64 # Kolmogorov-Smirnov test statistic for the difference between distributions
    hurst::Float64 # Hurst exponent: hurst < 0.5 => mean reverting; hurst == 0.5 => random walk; hurst > 0.5 => momentum
    gph::Float64 # GPH estimator representing long-range dependence
    adf::Float64 # ADF statistic representing random walk property of returns
    garch::Float64 # GARCH paramaters representing short-range dependence
    hill::Float64 # Hill estimator
    function Moments(log_returns1::Vector{Float64}, log_returns2::Vector{Float64})
        μ = mean(log_returns1); σ = std(log_returns1); κ = kurtosis(log_returns1)
        ks = ApproximateTwoSampleKSTest(log_returns1, log_returns2).δ
        hurst = HurstExponent(log_returns1)
        gph = GPH(abs.(log_returns1))
        adf = ADFTest(log_returns1, :none, 0).stat
        garch = sum(coef(ARCHModels.fit(GARCH{1, 1}, log_returns1))[2:3])
        hill = HillEstimator(log_returns1[findall(x -> (x >= quantile(log_returns1, 0.95)) && (x > 0), log_returns1)], 50)
        new(μ, σ, κ, ks, hurst, gph, adf, garch, hill)
    end
end
function HurstExponent(x, d = 100)
    function Divisors(n, n₀)
        return filter(x -> mod(n, x) == 0, n₀:floor(n/2))
    end
    N = length(x)
    if mod(N, 2) != 0 x = push!(x, (x[N - 1] + x[N]) / 2); N += 1 end
    N₁ = N₀ = min(floor(0.99 * N), N-1); dv = Divisors(N₁, d)
    for i in (N₀ + 1):N
        dw = Divisors(i, d)
        if length(dw) > length(dv) N₁ = i; dv = copy(dw) end
    end
    OptN = Int(N₁); d = dv
    x = x[1:OptN]
    RSempirical = map(i -> RS(x, i), d)
    return coeffs(Polynomials.fit(Polynomial, log10.(d), log10.(RSempirical), 1))[2] # Hurst is slope of log-log linear fit
end
function RS(z, n)
    y = reshape(z, (Int(n), Int(length(z) / n)))
    μ = mean(y, dims = 1)
    σ = std(y, dims = 1)
    temp = cumsum(y .- μ, dims = 1)
    return mean((maximum(temp, dims = 1) - minimum(temp, dims = 1)) / σ)
end
function GPH(x, bandwidthExponent = 0.5)
    n = length(x); g = Int(trunc(n^bandwidthExponent))
    j = 1:g; kk = 1:(n - 1)
    w = 2 .* π .* j ./ n # x .-= mean(x)
    σ = sum(x .^ 2) / n
    Σ = map(k -> sum(x[1:(n - k)] .* x[(1 + k):n]) / n, kk)
    periodogram = map(i -> σ + 2 * sum(Σ .* cos.(w[i] .* kk)), j)
    indeces = j[findall(x -> x > 0, periodogram)]
    x_reg = 2 .* log.(2 .* sin.(w[indeces] ./ 2)); y_reg = log.(periodogram[indeces] ./ (2 * π))
    regression = lm(hcat(ones(length(x_reg)), x_reg), y_reg)
    return abs(coef(regression)[2])
end
function HillEstimator(x, iterations)
    N = length(x)
    logx = log.(x)
    L = minimum(x); R = maximum(x)
    α = 1 / ((sum(logx) / N) - log(L))
    for i in 1:iterations
        C = (log(L) * (L^(-α)) - log(R) * (R^(-α))) / ((L^(-α)) - (R^(-α)))
        D = (R^α * L^α * (log(L) - log(R))^2) / (L^α - R^α)^2
        α = α * (1 + (α * (sum(logx) / N) - α * C - 1) / (α^2 * D - 1))
    end
    return α
end
function MovingBlockBootstrap(mid_price::Vector{Float64}, iterations::Int64 = 1000, window_size::Int64 = 2000)
    log_returns = diff(log.(mid_price))
    bootstrap_moments = Vector{Vector{Float64}}()
    Threads.@threads for i in 1:iterations
        indeces = rand(1:(length(log_returns) - window_size + 1), Int(ceil(length(log_returns) / window_size)))
        bootstrap_returns = Vector{Float64}()
        for index in indeces
            append!(bootstrap_returns, log_returns[index:(index  + window_size - 1)])
        end
        moments = Moments(bootstrap_returns[1:length(log_returns)], log_returns)
        push!(bootstrap_moments, [moments.μ, moments.σ, moments.κ, moments.ks, moments.hurst, moments.gph, moments.adf, moments.garch, moments.hill])
    end
    W = inv(cov(bootstrap_moments))
    save("data/W.jld", "W", W)
end
function WSSE(parameters::JacobLealParameters, replications::Int64, W::Array{Float64, 2}, empirical_moments::Moments, empirical_log_returns::Vector{Float64})
    errormatrix = fill(0.0, (replications, 9))
    Threads.@threads for i in 1:replications
        mid_price = prices(Simulate(parameters, options = Options(time = 10000)))
        log_returns = diff(log.(mid_price))
        simulated_moments = Moments(log_returns, empirical_log_returns)
        errormatrix[i, :] = [simulated_moments.μ-empirical_moments.μ simulated_moments.σ-empirical_moments.σ simulated_moments.κ-empirical_moments.κ simulated_moments.ks-empirical_moments.ks simulated_moments.hurst-empirical_moments.hurst simulated_moments.gph-empirical_moments.gph simulated_moments.adf-empirical_moments.adf simulated_moments.garch-empirical_moments.garch simulated_moments.hill-empirical_moments.hill]
    end
    errors = mean(errormatrix, dims = 1)
    return (errors * W * transpose(errors))[1]
end

mid_price = CSV.File("data/L1LOB.csv", missingstring = "missing", ignoreemptylines = true, select = [:MicroPrice], skipto = 20000, limit = 20000) |> Tables.matrix |> vec |> y -> filter(z -> !ismissing(z), y)
empirical_log_returns = diff(log.(mid_price))
empirical_moments = Moments(empirical_log_returns, empirical_log_returns)
MovingBlockBootstrap(mid_price, 10, 2000)
W = load("W.jld")["W"]
initial_solution = JacobLealParameters()
initial_solution = [initial_solution[1], initial_solution[2], initial_solution[3], initial_solution[4], initial_solution[5], initial_solution[6], initial_solution[7],
initial_solution[8], initial_solution[9], initial_solution[10], initial_solution[11], initial_solution[12],  initial_solution[13]]
ta_rounds = [12, 10, 8, 6]
f_reltol = collect(range(10, 0,  sum(ta_rounds)))
neldermeadstate = nothing
objective = NonDifferentiable(x -> WSSE(x -> JacobLealParameters(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12],  x[13]), 10, W, empirical_moments, empirical_log_returns), initial_solution)
optimization_options = Options(show_trace = true, store_trace = true, trace_simplex = true, extended_trace = true, iterations = 30, ξ = 0.15, ta_rounds = ta_rounds, f_reltol = f_reltol)
result = !isnothing(neldermeadstate) ? Optimize(objective, initial_solution, optimization_options, neldermead_state) : Optimize(objective, initial_solution, optimization_options)
save("OptimizationResult.jld", "result", result)
#---------------------------------------------------------------------------------------------------

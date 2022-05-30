using FinancialModels

#----- Calibration -----#
include("NMTA.jl")
import HypothesisTests: ADFTest, ApproximateTwoSampleKSTest
import Statistics: mean, quantile, std
import StatsBase.kurtosis
import GLM: lm, coef
using ARCHModels, Polynomials, JLD
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
    function Moments(logreturns1::Vector{Float64}, logreturns2::Vector{Float64})
        μ = mean(logreturns1); σ = std(logreturns1); κ = kurtosis(logreturns1)
        ks = ApproximateTwoSampleKSTest(logreturns1, logreturns2).δ
        hurst = HurstExponent(logreturns1)
        gph = GPH(abs.(logreturns1))
        adf = ADFTest(logreturns1, :none, 0).stat
        garch = sum(coef(ARCHModels.fit(GARCH{1, 1}, logreturns1))[2:3])
        hill = HillEstimator(logreturns1[findall(x -> (x >= quantile(logreturns1, 0.95)) && (x > 0), logreturns1)], 50)
        new(μ, σ, κ, ks, hurst, gph, adf, garch, hill)
    end
end
function HurstExponent(x, d = 100)
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
function Divisors(n, n₀)
    return filter(x -> mod(n, x) == 0, n₀:floor(n/2))
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
function MovingBlockBootstrap(midprice::Vector{Float64}, iterations::Int64 = 1000, windowsize::Int64 = 2000)
    logreturns = diff(log.(midprice))
    bootstrapmoments = Vector{Vector{Float64}}()
    Threads.@threads for i in 1:iterations
        indeces = rand(1:(length(logreturns) - windowsize + 1), Int(ceil(length(logreturns)/windowsize)))
        bootstrapreturns = Vector{Float64}()
        for index in indeces
            append!(bootstrapreturns, logreturns[index:(index  + windowsize - 1)])
        end
        moments = Moments(bootstrapreturns[1:length(logreturns)], logreturns)
        [moments.μ moments.σ moments.κ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
    end
    W = inv(cov(bootstrapmoments))
    save("Data/W.jld", "W", W)
end
function WeightedSumofSquaredErrors(parameters::Parameters, replications::Int64, W::Array{Float64, 2}, empiricalmoments::Moments, empiricallogreturns::Vector{Float64}, gateway::TradingGateway)
    errormatrix = fill(0.0, (replications, 9))
    for i in 1:replications
        microprice = InjectSimulation(gateway, parameters, seed = i)
        if !isempty(microprice)
            filter!(x -> !isnan(x), microprice)
            logreturns = diff(log.(microprice))
            try
                simulatedmoments = Moments(logreturns, empiricallogreturns)
                errormatrix[i, :] = [simulatedmoments.μ-empiricalmoments.μ simulatedmoments.σ-empiricalmoments.σ simulatedmoments.κ-empiricalmoments.κ simulatedmoments.ks-empiricalmoments.ks simulatedmoments.hurst-empiricalmoments.hurst simulatedmoments.gph-empiricalmoments.gph simulatedmoments.adf-empiricalmoments.adf simulatedmoments.garch-empiricalmoments.garch simulatedmoments.hill-empiricalmoments.hill]
            catch e
                println(e)
                errormatrix[i, :] = errormatrix[i - 1, :]
            end
        else
            return Inf
        end
    end
    GC.gc() # Garbage collection
    errors = mean(errormatrix, dims = 1)
    return (errors * W * transpose(errors))[1]
end
#---------------------------------------------------------------------------------------------------

#----- Simulation -----#
# options = Options(time = 10000, show_trace = true, show_every = 100)
# results = Simulate(JacobLealParameters(), options = options)
#---------------------------------------------------------------------------------------------------

#----- Monte-Carlo Simulation -----#
# monte_carlo_relpications = 10
# simulations = Vector{Vector{Float64}}()#[Vector{Float64}() for r in 1:monte_carlo_relpications]
# Threads.@threads for r in 1:monte_carlo_relpications # Parallel for loop\
#     print(r)
#     options = Options(time = 1000, seed = r)
# 	push!(simulations, prices(Simulate(JacobLealParameters(), options = options)))
# end
#---------------------------------------------------------------------------------------------------

#----- Visualisation -----#
# using Plots
# using Distributions: fit, Normal
# using StatsBase: autocor, quantile
# using StatsPlots: histogram, qqplot!, density, density!
# function SavePlot(p, fn::String, save::Union{Bool, String} = false, format = "png")
#     if isa(save, String)
#         savefig(p, string(save, "/", fn, format))
#     elseif save
#         savefig(p, string(fn, ".", format))
#     end
# end
# function PlotSimulation(results::SimulationResults, save::Union{Bool, String} = false, format = "png")
#     p = prices(results)
#     log_returns = diff(log.(p))
#     l = @layout([a; b{0.2h}])
#     p1 = plot(p, linecolor = :black, ylabel = "pₜ", legend = false, xticks = false)
#     p2 = plot(log_returns, seriestype = :line, linecolor = :black, legend = false, ylabel = "Log-returns", xlabel = "t")
#     simulation_plot = plot(p1, p2, layout = l, link = :x, tickfontsize = 5, guidefontsize = 7)
#     SavePlot(simulation_plot, "JacobLealSimulation", save, format)
#     return simulation_plot
# end
# function PlotLogReturnDistribution(results::SimulationResults, save::Union{Bool, String} = false, format = "png")
#     log_returns = diff(log.(prices(results)))
#     fitted_normal = fit(Normal, log_returns)
#     distribution = histogram(log_returns, normalize = :pdf, fillcolor = :orange, linecolor = :orange, xlabel = "Log returns",
#     ylabel = "Probability Density", label = "Empirical", legend = :topright, legendfontsize = 5, fg_legend = :transparent,
#     guidefontsize = 7, tickfontsize = 5, ylim = (0, 150), xlim = (-0.04, 0.04))
#     plot!(distribution, fitted_normal, line = (:black, 2), label = "Fitted Normal")
#     qqplot!(distribution, Normal, log_returns, xlabel = "Normal theoretical quantiles", ylabel = "Sample quantiles", linecolor = :black,
#     guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, marker = (:orange, stroke(:orange), 3), legend = false,
#     inset = (1, bbox(0.1, 0.03, 0.4, 0.4)), subplot = 2, title = "Normal QQ-plot", titlefontsize = 7)
#     SavePlot(distribution, "JacobLealDistribution", save, format)
#     return distribution
# end
# function PlotLogReturnAutoCorrelations(results::SimulationResults, save::Union{Bool, String} = false, format = "png")
#     log_returns = diff(log.(prices(results)))
#     auto_corr = autocor(log_returns, 1:500; demean = false)
#     abs_auto_corr = autocor(abs.(log_returns), 1:500; demean = false)
#     auto_corr_plot = plot(auto_corr, seriestype = [:sticks, :scatter], marker = (:orange, stroke(:orange), 3), linecolor = :black, xlabel = "Lag",
#     ylabel = "Autocorrelation", legend = false, guidefontsize = 7, tickfontsize = 5, ylim = (-0.4, 0.2))
#     plot!(auto_corr_plot, [1.96 / sqrt(length(log_returns)), -1.96 / sqrt(length(log_returns))], seriestype = :hline, line = (:dash, :black, 2))
#     plot!(auto_corr_plot, abs_auto_corr, seriestype = :scatter, marker = (:orange, stroke(:orange), 3), legend = false, xlabel = "Lag",
#     ylabel = "Autocorrelation", inset = (1, bbox(0.62, 0.5, 0.4, 0.4, :top)), subplot = 2, guidefontsize = 7, tickfontsize = 5, xrotation = 30,
#     yrotation = 30, title = "Absolute log-return autocorrelation", titlefontsize = 7)
#     SavePlot(auto_corr_plot, "JacobLealAutoCorrelation", save, format)
#     return auto_corr_plot
# end
# function PlotExtremeLogReturnDistribution(results::SimulationResults, save::Union{Bool, String} = false, format = "png")
#     log_returns = diff(log.(prices(results)))
#     upper_observations = log_returns[findall(x -> x >= quantile(log_returns, 0.95), log_returns)]; sort!(upper_observations)
#     upperxₘᵢₙ = minimum(upper_observations)
#     upperα = 1 + length(upper_observations) / sum(log.(upper_observations ./ upperxₘᵢₙ))
#     upper_theoretical_quantiles = map(i -> (1 - (i / length(upper_observations))) ^ (-1 / (upperα - 1)) * upperxₘᵢₙ, 1:length(upper_observations))
#     lower_observations = -log_returns[findall(x -> x <= quantile(log_returns, 0.05), log_returns)]; sort!(lower_observations)
#     lowerxₘᵢₙ = minimum(lower_observations)
#     lowerα = 1 + length(lower_observations) / sum(log.(lower_observations ./ lowerxₘᵢₙ))
#     lower_theoretical_quantiles = map(i -> (1 - (i / length(lower_observations))) ^ (-1 / (lowerα - 1)) * lowerxₘᵢₙ, 1:length(lower_observations))
#     distribution = density(upper_observations, seriestype = [:scatter, :line], marker = (:orange, stroke(:orange), :utriangle), linecolor = :orange,
#     xlabel = string("Log return extreme percentiles"), ylabel = "Density", label = string("Upper percentiles - α = ", round(upperα, digits = 3)),
#     legend = :topright, fg_legend = :transparent, guidefontsize = 7, tickfontsize = 5)
#     density!(distribution, lower_observations, seriestype = [:scatter, :line], marker = (:orange, stroke(:orange), :dtriangle), linecolor = :orange,
#     label = string("Lower percentiles - α = ", round(lowerα, digits = 3)))
#     plot!(distribution, hcat(upper_theoretical_quantiles, upper_theoretical_quantiles), hcat(upper_observations, upper_theoretical_quantiles),
#     scale = :log10, seriestype = [:scatter :line], inset = (1, bbox(0.6, 0.3, 0.34, 0.34, :top)), subplot = 2, guidefontsize = 7,
#     tickfontsize = 5, xrotation = 30, yrotation = 30, legend = :none, xlabel = "Power-Law Theoretical Quantiles", ylabel = "Sample Quantiles",
#     linecolor = :black, marker = (:orange, stroke(:orange), 3, [:utriangle :none]), fg_legend = :transparent, title = "Power-Law QQ-plot", titlefontsize = 7)
#     plot!(distribution, [lower_theoretical_quantiles lower_theoretical_quantiles], [lower_observations lower_theoretical_quantiles],
#     seriestype = [:scatter :line], subplot = 2, linecolor = :black, marker = (:orange, stroke(:orange), 3, [:dtriangle :none]))
#     SavePlot(distribution, "JacobLealExtremeDistribution", save, format)
#     return distribution
# end
# PlotSimulation(results, false)
# PlotLogReturnDistribution(results, false)
# PlotLogReturnAutoCorrelations(results, false)
# PlotExtremeLogReturnDistribution(results, false)
#---------------------------------------------------------------------------------------------------
using FinancialModels, Plots
using Distributions: fit, Normal
using StatsBase: autocor, quantile
using StatsPlots: histogram, qqplot!, density, density!

function SavePlot(p, fn::String, save::Union{Bool, String} = false, format = "png")
    if isa(save, String)
        savefig(p, string(save, "/", fn, format))
    elseif save
        savefig(p, string(fn, ".", format))
    end
end

function PlotSimulation(results::SimulationResults, save::Union{Bool, String} = false, format = "png")
    p = prices(results)
    log_returns = diff(log.(p))
    l = @layout([a; b{0.2h}])
    p1 = plot(p, linecolor = :black, ylabel = "pₜ", legend = false, xticks = false)
    p2 = plot(log_returns, seriestype = :line, linecolor = :black, legend = false, ylabel = "Log-returns", xlabel = "t")
    simulation_plot = plot(p1, p2, layout = l, link = :x, tickfontsize = 5, guidefontsize = 7)
    SavePlot(simulation_plot, "JacobLealSimulation", save, format)
    return simulation_plot
end

function PlotLogReturnDistribution(results::SimulationResults, save::Union{Bool, String} = false, format = "png")
    log_returns = diff(log.(prices(results)))
    fitted_normal = fit(Normal, log_returns)
    distribution = histogram(log_returns, normalize = :pdf, fillcolor = :orange, linecolor = :orange, xlabel = "Log returns",
    ylabel = "Probability Density", label = "Empirical", legend = :topright, legendfontsize = 5, fg_legend = :transparent,
    guidefontsize = 7, tickfontsize = 5, ylim = (0, 100))
    plot!(distribution, fitted_normal, line = (:black, 2), label = "Fitted Normal")
    qqplot!(distribution, Normal, log_returns, xlabel = "Normal theoretical quantiles", ylabel = "Sample quantiles", linecolor = :black,
    guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, marker = (:orange, stroke(:orange), 3), legend = false,
    inset = (1, bbox(0.62, 0.2, 0.35, 0.35)), subplot = 2, title = "Normal QQ-plot", titlefontsize = 7)
    SavePlot(distribution, "JacobLealDistribution", save, format)
    return distribution
end

function PlotLogReturnAutoCorrelations(results::SimulationResults, save::Union{Bool, String} = false, format = "png")
    log_returns = diff(log.(prices(results)))
    auto_corr = autocor(log_returns, 1:100; demean = false)
    abs_auto_corr = autocor(abs.(log_returns), 1:100; demean = false)
    auto_corr_plot = plot(auto_corr, seriestype = [:sticks, :scatter], marker = (:orange, stroke(:orange), 3), linecolor = :black, xlabel = "Lag",
    ylabel = "Autocorrelation", legend = false, guidefontsize = 7, tickfontsize = 5)
    plot!(auto_corr_plot, [1.96 / sqrt(length(log_returns)), -1.96 / sqrt(length(log_returns))], seriestype = :hline, line = (:dash, :black, 2))
    plot!(auto_corr_plot, abs_auto_corr, seriestype = :scatter, marker = (:orange, stroke(:orange), 3), legend = false, xlabel = "Lag",
    ylabel = "Autocorrelation", inset = (1, bbox(0.62, 0.1, 0.4, 0.4, :top)), subplot = 2, guidefontsize = 7, tickfontsize = 5, xrotation = 30,
    yrotation = 30, title = "Absolute log-return autocorrelation", titlefontsize = 7)
    SavePlot(auto_corr_plot, "JacobLealAutoCorrelation", save, format)
    return auto_corr_plot
end

function PlotExtremeLogReturnDistribution(results::SimulationResults, save::Union{Bool, String} = false, format = "png")
    log_returns = diff(log.(prices(results)))
    upper_observations = log_returns[findall(x -> x >= quantile(log_returns, 0.95), log_returns)]; sort!(upper_observations)
    upperxₘᵢₙ = minimum(upper_observations)
    upperα = 1 + length(upper_observations) / sum(log.(upper_observations ./ upperxₘᵢₙ))
    upper_theoretical_quantiles = map(i -> (1 - (i / length(upper_observations))) ^ (-1 / (upperα - 1)) * upperxₘᵢₙ, 1:length(upper_observations))
    lower_observations = -log_returns[findall(x -> x <= quantile(log_returns, 0.05), log_returns)]; sort!(lower_observations)
    lowerxₘᵢₙ = minimum(lower_observations)
    lowerα = 1 + length(lower_observations) / sum(log.(lower_observations ./ lowerxₘᵢₙ))
    lower_theoretical_quantiles = map(i -> (1 - (i / length(lower_observations))) ^ (-1 / (lowerα - 1)) * lowerxₘᵢₙ, 1:length(lower_observations))
    distribution = density(upper_observations, seriestype = [:scatter, :line], marker = (:orange, stroke(:orange), :utriangle), linecolor = :orange,
    xlabel = string("Log return extreme percentiles"), ylabel = "Density", label = string("Upper percentiles - α = ", round(upperα, digits = 3)),
    legend = :topright, fg_legend = :transparent, guidefontsize = 7, tickfontsize = 5)
    density!(distribution, lower_observations, seriestype = [:scatter, :line], marker = (:orange, stroke(:orange), :dtriangle), linecolor = :orange,
    label = string("Lower percentiles - α = ", round(lowerα, digits = 3)))
    plot!(distribution, hcat(upper_theoretical_quantiles, upper_theoretical_quantiles), hcat(upper_observations, upper_theoretical_quantiles),
    scale = :log10, seriestype = [:scatter :line], inset = (1, bbox(0.6, 0.3, 0.34, 0.34, :top)), subplot = 2, guidefontsize = 7,
    tickfontsize = 5, xrotation = 30, yrotation = 30, legend = :none, xlabel = "Power-Law Theoretical Quantiles", ylabel = "Sample Quantiles",
    linecolor = :black, marker = (:orange, stroke(:orange), 3, [:utriangle :none]), fg_legend = :transparent, title = "Power-Law QQ-plot", titlefontsize = 7)
    plot!(distribution, [lower_theoretical_quantiles lower_theoretical_quantiles], [lower_observations lower_theoretical_quantiles],
    seriestype = [:scatter :line], subplot = 2, linecolor = :black, marker = (:orange, stroke(:orange), 3, [:dtriangle :none]))
    SavePlot(distribution, "JacobLealExtremeDistribution", save, format)
    return distribution
end

options = Options(time = 10000, show_trace = true, store_trace = false, show_every = 100)
results = Simulate(JacobLealParameters(), options = options)
save = true
PlotSimulation(results, save, "pdf")
PlotLogReturnDistribution(results, save, "pdf")
PlotLogReturnAutoCorrelations(results, save, "pdf")
PlotExtremeLogReturnDistribution(results, save, "pdf")
#---------------------------------------------------------------------------------------------------

#----- Monte-Carlo Simulation -----#
# monte_carlo_relpications = 50
# parameters = ModelParameters()
# closing_prices = @showprogress 1 "Simulating" @distributed (hcat) for r in 1:monte_carlo_relpications # Parallel for loop
# 	return SimulateJacobLealModel(parameters)
# end
# save("src/data/SimulationResults.jld", "results", results)
#---------------------------------------------------------------------------------------------------
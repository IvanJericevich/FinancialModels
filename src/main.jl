import FinancialModels

options = Options(time = 1000, show_trace = true, store_trace = true, show_every = 100)
x = Simulate(JacobLealParameters(), options = options)








# l = @layout([a; b{0.2h}])
# mid_price = plot(x, linecolor = :black, ylabel = "pₜ", legend = false, xticks = false)
# logreturns = diff(log.(x))
# returns = plot(logreturns, seriestype = :line, linecolor = :black, legend = false, ylabel = "Log-returns", xlabel = "t")
# plot(mid_price, returns, layout = l, link = :x, tickfontsize = 5, guidefontsize = 7)



# fitted_normal = fit(Normal, logreturns)
# distribution = histogram(logreturns, normalize = :pdf, fillcolor = :red, linecolor = :red, xlabel = "Log returns", ylabel = "Probability Density", label = "Empirical", legend = :topright, legendfontsize = 5, fg_legend = :transparent, guidefontsize = 7, tickfontsize = 5)
# plot!(distribution, fitted_normal, line = (:black, 2), label = "Fitted Normal")
# qqplot!(distribution, Normal, logreturns, xlabel = "Normal theoretical quantiles", ylabel = "Sample quantiles", linecolor = :black, guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, marker = (:red, stroke(:red), 3), legend = false, inset = (1, bbox(0.6, 0.2, 0.4, 0.4)), subplot = 2, title = "Normal QQ-plot", titlefontsize = 7)


# auto_corr = autocor(logreturns, 1:100; demean = false)
# abs_auto_corr = autocor(abs.(logreturns), 1:100; demean = false)
# auto_corr_plot = plot(auto_corr, seriestype = [:sticks, :scatter], marker = (:red, stroke(:red), 3), linecolor = :black, xlabel = "Lag", ylabel = "Autocorrelation", legend = false, guidefontsize = 7, tickfontsize = 5)
# plot!(auto_corr_plot, [1.96 / sqrt(length(logreturns)), -1.96 / sqrt(length(logreturns))], seriestype = :hline, line = (:dash, :black, 2))
# plot!(auto_corr_plot, abs_auto_corr, seriestype = :scatter, marker = (:red, stroke(:red), 3), legend = false, xlabel = "Lag", ylabel = "Autocorrelation", inset = (1, bbox(0.62, 0.1, 0.4, 0.4, :top)), subplot = 2, guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, title = "Absolute log-return autocorrelation", titlefontsize = 7)
#---------------------------------------------------------------------------------------------------

#----- Monte-Carlo Simulation -----#
# monte_carlo_relpications = 50
# parameters = ModelParameters()
# closing_prices = @showprogress 1 "Simulating" @distributed (hcat) for r in 1:monte_carlo_relpications # Parallel for loop
# 	return SimulateJacobLealModel(parameters)
# end
# save("src/data/SimulationResults.jld", "results", results)
#---------------------------------------------------------------------------------------------------
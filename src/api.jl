#----- API -----#
trace(results::SimulationResults) = length(results.trace) > 0 ? results.trace : error("No trace in optimization results. To get a trace, run simulation with store_trace = true.")
prices(results::SimulationResults) = results.p̄
#---------------------------------------------------------------------------------------------------
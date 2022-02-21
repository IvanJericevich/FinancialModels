#----- API -----#
trace(results::Results) = length(results.trace) > 0 ? results.trace : error("No trace in optimization results. To get a trace, run simulation with store_trace = true.")
prices(results::Results) = results.p̄
#---------------------------------------------------------------------------------------------------
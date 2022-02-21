#=
Jacob-Leal model
- Julia version: 1.7.1
- Authors: Ivan Jericevich, Patrick Chang, Tim Gebbie
=#
using Random, Distributions, ProgressMeter, Printf

#----- Structures -----#
mutable struct JacobLealParameters <: Parameters
	# Fixed parameters
    Nᴸ::Int64 # Number of low-frequency agents
    Nᴴ::Int64 # Number of high-frequency agents
    θ::Int64 # Mean trading frequency for LF trader (minutes)
	θ⁺::Int64 # Upper-bound trading frequency for LF traders (minutes)
	θ⁻::Int64 # Lower-bound trading frequency for LF traders (minutes)
	σ_chartist::Float64 # Chartists' random normal order size standard deviation
	α_chartist::Float64 # Chartists' order size scaling factor (0 < α < 1)
	σ_fundamentalist::Float64 # Fundamentlists' random normal order size standard deviation
	α_fundamentalist::Float64 # Fundamentalists' order size scaling factor
	σʸ::Float64 # Fundamental value standard deviation
	σᶻ::Float64 # LF traders' limit-price tick standard deviation
	δ::Float64 # Price drift for LF traders
	γᴸ::Int64 # LF traders' unexectuted order resting period until expiry
	γᴴ::Int64 # HF traders' unexectuted order resting period until expiry
	κ⁺::Float64 # Uniform distribution upper-bound for HF traders' order-price support
	κ⁻::Float64 # Uniform distribution lower-bound for HF traders' order-price support
	η⁺::Float64 # Uniform distribution upper-bound for HF traders' activation
	η⁻::Float64 # Uniform distribution lower-bound for HF traders' activation
	λ::Float64 # Market volumes weight in HF traders' order size distribution
	ζ::Float64 # Intensity of switching parameter for chartist and fundamentalist strategies
	# Time dependent parameters
	ϵₜᶜ::Float64 # Chartist order size noise
	ϵₜᶠ::Float64 # Fundamentalist order size noise
	function JacobLealParameters(; Nᴸ = 10000, Nᴴ = 100, θ = 20, θ⁺ = 40, θ⁻ = 10, σ_chartist = 0.05,
		α_chartist = 0.04, σ_fundamentalist = 0.01, α_fundamentalist = 0.04, σʸ = 0.01,
		σᶻ = 0.01, δ = 0.0001, γᴸ = 20, γᴴ = 1, κ⁺ = 0.01, κ⁻ = 0, η⁺ = 0.2, η⁻ = 0,
		λ = 0.625, ζ = 1) # Constructor with default parameter values from Jacob-Leal
		if θ⁻ > θ⁺ # Lower bound must be less than upper bound
			error("Incorrect parameter values. Ensure θ⁻ < θ⁺")
		end
		if κ⁻ > κ⁺ # Lower bound must be less than upper bound
			error("Incorrect parameter values. Ensure κ⁻ < κ⁺")
		end
		if η⁻ > η⁺ # Lower bound must be less than upper bound
			error("Incorrect parameter values. Ensure η⁻ < η⁺")
		end
		if α_chartist < 0 || α_chartist > 1 || α_fundamentalist < 0 || α_fundamentalist > 1 # Constrained between 0 and 1
			error("Incorrect parameter values. Ensure 0 < α < 1")
		end
		if λ < 0 || λ > 1 # Constrained between 0 and 1
			error("Incorrect parameter values. Ensure 0 < λ < 1")
		end
		if δ < 0 # Constrained to be greater than 0
			error("Incorrect parameter values. Ensure δ > 0")
		end
		if ζ < 0 # Constrained to be greater than 0
			error("Incorrect parameter values. Ensure ζ > 0")
		end
		if γᴸ < γᴴ # HF orders expire faster than LF orders
			error("Incorrect parameter values. Ensure γᴸ > γᴴ")
		end
		if Nᴸ < 0 || Nᴴ < 0 # More LF agents than HF agents
			error("Incorrect parameter values. Ensure N > 0")
		end
		new(Nᴸ, Nᴴ, θ, θ⁺, θ⁻, σ_chartist, α_chartist, σ_fundamentalist, α_fundamentalist, σʸ, σᶻ, δ, γᴸ, γᴴ, κ⁺, κ⁻, η⁺, η⁻, λ, ζ)
	end
end
@enum Strategy begin
	chartist = 1
	fundamentalist = 2
end
mutable struct LowFrequency <: Agent
	type::Strategy # Chartist or Fundamentalist
	activated::Bool # Activated or not for next trading sesion (Not used ??CORRECT??)
	θ::Float64 # Agent trading frequency randomly sampled from truncated exponential (Not used ??CORRECT??)
	πₜᶜ::Float64 # Agent's chartist strategy profit at time t
	πₜᶠ::Float64 # Agent's fundamentalist strategy profit at time t
	ϕₜᶜ::Float64 # Probability of agent adopting chartist strategy at time t
	ϕₜᶠ::Float64 # Probability of agent adopting fundamentalist strategy at time t
	dₜᶜ::Float64 # Limit order size for chartist strategy placed at time t
	dₜᶠ::Float64 # Limit order size for fundamentalist strategy placed at time t
	pₜ::Float64 # Limit order price placed at time t
end
mutable struct HighFrequency <: Agent
	activated::Bool # Activated or not for next trading sesion
	θ::Float64 # Agent trading frequency randomly sampled from truncated exponential (Not used ??CORRECT??)
	πₜ::Float64 # Agent profit at time t (Not used ??CORRECT??)
	Δx::Float64 # Relative price difference activation threshold
	κ::Float64 # Order depth parameter
	dₜ::Float64 # Limit order size placed at time t
	pₜ::Float64 # Limit order price placed at time t
end
mutable struct LimitOrder
	periods_until_expiry::Int64
	size::Float64
	price::Float64
end
mutable struct LimitOrderBook
	bₜ::Float64 # Best bid
    aₜ::Float64 # Best ask
    bids::Vector{LimitOrder} # Stores all the active bids
    asks::Vector{LimitOrder} # Stores all the active asks
	fₜ::Float64 # Fundamental value of stock at time t
	fₜ₋₁::Float64 # Fundamental value of stock at time t - 1
	p̄::Vector{Float64} # List of closing prices
end
#---------------------------------------------------------------------------------------------------

#----- Results and tracing -----#
mutable struct Options <: SimulationOptions
	time::Int64 # Simulation time (number of trading sessions in days)
	p₀::Float64 # Initial price
	store_trace::Bool
	show_trace::Bool
	show_every::Int64
	seed::Int64
	function Options(; time::Int64 = 100, p₀::Float64 = 100.0, store_trace::Bool = false, show_trace::Bool = false, show_every::Int64 = 1, seed::Int64 = 1)
		new(time, p₀, store_trace, show_trace, show_every, seed)
	end
end
mutable struct State <: SimulationState
	t::Int64
	LOB_state::Dict{String, Float64}
	agents::Dict{String, Int64}
end
const Trace = Vector{State}
struct Results <: SimulationResults
	p̄::Vector{Float64}
	trace::Trace
end
function Trace!(trace::Trace, options::Options, t, agents, LOB)
	ask_volume = isempty(LOB.asks) ? 0.0 : sum(ask.size for ask in LOB.asks)
	bid_volume = isempty(LOB.bids) ? 0.0 : sum(bid.size for bid in LOB.bids)
	fundamentalists, chartists, highfrequency = 0, 0, 0
	for agent in agents
		if agent.activated
			if isa(agent, LowFrequency)
				if agent.type == fundamentalist
					fundamentalists += 1
				else
					chartists += 1
				end
			else
				highfrequency += 1
			end
		end
	end
	agents = Dict{String, Int64}("Fundamentalists" => fundamentalists, "Chartsists" => chartists, "High-Frequency Traders" => highfrequency)
	LOB_state = Dict{String, Float64}("price" => LOB.p̄[end], "value" => LOB.fₜ, "ask volume" => ask_volume, "bid volume" => bid_volume)
	state = State(t, LOB_state, agents)
	if options.store_trace
        push!(trace, state)
    end
	if options.show_trace
        if state.t % options.show_every == 0
			LOB_imbalance = round((bid_volume - ask_volume) / (bid_volume + ask_volume), digits = 3)
			@printf("%-5s    %-6d:%6d                  %-5d   %-6.2f          %-6.2f  %-4.3f\n",
			t, fundamentalists, chartists, highfrequency, round(LOB.p̄[end], digits = 2), round(LOB.fₜ, digits = 2), LOB_imbalance)
        end
    end
end
#---------------------------------------------------------------------------------------------------


#----- Simulation initialisation -----#
function InitialiseSimulation(parameters::JacobLealParameters, options::Options)::Tuple{Vector{Agent}, LimitOrderBook}
	agents = Vector{Agent}() # Initialization
	for i in 1:parameters.Nᴸ # Iterate and populate low-frequency agents
		type = rand() < 0.5 ? chartist : fundamentalist # Choose strategy with 50/50 probability
		θ = rand(Truncated(Exponential(parameters.θ), parameters.θ⁻, parameters.θ⁺)) # Sample trading speed from truncated exponential (Not used ??CORRECT??)
		push!(agents, LowFrequency(type, true, θ, 0, 0, 0.5, 0.5, 0, 0, 0)) # All agents activated at start
	end
	for j in 1:parameters.Nᴴ # Iterate and populate high-frequency agents
		θ = rand(Truncated(Exponential(parameters.θ), parameters.θ⁻, parameters.θ⁺)) # Sample trading speed from truncated exponential (Not used ??CORRECT??)
		Δx = rand(Uniform(parameters.η⁻, parameters.η⁺))
		κ = rand(Uniform(parameters.κ⁻, parameters.κ⁺))
		push!(agents, HighFrequency(true, θ, 0, Δx, κ, 0, 0)) # All agents actiivated at start
	end
	LOB = LimitOrderBook(options.p₀, options.p₀, Vector{LimitOrder}(), Vector{LimitOrder}(), options.p₀ - 5, options.p₀ - 5, [options.p₀, options.p₀ + 5]) # Initial prices and fundamental values ??CORRECT??
	return agents, LOB
end
#---------------------------------------------------------------------------------------------------

#----- Order creation -----#
# Set order sizes and prices
# Post orders to LOB
function SubmitOrder!(agent::LowFrequency, LOB::LimitOrderBook, parameters::JacobLealParameters)
	zₜ = rand(Normal(0, parameters.σᶻ))
	agent.dₜᶜ = parameters.α_chartist * (LOB.p̄[end] - LOB.p̄[end - 1]) + parameters.ϵₜᶜ # Chartist strategy limit order size
	agent.dₜᶠ = parameters.α_fundamentalist * (LOB.fₜ - LOB.p̄[end]) + parameters.ϵₜᶠ # Fundamentalist strategy limit order size
	dₜ = agent.type == chartist ? agent.dₜᶜ : agent.dₜᶠ # Choose order size corresponding to agent strategy
	agent.pₜ = LOB.p̄[end] * (1 + parameters.δ) * (1 + zₜ) # Limit order price
	order = LimitOrder(parameters.γᴸ, abs(dₜ), agent.pₜ) # Create order
	# Push order to LOB
	if dₜ < 0 # Sell order
		push!(LOB.asks, order)
	elseif dₜ > 0 # Buy order
		push!(LOB.bids, order)
	end
end
function SubmitOrder!(agent::HighFrequency, LOB::LimitOrderBook, parameters::JacobLealParameters)
	if agent.activated
		if rand() < 0.5 # Buy order
			if !isempty(LOB.asks) # Only submit order if there is liquidity on opposite side
				# Sample order size from Poisson with mean depending on mean volume on opposite side tuncated from above by the total volume on the opposite side
				total_ask_volume = sum(order.size for order in LOB.asks)
				λ = total_ask_volume / length(LOB.asks)
				agent.dₜ = abs(rand(truncated(Poisson(λ), upper = total_ask_volume)))
				agent.pₜ = LOB.bₜ *  (1 - agent.κ)
				if agent.dₜ > 0 # Only post if order size > 0
					push!(LOB.bids, LimitOrder(parameters.γᴴ, agent.dₜ, agent.pₜ))
				end
			end
		else # Sell order
			if !isempty(LOB.bids) # Only submit order if there is liquidity on opposite side
				# Sample order size from Poisson with mean depending on mean volume on opposite side tuncated from above by the total volume on the opposite side
				total_bid_volume = sum(order.size for order in LOB.bids)
				λ = total_bid_volume / length(LOB.bids)
				agent.dₜ = abs(rand(truncated(Poisson(λ), upper = total_bid_volume)))
				agent.pₜ = LOB.aₜ *  (1 + agent.κ)
				if agent.dₜ > 0  # Only post if order size > 0
					push!(LOB.asks, LimitOrder(parameters.γᴴ, agent.dₜ, agent.pₜ))
				end
			end
		end
	end
end
#---------------------------------------------------------------------------------------------------

#----- Start of session -----#
# Set fundamental market value
# Update time-dependent parameters
# Update LOB parameters
function PreOpen!(LOB::LimitOrderBook, parameters::JacobLealParameters)
	parameters.ϵₜᶜ = rand(Normal(0, parameters.σ_chartist))
	parameters.ϵₜᶠ = rand(Normal(0, parameters.σ_fundamentalist))
	yₜ = rand(Normal(0, parameters.σʸ))
	LOB.fₜ₋₁ = LOB.fₜ
	LOB.fₜ = LOB.fₜ₋₁ * (1 + parameters.δ) * (1 + yₜ)
end
#---------------------------------------------------------------------------------------------------

#----- Switch low-frequency trader strategy -----#
# Adapt trader strategies for session based on profit from previous session and strategy switching probability
function AdaptStrategy!(agent::LowFrequency, LOB::LimitOrderBook)
	agent.type = rand() < agent.ϕₜᶜ ? chartist : fundamentalist
end
function AdaptStrategy!(agent::HighFrequency, LOB::LimitOrderBook)
	agent.activated = abs((LOB.p̄[end] -  LOB.p̄[end - 1]) / LOB.p̄[end - 1]) > agent.Δx # Activate HF agent if relative price difference exceeds threshold
end
#---------------------------------------------------------------------------------------------------

#----- End of session -----#
# Calculate profits
# Expire LOB orders
function PostClose!(agents::Vector{Agent}, LOB::LimitOrderBook, parameters::JacobLealParameters)
	for agent in agents
		if isa(agent, LowFrequency) # Compute LF agent profitability
			# LF agents must compute profits for both strategies (whether adopted or not) ??CORRECT??
			agent.πₜᶠ = (LOB.p̄[end] - agent.pₜ) * abs(agent.dₜᶠ)
			agent.πₜᶜ = (LOB.p̄[end] - agent.pₜ) * abs(agent.dₜᶜ)
			agent.ϕₜᶜ = exp(agent.πₜᶜ / parameters.ζ) / (exp(agent.πₜᶜ / parameters.ζ) + exp(agent.πₜᶠ / parameters.ζ))
			agent.ϕₜᶠ = 1 - agent.ϕₜᶜ
		else # Compute HF agent profitability
			agent.πₜ = (LOB.p̄[end] - agent.pₜ) * agent.dₜ # (Not used ??CORRECT??)
		end
	end
	expired_orders = Vector{Int64}()
	for (index, order) in enumerate(LOB.asks)
		order.periods_until_expiry -= 1
		if order.periods_until_expiry == 0
			push!(expired_orders, index)
		end
	end
	deleteat!(LOB.asks, expired_orders)
	expired_orders = Vector{Int64}()
	for (index, order) in enumerate(LOB.bids)
		order.periods_until_expiry -= 1
		if order.periods_until_expiry == 0
			push!(expired_orders, index)
		end
	end
	deleteat!(LOB.bids, expired_orders)
end
#---------------------------------------------------------------------------------------------------

#----- Closing Auction -----#
function ClosingAuction!(LOB::LimitOrderBook)
	if !(isempty(LOB.bids) || isempty(LOB.asks))
		# Sort by price
		sort!(LOB.asks, by = x -> x.price)
		sort!(LOB.bids, by = x -> x.price, rev = true)
		LOB.aₜ, LOB.bₜ = LOB.asks[1].price, LOB.bids[1].price # Set best bid and best ask
		closing_price = 0
		while LOB.aₜ <= LOB.bₜ # Match order until none cross the spread
			ask_size, bid_size = LOB.asks[1].size, LOB.bids[1].size # Get order sizes on TOB
			if bid_size > ask_size # Bid is partially filled
				bid_size -= ask_size
				LOB.bids[1].size = bid_size
				popfirst!(LOB.asks) # Remove matched orders from TOB on ask side
			elseif bid_size < ask_size # Ask is partially filled
				ask_size -= bid_size
				LOB.asks[1].size = ask_size
				popfirst!(LOB.bids) # Remove matched orders from TOB on bid side
			else # Both sides are completely filled
				# Remove matched orders from TOB on both sides
				popfirst!(LOB.asks)
				popfirst!(LOB.bids)
			end
			closing_price = (LOB.aₜ + LOB.bₜ) / 2 # Compute mid-price
			if isempty(LOB.asks) || isempty(LOB.bids)
				break
			end
			LOB.aₜ, LOB.bₜ = LOB.asks[1].price, LOB.bids[1].price # Update best ask and best bid
		end
		# Append new closing price
		if closing_price > 0
			push!(LOB.p̄, closing_price)
		else
			push!(LOB.p̄, LOB.p̄[end])
		end
	else
		push!(LOB.p̄, LOB.p̄[end])
	end
end
#---------------------------------------------------------------------------------------------------

#----- Model -----#
function Simulate(parameters::JacobLealParameters = JacobLealParameters(); options::Options = Options())::Results
	Random.seed!(options.seed)
	tracing = options.store_trace || options.show_trace
	trace = Trace()
	agents, LOB = InitialiseSimulation(parameters, options) # Initialise agents and parameters
	if options.show_trace # Print header
        println("t        Fundamentalists:Chartists\tNᴴ\tpₜ\t\tfₜ\tLOB Imbalance")
		println("-------- ------------------------------ ------- --------------- ------- -------------")
    end
	Trace!(trace, options, 0, agents, LOB)
	for t in 1:options.time # Iterate through days/sessions
		PreOpen!(LOB, parameters) # Update stock fundamental value and parameters for session
		for agent in agents
			AdaptStrategy!(agent, LOB) # Switch strategy for current session based on previous session profits
			SubmitOrder!(agent, LOB, parameters) # Post order to LOB
		end
		ClosingAuction!(LOB) # Execute order matching for current trading session
		PostClose!(agents, LOB, parameters) # Calculate agent profit and probability of switching strategies
		if tracing
			Trace!(trace, options, t, agents, LOB)
		end
	end
	return Results(LOB.p̄, trace)
end
#---------------------------------------------------------------------------------------------------

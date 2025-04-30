using StatsBase
using Statistics
using Plots
using Random
using CSV
using GLM
using LinearAlgebra
using HypothesisTests
using LaTeXStrings
using SpecialFunctions
using SummaryTables 
using StatsPlots
using Clustering
using DataFrames
using TidierData
using Revise
using ProductPartitionModels

include("simFunctions.jl")

# read arguments from command line
# N, nc, variance, interEffect, common, xdiff

N = parse(Int, ARGS[1])
nc =parse(Int, ARGS[2])
variance = parse(Float64, ARGS[3])
interEffect = parse(Float64, ARGS[4])
common = parse(Float64, ARGS[5])
xdiff=parse(Float64, ARGS[6])
dims=parse(Int, ARGS[7])

# construct a file name from the user inputs
outputName = "results/c$(nc)_N$(N)_inter$(interEffect)_common$(common)_xd$(xdiff)_v$(variance)_dim$(dims)"

# END user input

reps = 25
niters=4000
rng = Random.MersenneTwister(0)
fractions = repeat([1/nc], nc)

# END other controls


results = Vector{DataFrame}(undef, reps)
seeds = MersenneTwister.(rand(1:10^8, Threads.nthreads()))  # or generate from original rng
if !isfile("$(outputName).csv")
    # n,  fractions, variance, interEffect, common
    Threads.@threads for i in 1:reps
        try
           println(i)
           results[i] = simExperiment(seeds[Threads.threadid()]; N=N, fractions=fractions, variance = variance, interEffect = interEffect, common = common , niters=niters, plotSim =  false, xdiff = xdiff, dims = dims)
        catch err
            println("sim Failed")
        end
    end   
    defined_results = [results[i] for i in 1:reps if isassigned(results, i)]
    df = vcat(defined_results...)
    CSV.write("$(outputName).csv", df, writeheader = true, append = true)
end

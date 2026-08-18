# Generate representative covariate distributions (X1..X5 by group) across the
# xdiff grid used in the dim5xdiff simulations, for a supplement figure.
#
# Usage: julia --project=. covDist.jl

using Random
using Distributions
using DataFrames
using LinearAlgebra
using CSV
using GLM
using StatsModels
using StatsBase
using Statistics
using StatsPlots
using Clustering
using TidierData
using Plots
using LaTeXStrings
using SpecialFunctions
using SummaryTables
using HypothesisTests
using ProductPartitionModels
using DPMM

include("simFunctions.jl")

const N = 1000
const nc = 8
const DIMS = 5
const XDIFFS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
const SEED = 42

rng = MersenneTwister(SEED)
fractions = repeat([1 / nc], nc)

rows = DataFrame[]
for xd in XDIFFS
    df = simData(rng; N=N, fractions=fractions, variance=0.05,
                 interEffect=1.0, common=1.0, xdiff=xd, dims=DIMS)
    out = DataFrame(xdiff=xd, group=df.group)
    for d in 1:DIMS
        out[!, "X$d"] = df[!, "X$d"]
    end
    push!(rows, out)
end

res = vcat(rows...)
CSV.write("results/covariates.csv", res)
println("wrote results/covariates.csv with ", nrow(res), " rows")
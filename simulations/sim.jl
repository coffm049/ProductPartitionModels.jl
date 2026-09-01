using StatsBase
using Statistics
using Distributions
using Plots
using Random
using CSV
using GLM
using StatsModels
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
using DPMM

include("simFunctions.jl")
include("salsoUtils.jl")

# read arguments from command line
# N, nc, variance, interEffect, common, xdiff
N = parse(Int, ARGS[1])
nc = parse(Int, ARGS[2])
variance = parse(Float64, ARGS[3])
interEffect = parse(Float64, ARGS[4])
common = parse(Float64, ARGS[5])
xdiff = parse(Float64, ARGS[6])
dims = parse(Int, ARGS[7])
prec = parse(Float64, ARGS[8])
alph = parse(Float64, ARGS[9])
bet = parse(Float64, ARGS[10])
reps = parse(Int, ARGS[11])
niters = parse(Int, ARGS[12])
massa = parse(Float64, ARGS[13])
massb = parse(Float64, ARGS[14])
# optional DPM baseline arguments (default off so existing invocations are unchanged)
runDPM = length(ARGS) >= 15 ? parse(Int, ARGS[15]) : 0
DPMalpha = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : 1.0
DPMiters = length(ARGS) >= 17 ? parse(Int, ARGS[17]) : 500
# optional imbalanced-clusters flag (1 => exponential-decay cluster sizes, Imbal_ file prefix)
imbalanced = length(ARGS) >= 18 ? parse(Int, ARGS[18]) : 0

# construct a file name from the user inputs
baseName = "N$(N)_c$(nc)_inter$(interEffect)_common$(common)_xd$(xdiff)_v$(variance)_dim$(dims)_prec$(prec)alph$(alph)bet$(bet)_mass$(massa)$(massb)"
if imbalanced == 1
    baseName = "Imbal_" * baseName
end
outputName = "results/" * baseName
mkpath("results")
if runDPM == 1
    outputName = outputName * "_dpm$(DPMalpha)_$(DPMiters)"
end

# END user input
#
fractions = imbalanced == 1 ? exp.(-collect(0:(nc-1)) .* 0.8) ./ sum(exp.(-collect(0:(nc-1)) .* 0.8)) : repeat([1 / nc], nc)

# END other controls


results = Vector{DataFrame}(undef, reps)
CmixAll = Vector{Any}(undef, reps)
CdpmAll = Vector{Any}(undef, reps)
CmixoosAll = Vector{Any}(undef, reps)
CdpmoosAll = Vector{Any}(undef, reps)
truthAll = Vector{Any}(undef, reps)
truthoosAll = Vector{Any}(undef, reps)
seeds = MersenneTwister.(rand(1:10^8, Threads.nthreads()))  # or generate from original rng
# n,  fractions, variance, interEffect, common
Threads.@threads for i in 1:reps
    try
        println(i)
        out = simExperiment(seeds[Threads.threadid()]; N=N, fractions=fractions, variance=variance, interEffect=interEffect, common=common, niters=niters, plotSim=false, xdiff=xdiff, dims=dims, prec=prec, alph=alph, bet=bet, massParams = [massa, massb], runDPM=(runDPM == 1), DPMalpha=DPMalpha, DPMiters=DPMiters, returnC=true)
        results[i] = out.result
        CmixAll[i] = out.Cmix
        CdpmAll[i] = out.Cdpm
        CmixoosAll[i] = out.Cmixoos
        CdpmoosAll[i] = out.Cdpmoos
        truthAll[i] = out.truth
        truthoosAll[i] = out.truthoos
    catch err
        println("sim Failed")
    end
end
defined_results = [results[i] for i in 1:reps if isassigned(results, i)]
df = vcat(defined_results...)
definedIdx = findall(i -> isassigned(results, i), 1:reps)

# SALSO point estimates (Binder + VI) - needs R on the main thread, so run
# after the threaded loop. ARI reported for the PPMx (mixDPM) models only.
salsoBinderARI_Mix = Vector{Union{Missing,Float64}}(undef, reps)
salsoVIARI_Mix = Vector{Union{Missing,Float64}}(undef, reps)
salsoBinderARI_Mixoos = Vector{Union{Missing,Float64}}(undef, reps)
salsoVIARI_Mixoos = Vector{Union{Missing,Float64}}(undef, reps)
salsoBinderARI_DPM = Vector{Union{Missing,Float64}}(undef, reps)
salsoVIARI_DPM = Vector{Union{Missing,Float64}}(undef, reps)
salsoBinderARI_DPMoos = Vector{Union{Missing,Float64}}(undef, reps)
salsoVIARI_DPMoos = Vector{Union{Missing,Float64}}(undef, reps)
for i in 1:reps
    if !isassigned(CmixAll, i)
        salsoBinderARI_Mix[i] = missing
        salsoVIARI_Mix[i] = missing
        salsoBinderARI_Mixoos[i] = missing
        salsoVIARI_Mixoos[i] = missing
        salsoBinderARI_DPM[i] = missing
        salsoVIARI_DPM[i] = missing
        salsoBinderARI_DPMoos[i] = missing
        salsoVIARI_DPMoos[i] = missing
        continue
    end
    salsoBinderARI_Mix[i] = salso_ari(CmixAll[i], truthAll[i]; loss=:binder).ari
    salsoVIARI_Mix[i] = salso_ari(CmixAll[i], truthAll[i]; loss=:VI).ari
    salsoBinderARI_Mixoos[i] = salso_ari(CmixoosAll[i], truthoosAll[i]; loss=:binder).ari
    salsoVIARI_Mixoos[i] = salso_ari(CmixoosAll[i], truthoosAll[i]; loss=:VI).ari
    salsoBinderARI_DPM[i] = salso_ari(CdpmAll[i], truthAll[i]; loss=:binder).ari
    salsoVIARI_DPM[i] = salso_ari(CdpmAll[i], truthAll[i]; loss=:VI).ari
    salsoBinderARI_DPMoos[i] = salso_ari(CdpmoosAll[i], truthoosAll[i]; loss=:binder).ari
    salsoVIARI_DPMoos[i] = salso_ari(CdpmoosAll[i], truthoosAll[i]; loss=:VI).ari
end
df.salsoBinderARI_Mix = salsoBinderARI_Mix[definedIdx]
df.salsoVIARI_Mix = salsoVIARI_Mix[definedIdx]
df.salsoBinderARI_Mixoos = salsoBinderARI_Mixoos[definedIdx]
df.salsoVIARI_Mixoos = salsoVIARI_Mixoos[definedIdx]
df.salsoBinderARI_DPM = salsoBinderARI_DPM[definedIdx]
df.salsoVIARI_DPM = salsoVIARI_DPM[definedIdx]
df.salsoBinderARI_DPMoos = salsoBinderARI_DPMoos[definedIdx]
df.salsoVIARI_DPMoos = salsoVIARI_DPMoos[definedIdx]

CSV.write("$(outputName).csv", df, writeheader=true, append=false)

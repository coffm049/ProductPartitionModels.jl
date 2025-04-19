using StatsBase
using DataFrames
using Plots
using CSV

df = DataFrame(CSV.File("4clustResults.csv"))

bins = [0:0.05:1;]


# rand Index
p1 = histogram(df.rindMix, label = "Mix", fillopacity = 0.33, bins = bins)
histogram!(df.rindDPM, label = "DPM", fillopacity = 0.33, bins = bins)
p2 = histogram(df.clusterPvalue, label = "Mix > DPM", fillopacity = 0.33, bins = bins)
p = plot(p1, p2, layout = (2,1))

# RMSE
p1 = histogram(df.midMix, label = "Mix", fillopacity = 0.33)
histogram!(df.midDPM, label = "DPM", fillopacity = 0.33)

p2 = histogram(df.rmsePmix, label = "Mix", fillopacity = 0.33)
histogram!(df.rmsePdpm, label = "DPM", fillopacity = 0.33)

# common effects
p1 = histogram(df.priorBeta1, label = "Mix", fillopacity = 0.33)
histogram!(df.priorBeta2, label = "DPM", fillopacity = 0.33)


# testMCMC.jl

using ProductPartitionModels
using StatsBase


n = 100
p = 5
prop_mis = 0.5
nmis = Int(floor(prop_mis*n*p))
nobs = n*p - nmis
X = Matrix{Union{Missing, Float64}}(missing, n, p)
obs_indx_sim = sample(1:(n*p), nobs; replace=false)
X[obs_indx_sim] = randn(nobs)
size(X)
X

Ctrue = vcat(fill(1, Int(floor(.5*n))), fill(2, Int(floor(.3*n))), fill(3, Int(floor(.2*n))))
length(Ctrue) == n
for i in findall(Ctrue .== 2)
    X[i,1:2] += [3.0, -2.0]
end
for i in findall(Ctrue .== 3)
  X[i,1:2] += [-1.0, 2.0]
end
X

α = 0.5
logα = log(α)
cohesion = Cohesion_CRP(logα, 0, true)
# similarity = Similarity_NiG_indep(0.0, 0.1, 1.0, 1.0)
similarity = Similarity_NiG_indep(0.0, 0.5, 4.0, 4.0)
C = deepcopy(Ctrue)
# C, K, S, lcohes, Xstat, lsimilar = sim_partition_PPMx(logα, X, similarity)
# C
# K
# S

K = maximum(C)
lcohes, Xstat, lsimilar = get_lcohlsim(C, X, cohesion, similarity)

## G0; controls only y|x
μ0 = 0.0
σ0 = 20.0
τ0 = 2.0 # scale of DL shrinkage
upper_σ = 3.0
G0 = Baseline_NormDLUnif(μ0, σ0, τ0, upper_σ)

y, μ, β, σ = sim_lik(C, X, similarity, Xstat, G0)

y
μ
β
σ

# mod = Model_PPMx(y, X, C)
mod = Model_PPMx(y, X, 0) # C_init = 0 --> n clusters ; 1 --> 1 cluster
fieldnames(typeof(mod))
fieldnames(typeof(mod.state))
mod.state.C

mod.state.baseline = deepcopy(G0)
mod.state.cohesion = deepcopy(cohesion)
mod.state.similarity = deepcopy(similarity)

refresh!(mod.state, mod.y, mod.X, mod.obsXIndx, true)
mod.state.llik

mcmc!(mod, 1000,
    save=false,
    thin=1,
    n_procs=1,
    report_filename="out_progress.txt",
    report_freq=100,
    update=[:C, :lik_params],
    monitor=[:C, :mu, :sig, :beta]
)

sims = mcmc!(mod, 1000,
    save=true,
    thin=1,
    n_procs=1,
    report_filename="out_progress.txt",
    report_freq=100,
    update=[:C, :lik_params],
    monitor=[:C, :mu, :sig, :beta]
)


sims[1]
sims[1000]
sims[2][:lik_params][2]
mod.state.lik_params[1].mu

sims_llik = [ sims[ii][:llik] for ii in 1:length(sims) ]
sims_K = [ maximum(sims[ii][:C]) for ii in 1:length(sims) ]
Kmax = maximum(sims_K)
sims_S = permutedims( hcat( [ counts(sims[ii][:C], Kmax) for ii in 1:length(sims) ]...) )
sims_Sord = permutedims( hcat( [ sort(counts(sims[ii][:C], maximum(sims_K)), rev=true) for ii in 1:length(sims) ]...) )

using Plots
Plots.PlotlyBackend()


plot(sims_llik)
plot(sims_K)
plot(sims_S)

[ sims[ii][:C][54] for ii in 1:length(sims) ]

## monitoring lik_params is only useful if C is not changing
Kuse = 3

sims_mu = [ sims[ii][:lik_params][kk][:mu] for ii in 1:length(sims), kk in 1:Kuse ]
plot(sims_mu)
plot(sims_mu[:,1])
μ

sims_sig = [ sims[ii][:lik_params][kk][:sig] for ii in 1:length(sims), kk in 1:Kuse ]
plot(sims_sig)
plot(sims_sig[:,1])
σ

sims_beta = [ sims[ii][:lik_params][kk][:beta][j] for ii in 1:length(sims), kk in 1:Kuse, j in 1:p ]
plot(reshape(sims_beta[:,1,:], (length(sims), p)))
plot(reshape(sims_beta[:,2,:], (length(sims), p)))
plot(reshape(sims_beta[:,3,:], (length(sims), p)))
plot(reshape(sims_beta[:,4,:], (length(sims), p)))
β

plot(reshape(sims_beta[:,3,2], (length(sims))))



using Plotly # run pkg> activate to be outside the package

C_use = deepcopy(C)
C_use = deepcopy(mod.state.C)

indx_cc = findall( [ all(.!ismissing.(X[i,1:2])) for i in 1:n ] )
indx_x1m = findall( [ ismissing(X[i,1]) & !ismissing(X[i,2]) for i in 1:n ] )
indx_x2m = findall( [ !ismissing(X[i,1]) & ismissing(X[i,2]) for i in 1:n ] )
indx_allmiss = findall( [ all(ismissing.(X[i,:])) for i in 1:n ] )

colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

trace1 = Plotly.scatter3d(Dict(
  :x => convert(Vector{Float64}, X[indx_cc,1]),
  :y => convert(Vector{Float64}, X[indx_cc,2]),
  :z => y[indx_cc],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_use[indx_cc]], :size => 5.0)
))
Plotly.plot([trace1])

trace2 = Plotly.scatter3d(Dict(
  :x => convert(Vector{Float64}, X[indx_x2m,1]),
  :y => zeros(length(indx_x2m)),
  :z => y[indx_x2m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_use[indx_x2m]], :size => 5.0)
))
plot([trace2])

trace3 = Plotly.scatter3d(Dict(
  :x => zeros(length(indx_x2m)),
  :y => convert(Vector{Float64}, X[indx_x1m,2]),
  :z => y[indx_x1m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_use[indx_x1m]], :size => 5.0)
))
plot([trace3])


trace4 = Plotly.scatter3d(Dict(
  :x => zeros(length(indx_allmiss)),
  :y => zeros(length(indx_allmiss)),
  :z => y[indx_allmiss],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_use[indx_allmiss]], :size => 5.0)
))
plot([trace4])




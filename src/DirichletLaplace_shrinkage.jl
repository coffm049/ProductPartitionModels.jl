# DirichletLaplace_shrinkage.jl

# export ;

# using Distributions
# include("generalizedinversegaussian.jl")

mutable struct DirLap{T <: Real}
    K::Int # in the paper this is n
    α::T # in the paper this is a
    β::Vector{T} # in the paper this is theta
    ψ::Vector{T}
    ϕ::Vector{T}
    τ::T
end

function update_ψ(ϕ::Vector{T}, β::Vector{T}, τ::T) where T <: Real

    μ = τ .* ϕ ./ abs.(β)
    ψinv = [ rand( InverseGaussian(μ[k]) ) for k in 1:length(μ) ]

    ψ = 1.0 ./ ψinv

    inftest = ψ .== Inf
    if any(inftest)
        indx = findall(inftest)
        ψ[indx] .= exp(500.0) # cheater's method
    end

    return ψ
end

function update_τ(ϕ::Vector{T}, β::Vector{T}, α::T) where T <: Real

    K = length(β)
    p = float(K)*(α - 1.0)
    b = 2.0 * sum( abs.(β) ./ ϕ )

    return rand( GeneralizedInverseGaussian(1.0, b, p) )
end

function update_ϕ(β::Vector{T}, α::T) where T<: Real

    p = α - 1.0
    b = 2.0 .* abs.(β)
    Tvec = [ rand( GeneralizedInverseGaussian(1.0, b[k], p) ) for k in 1:length(β) ]

    return Tvec ./ sum(Tvec)
end

function rpost_normlmDiagBeta_beta1(y::Vector{T}, X::Matrix{T},
  σ2::T, Vdiag::Vector{T}, β0::T=0.0) where T <: Real

  σ2 > 0.0 || throw(ArgumentError("σ2 must be positive."))

  n,p = size(X) # assumes X was a matrix
  length(y) == n || throw(ArgumentError("y and X dimension mismatch"))
  ystar = y .- β0

  A = Diagonal(1.0 ./ Vdiag) + X'X ./ σ2
  U = (cholesky(A)).U
  Ut = transpose(U)

  # μ_a = At_ldiv_B(U, (X'ystar/σ2))
  μ_a = Ut \ (X'ystar ./ σ2)
  μ = U \ μ_a

  z = randn(p)
  β = U \ z + μ

  zerotest = β .== 0.0
  if any(zerotest)
      indx = findall(zerotest)
      β[indx] .= exp(-500.0) # cheater's method
  end

  return β
end


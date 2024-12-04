#using Distributions


export sample_totalMass

# following Mulelr 2015 Bayesian Nonparmetric data analysis
# to update total mass parameter (M) for DP
# page 19 section 2.4.2
"""
    update_totalMass(M, n, nclusts, a, b)

assuming 
M ~ Ga(a,b)
p(M|ϕ ,k) = πGa(a + k, b-log(ϕ)) + (1-π)Ga(a+k-1, b-log(ϕ))

return: M
"""
function sample_totalMass(
  M::Int, 
  n::Int,
  nclusts::Int,
  a::Real,
  b::Real)

  ϕ = rand(Beta(M+1, n))
  π = (a+nclusts-1)/(n * (b- log(ϕ)) + a + nclusts - 1)
  
  if rand(Bernoulli(π))
    newM = Gamma(a+ nclusts, b - log(ϕ))
  else 
    newM = Gamma(a+ nclusts - 1, b - log(ϕ))
  end
  return rand(newM)
end

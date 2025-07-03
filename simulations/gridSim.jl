using LinearAlgebra   # for `norm`

"""
    grid_maximin(K, p; m = nothing)

Return a `K × p` matrix whose rows are points in `[0,1]^p`.

* Build an `m^p` Cartesian grid, where `m = ceil(K^(1/p))` unless the
  caller supplies `m` explicitly.
* Pick the grid node closest to the hyper-cube centre `(0.5,…,0.5)`
  as the first point.
* Repeatedly select the grid node whose **nearest** distance to the
  already–chosen set is **maximal** (deterministic farthest-point rule).

For moderate `p ≤ 6` and `K ≤ 500` the whole procedure runs in a few
seconds on a laptop.
"""
function grid_maximin(K::Int, p::Int; m::Union{Int,Nothing}=nothing)

    @assert K ≥ 1 && p ≥ 1
    m = isnothing(m) ? ceil(Int, K^(1 / p)) : m
    # 1 ───────── grid coordinates along each axis
    coords = ((0:m-1) .+ 0.5) ./ m                     # centres of cells
    # Cartesian product of axes → all grid nodes
    grid_iter = Iterators.product(ntuple(_ -> coords, p)...)
    grid = collect(grid_iter)                     # Vector of NTuples
    ngrid = length(grid)
    @assert K ≤ ngrid "grid too coarse; increase m or lower K"

    # 2 ───────── choose the seed (closest to centre)
    centre = ntuple(_ -> 0.5, p)
    seed_idx = argmin(map(pt -> sum((pt .- centre) .^ 2), grid))
    design = Vector{NTuple{p,Float64}}(undef, K)
    design[1] = grid[seed_idx]

    # min-squared-distance from every grid node to current design
    min_d2 = map(pt -> sum((pt .- design[1]) .^ 2), grid)

    # 3 ───────── greedy farthest-first selection
    for k in 2:K
        # pick the candidate whose nearest‐neighbour distance is largest
        next_idx = argmax(min_d2)
        design[k] = grid[next_idx]

        # update min-distance table with the new point
        newpt = design[k]
        for j in 1:ngrid
            d2 = sum((grid[j] .- newpt) .^ 2)
            if d2 < min_d2[j]
                min_d2[j] = d2
            end
        end
    end

    # convert Vector{NTuple} → K×p Matrix
    return reduce(vcat, (collect(x)' for x in design))
end


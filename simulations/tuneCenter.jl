using Distributions
using LinearAlgebra
using DataFrames
using Random
#using UnicodePlots
using Tables
using Plots



function independent_sampler(X, mu0, kappa0, alpha0, beta0, nsamples)
    n, p = size(X)  # Number of observations and dimensions
    X_bar = mean(X, dims=1)  # Sample means for each dimension

    # Storage for samples
    mu_samples = Matrix{Float64}(undef, p, nsamples)
    sigma2_samples = Matrix{Float64}(undef, p, nsamples)

    for j in 1:p
        # Data for the j-th dimension
        xj = X[:, j]

        # Posterior hyperparameters for the j-th dimension
        kappa_n = kappa0[j] + n
        mu_n = (kappa0[j] * mu0[j] + n * X_bar[j]) / kappa_n
        alpha_n = alpha0[j] + n / 2
        beta_n = beta0[j] + 0.5 * sum((xj .- X_bar[j]).^2) +
                 (kappa0[j] * n * (X_bar[j] - mu0[j])^2) / (2 * kappa_n)

        for i in 1:nsamples
            # Sample σ_j^2 from Inverse-Gamma
            sigma2 = rand(InverseGamma(alpha_n, beta_n))
            sigma2_samples[j, i] = sigma2

            # Sample μ_j from Normal
            mu = rand(Normal(mu_n, sqrt(sigma2 / kappa_n)))
            mu_samples[j, i] = mu
        end
    end

    return mu_samples, sigma2_samples
end

function simMSE(X, mu0, kap, alph, bet, nsamples)
    dim = size(X)[2]
    mu_samples, sigma2_samples = independent_sampler(
    X, repeat([mu0], dim), repeat([kap], dim), repeat([alph], dim), repeat([bet], dim), nsamples)
    mse = mean((mu_samples .- true_mu) .^2, dims = 2)[1]
    mse2 = mean((sigma2_samples .- true_sigma2) .^2, dims = 2)[1]
  return mse, mse2
end

# Simulated data
Random.seed!(123)
N, p = 7, 3
true_mu = repeat([1.0], p)
true_sigma2 = repeat([0.5], p)
# X = hcat([rand(Normal(true_mu[j], sqrt(true_sigma2[j])), N) for j in 1:p]...)
X = [quantile(Normal(true_mu[p], true_sigma2[p]), i/(N + 1)) for i in 1:N, j in 1:p]

# Prior hyperparameters
mu0 = [0.0]  # Prior mean
kappa0 = [0.1]  # Prior precision variance is the 1/0.1 = 10
alpha0 = [0.1, 1.0]  # Prior shape for σ^2
beta0 = [0.01, 0.1, 0.5, 1.0, 2.0]  # Prior scale for σ^2 for alph 0.1


# Number of posterior samples
nsamples = 100

# loop over all combinations ofr priors in for loop 
# use the independent sampler given those priors then add 
# MSE to store vector
experimentDF = DataFrame(collect(Base.Iterators.product(mu0, kappa0, alpha0, beta0)), [:priorμ, :priorκ, :priorα, :priorβ])
test = Tables.matrix([simMSE(X, row.priorμ, row.priorκ, row.priorα, row.priorβ, nsamples) for row in eachrow(experimentDF)])
experimentDF[!, :mse] = test[:,1]
experimentDF[!, :mse2] = test[:,2]


# scatterplot(experimentDF.priorκ, experimentDF.mse) # 1.0
# scatterplot(experimentDF.priorα, experimentDF.mse) # 5.0
# scatterplot(experimentDF.priorβ, experimentDF.mse) # 1.0
# scatterplot(experimentDF.priorκ, experimentDF.mse2) # 1.0
# scatterplot(experimentDF.priorα, experimentDF.mse2) # 5.0
# scatterplot(experimentDF.priorβ, experimentDF.mse2) # 1.0
scatter(experimentDF.priorκ, experimentDF.mse) # 1.0
scatter(experimentDF.priorα, experimentDF.mse) # 3.0
scatter(experimentDF.priorβ, experimentDF.mse) # 1.0
scatter(experimentDF.priorκ, experimentDF.mse2) # 1.0
scatter(experimentDF.priorα, experimentDF.mse2) # 3.0
scatter(experimentDF.priorβ, experimentDF.mse2) # 1.0

# resulting prior distribution
mud = Normal(0, 1.0)
sigd = InverseGamma(1, 1.0)
x = -4:0.01:4
plot(x, pdf.(mud, x), title = "center prior")
plot(abs.(x), pdf.(sigd, abs.(x)), title = "spread prior")

# Example of the fit
mu_samples, sigma2_samples = independent_sampler(
X, repeat([0], 3), repeat([1.0], 3), repeat([1.0], 3), repeat([1.0], 3), nsamples)
# Inspect results
p1 = histogram(mu_samples[1,:], label = "sample", title = "Center")
vline!([true_mu[1], mean(mu_samples[1,:])], label = ["true" "mean"], color = ["black" "red"])
p2 = histogram(sigma2_samples[1,:], label = "sample", title = "Variance")
vline!([true_sigma2[1], mean(sigma2_samples[1,:])], label = ["true" "mean"], color = [:black :red])
plot(p1, p2)
# p2 = histogram(mu_samples[2,:], label = nothing)
# vline!([true_mu[2]], label = nothing)
# vline!([true_mu[3]], label = nothing)
# p = plot(p1, p2, p3, layout=(3,1))
# save plot
savefig(p, "NormIGverification.png")


--- Startup times for process: Primary/TUI ---

times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.005  000.005: --- NVIM STARTING ---
000.512  000.507: event init
000.742  000.230: early init
000.870  000.129: locale set
001.034  000.164: init first window
001.675  000.640: inits 1
001.702  000.027: window checked
001.846  000.144: parsing arguments
003.085  000.089  000.089: require('vim.shared')
003.306  000.104  000.104: require('vim.inspect')
003.432  000.094  000.094: require('vim._options')
003.436  000.338  000.141: require('vim._editor')
003.440  000.509  000.082: require('vim._init_packages')
003.444  001.089: init lua interpreter
005.186  001.742: --- NVIM STARTED ---

--- Startup times for process: Embedded ---

times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.005  000.005: --- NVIM STARTING ---
000.438  000.433: event init
000.683  000.246: early init
000.818  000.134: locale set
000.961  000.143: init first window
001.724  000.763: inits 1
001.769  000.044: window checked
001.958  000.189: parsing arguments
003.248  000.087  000.087: require('vim.shared')
003.464  000.104  000.104: require('vim.inspect')
003.593  000.097  000.097: require('vim._options')
003.597  000.337  000.136: require('vim._editor')
003.600  000.501  000.077: require('vim._init_packages')
003.605  001.146: init lua interpreter
003.773  000.169: expanding arguments
003.826  000.053: inits 2
004.216  000.390: init highlight
004.218  000.002: waiting for UI
004.381  000.162: done waiting for UI
004.385  000.004: clear screen
004.508  000.012  000.012: require('vim.keymap')
005.597  001.208  001.197: require('vim._defaults')
005.600  000.007: init default mappings & autocommands
006.202  000.064  000.064: sourcing /opt/nvim-linux64/share/nvim/runtime/ftplugin.vim
006.283  000.032  000.032: sourcing /opt/nvim-linux64/share/nvim/runtime/indent.vim
006.607  000.053  000.053: require('misc.style')
008.412  000.008  000.008: require('vim.F')
008.453  001.534  001.526: require('vim.diagnostic')
009.843  001.384  001.384: require('vim.filetype')
011.071  000.753  000.753: require('_vim9script')
011.082  000.867  000.113: sourcing /opt/nvim-linux64/share/nvim/runtime/pack/dist/opt/cfilter/plugin/cfilter.lua
011.173  004.767  000.929: require('config.global')
011.807  000.477  000.477: require('lazy')
011.837  000.016  000.016: require('ffi')
011.888  000.023  000.023: require('vim.fs')
011.998  000.105  000.105: require('vim.uri')
012.010  000.171  000.042: require('vim.loader')
012.310  000.283  000.283: require('lazy.stats')
012.425  000.096  000.096: require('lazy.core.util')
012.532  000.104  000.104: require('lazy.core.config')
012.730  000.066  000.066: require('lazy.core.handler')
013.000  000.084  000.084: require('lazy.pkg')
013.005  000.194  000.110: require('lazy.core.meta')
013.012  000.279  000.085: require('lazy.core.plugin')
013.016  000.482  000.137: require('lazy.core.loader')
013.451  000.120  000.120: require('lazy.core.fragments')
036.022  000.057  000.057: require('lazy.core.handler.event')
036.031  000.125  000.067: require('lazy.core.handler.ft')
036.097  000.061  000.061: require('lazy.core.handler.keys')
036.143  000.043  000.043: require('lazy.core.handler.cmd')
039.040  000.236  000.236: sourcing /opt/nvim-linux64/share/nvim/runtime/filetype.lua
040.555  000.066  000.066: require('gitblame.utils')
040.561  000.176  000.110: require('gitblame.git')
040.686  000.068  000.068: require('lua-timeago/languages/en')
040.690  000.127  000.059: require('lua-timeago')
040.703  000.629  000.326: require('gitblame')
040.799  000.095  000.095: require('gitblame.config')
040.862  000.829  000.105: sourcing /home/christian/.local/share/nvim/lazy/git-blame.nvim/plugin/gitblame.lua
040.888  001.755  000.926: require('gitblame')
041.915  000.320  000.320: require('git-conflict.colors')
042.004  000.085  000.085: require('git-conflict.utils')
042.129  000.121  000.121: require('vim.highlight')
042.159  001.131  000.606: require('git-conflict')
043.786  000.364  000.364: require('kanagawa')
043.893  000.104  000.104: require('kanagawa.utils')
044.016  000.112  000.112: require('kanagawa.colors')
044.480  000.159  000.159: require('kanagawa.lib.hsluv')
044.492  000.294  000.135: require('kanagawa.lib.color')
044.493  000.415  000.121: require('kanagawa.themes')
044.588  000.072  000.072: require('kanagawa.highlights')
044.715  000.124  000.124: require('kanagawa.highlights.editor')
044.888  000.081  000.081: require('kanagawa.highlights.syntax')
045.066  000.082  000.082: require('kanagawa.highlights.treesitter')
045.173  000.074  000.074: require('kanagawa.highlights.lsp')
045.518  000.310  000.310: require('kanagawa.highlights.plugins')
046.766  003.359  001.620: sourcing /home/christian/.local/share/nvim/lazy/kanagawa.nvim/colors/kanagawa.vim
047.017  000.052  000.052: sourcing /home/christian/.local/share/nvim/lazy/which-key.nvim/plugin/which-key.lua
047.381  000.342  000.342: require('which-key')
047.477  000.092  000.092: require('which-key.config')
048.232  000.043  000.043: require('vim.treesitter.language')
048.292  000.057  000.057: require('vim.func')
048.375  000.079  000.079: require('vim.func._memoize')
048.389  000.263  000.084: require('vim.treesitter.query')
048.435  000.045  000.045: require('vim.treesitter._range')
048.445  000.431  000.123: require('vim.treesitter.languagetree')
048.450  000.531  000.100: require('vim.treesitter')
048.863  000.068  000.068: require('vim.lsp.log')
049.288  000.421  000.421: require('vim.lsp.protocol')
049.624  000.175  000.175: require('vim.lsp._snippet_grammar')
049.637  000.345  000.170: require('vim.lsp.util')
049.825  000.098  000.098: require('vim.lsp.sync')
049.833  000.193  000.095: require('vim.lsp._changetracking')
049.977  000.142  000.142: require('vim.lsp.rpc')
050.065  001.612  000.442: require('vim.lsp')
050.188  000.119  000.119: require('vim.lsp.buf')
052.438  000.085  000.085: require('mason-core.functional')
052.500  000.058  000.058: require('mason-core.path')
052.563  000.062  000.062: require('mason.settings')
052.586  000.292  000.087: require('mason-core.log')
052.590  000.349  000.057: require('mason-core.EventEmitter')
052.643  000.051  000.051: require('mason-core.optional')
052.784  000.083  000.083: require('mason-core.async')
052.844  000.056  000.056: require('mason-core.async.uv')
052.853  000.209  000.070: require('mason-core.fs')
052.957  000.102  000.102: require('mason-registry.sources')
053.120  000.073  000.073: require('mason-core.functional.data')
053.217  000.094  000.094: require('mason-core.functional.function')
053.246  000.285  000.118: require('mason-core.functional.list')
053.267  001.219  000.224: require('mason-registry')
053.278  001.837  000.618: require('mason-tool-installer')
053.295  001.938  000.101: sourcing /home/christian/.local/share/nvim/lazy/mason-tool-installer.nvim/plugin/mason-tool-installer.lua
053.865  000.382  000.382: require('neodev')
053.934  000.065  000.065: require('neodev.config')
054.130  000.076  000.076: require('neodev.util')
054.135  000.152  000.077: require('neodev.lsp')
054.379  000.242  000.242: require('lspconfig.util')
057.244  000.169  000.169: sourcing /home/christian/.local/share/nvim/lazy/nvim-lspconfig/plugin/lspconfig.lua
057.839  000.333  000.333: require('lspconfig.async')
057.843  000.466  000.132: require('lspconfig.configs')
057.869  000.590  000.124: require('lspconfig')
058.417  000.132  000.132: require('mason-core.functional.relation')
058.601  000.158  000.158: require('mason-core.functional.logic')
058.648  000.629  000.340: require('mason-core.platform')
058.651  000.780  000.151: require('mason')
059.268  000.224  000.224: require('mason-core.functional.string')
059.302  000.628  000.405: require('mason.api.command')
059.492  000.043  000.043: require('mason-lspconfig.settings')
059.497  000.172  000.130: require('mason-lspconfig')
059.583  000.035  000.035: require('mason-lspconfig.notify')
059.591  000.085  000.051: require('mason-lspconfig.lspconfig_hook')
059.730  000.047  000.047: require('mason-core.functional.table')
059.790  000.198  000.151: require('mason-lspconfig.mappings.server')
059.838  000.045  000.045: require('mason-lspconfig.server_config_extensions')
059.904  000.064  000.064: require('lspconfig.configs.omnisharp')
060.247  000.045  000.045: require('mason-core.functional.number')
060.267  000.128  000.083: require('mason-lspconfig.api.command')
061.043  000.115  000.115: require('vim.lsp.handlers')
061.763  000.275  000.275: require('cmp_nvim_lsp.source')
061.768  000.636  000.361: require('cmp_nvim_lsp')
061.902  000.049  000.049: require('lspconfig.configs.marksman')
063.183  000.062  000.062: require('lspconfig.manager')
063.253  000.066  000.066: require('lspconfig.configs.r_language_server')
063.557  000.053  000.053: require('mason-lspconfig.server_configurations.r_language_server')
063.915  000.051  000.051: require('lspconfig.configs.yamlls')
064.470  000.046  000.046: require('lspconfig.configs.dotls')
085.923  000.182  000.182: require('lspconfig.configs.lua_ls')
086.687  000.080  000.080: require('lspconfig.configs.julials')
086.807  000.067  000.067: require('mason-lspconfig.server_configurations.julials')
086.984  000.067  000.067: require('lspconfig.configs.bashls')
087.572  000.070  000.070: require('lspconfig.configs.pyright')
089.560  000.116  000.116: require('nvim-treesitter.compat')
091.965  002.184  002.184: require('nvim-treesitter.parsers')
092.130  000.159  000.159: require('nvim-treesitter.utils')
092.143  002.493  000.150: require('nvim-treesitter.ts_utils')
092.152  002.587  000.094: require('nvim-treesitter.tsrange')
092.250  000.096  000.096: require('nvim-treesitter.caching')
092.274  002.972  000.174: require('nvim-treesitter.query')
092.307  003.184  000.212: require('nvim-treesitter.configs')
092.310  003.660  000.476: require('nvim-treesitter-textobjects')
092.938  000.205  000.205: require('nvim-treesitter.info')
093.142  000.200  000.200: require('nvim-treesitter.shell_command_selectors')
093.224  000.707  000.303: require('nvim-treesitter.install')
093.358  000.130  000.130: require('nvim-treesitter.statusline')
093.524  000.163  000.163: require('nvim-treesitter.query_predicates')
093.530  001.217  000.218: require('nvim-treesitter')
093.810  000.161  000.161: require('nvim-treesitter.textobjects.shared')
093.821  000.280  000.119: require('nvim-treesitter.textobjects.select')
094.100  000.120  000.120: require('nvim-treesitter.textobjects.attach')
094.270  000.166  000.166: require('nvim-treesitter.textobjects.repeatable_move')
094.281  000.418  000.132: require('nvim-treesitter.textobjects.move')
094.803  000.112  000.112: require('nvim-treesitter.textobjects.swap')
094.964  000.118  000.118: require('nvim-treesitter.textobjects.lsp_interop')
094.996  006.364  000.559: sourcing /home/christian/.local/share/nvim/lazy/nvim-treesitter-textobjects/plugin/nvim-treesitter-textobjects.vim
095.956  000.830  000.830: sourcing /home/christian/.local/share/nvim/lazy/nvim-treesitter/plugin/nvim-treesitter.lua
099.211  000.158  000.158: require('nvim-treesitter.locals')
099.227  000.354  000.196: require('nvim-treesitter.incremental_selection')
099.429  000.128  000.128: require('nvim-treesitter.indent')
099.766  000.083  000.083: require('nvim-treesitter.highlight')
105.686  000.331  000.331: require('luasnip.util.types')
105.708  000.633  000.302: require('luasnip.util.ext_opts')
106.423  000.303  000.303: require('luasnip.util.lazy_table')
106.472  000.045  000.045: require('luasnip.extras.filetype_functions')
106.489  000.608  000.260: require('luasnip.default_config')
106.492  000.782  000.174: require('luasnip.session')
106.495  002.196  000.781: require('luasnip.config')
106.672  000.066  000.066: require('luasnip.util.util')
106.775  000.039  000.039: require('luasnip.nodes.key_indexer')
106.826  000.048  000.048: require('luasnip.util.feedkeys')
106.830  000.155  000.068: require('luasnip.nodes.util')
106.976  000.047  000.047: require('luasnip.session.snippet_collection.source')
107.019  000.040  000.040: require('luasnip.util.table')
107.062  000.041  000.041: require('luasnip.util.auto_table')
107.069  000.237  000.109: require('luasnip.session.snippet_collection')
107.219  000.045  000.045: require('luasnip.util.select')
107.259  000.039  000.039: require('luasnip.util.time')
107.661  000.546  000.462: require('luasnip.util._builtin_vars')
107.758  000.687  000.142: require('luasnip.util.environ')
107.812  000.051  000.051: require('luasnip.util.extend_decorator')
107.948  000.081  000.081: require('luasnip.util.path')
108.076  000.070  000.070: require('luasnip.util.log')
108.082  000.132  000.061: require('luasnip.loaders.util')
108.126  000.042  000.042: require('luasnip.loaders.data')
108.219  000.092  000.092: require('luasnip.loaders.fs_watchers')
108.224  000.410  000.063: require('luasnip.loaders')
108.240  001.736  000.129: require('luasnip')
108.260  004.497  000.564: sourcing /home/christian/.local/share/nvim/lazy/LuaSnip/plugin/luasnip.lua
108.364  000.030  000.030: sourcing /home/christian/.local/share/nvim/lazy/LuaSnip/plugin/luasnip.vim
109.005  000.056  000.056: require('cmp.utils.api')
109.093  000.041  000.041: require('cmp.types.cmp')
109.258  000.110  000.110: require('cmp.utils.misc')
109.299  000.204  000.094: require('cmp.types.lsp')
109.344  000.043  000.043: require('cmp.types.vim')
109.347  000.339  000.050: require('cmp.types')
109.389  000.041  000.041: require('cmp.utils.highlight')
109.500  000.067  000.067: require('cmp.utils.debug')
109.507  000.116  000.049: require('cmp.utils.autocmd')
109.811  000.916  000.364: sourcing /home/christian/.local/share/nvim/lazy/nvim-cmp/plugin/cmp.lua
110.114  000.069  000.069: require('cmp.utils.char')
110.121  000.121  000.052: require('cmp.utils.str')
110.265  000.036  000.036: require('cmp.utils.buffer')
110.275  000.109  000.072: require('cmp.utils.keymap')
110.278  000.155  000.047: require('cmp.utils.feedkeys')
110.442  000.055  000.055: require('cmp.config.mapping')
110.483  000.039  000.039: require('cmp.utils.cache')
110.591  000.052  000.052: require('cmp.config.compare')
110.593  000.107  000.055: require('cmp.config.default')
110.607  000.271  000.070: require('cmp.config')
110.617  000.337  000.066: require('cmp.utils.async')
110.700  000.038  000.038: require('cmp.utils.pattern')
110.704  000.086  000.048: require('cmp.context')
110.928  000.099  000.099: require('cmp.utils.snippet')
110.977  000.045  000.045: require('cmp.matcher')
110.983  000.221  000.076: require('cmp.entry')
110.989  000.284  000.063: require('cmp.source')
111.077  000.037  000.037: require('cmp.utils.event')
111.216  000.037  000.037: require('cmp.utils.options')
111.221  000.097  000.060: require('cmp.utils.window')
111.223  000.144  000.047: require('cmp.view.docs_view')
111.294  000.069  000.069: require('cmp.view.custom_entries_view')
111.356  000.060  000.060: require('cmp.view.wildmenu_entries_view')
111.408  000.050  000.050: require('cmp.view.native_entries_view')
111.470  000.061  000.061: require('cmp.view.ghost_text_view')
111.478  000.488  000.066: require('cmp.view')
112.135  002.217  000.746: require('cmp.core')
112.376  000.077  000.077: require('cmp.config.sources')
112.421  000.041  000.041: require('cmp.config.window')
112.470  002.636  000.300: require('cmp')
112.964  000.492  000.492: require('lspkind')
113.405  000.042  000.042: require('luasnip.session.enqueueable_operations')
113.654  000.037  000.037: require('luasnip.util.events')
113.663  000.106  000.069: require('luasnip.nodes.node')
113.742  000.077  000.077: require('luasnip.nodes.insertNode')
113.797  000.053  000.053: require('luasnip.nodes.textNode')
113.848  000.049  000.049: require('luasnip.util.mark')
113.893  000.042  000.042: require('luasnip.util.pattern_tokenizer')
113.943  000.048  000.048: require('luasnip.util.dict')
114.480  000.496  000.496: require('luasnip.util.jsregexp')
114.485  000.539  000.044: require('luasnip.nodes.util.trig_engines')
114.581  001.123  000.209: require('luasnip.nodes.snippet')
114.753  000.044  000.044: require('luasnip.util.parser.neovim_ast')
114.801  000.046  000.046: require('luasnip.util.str')
115.140  000.335  000.335: require('luasnip.util.jsregexp')
115.196  000.054  000.054: require('luasnip.util.directed_graph')
115.200  000.540  000.062: require('luasnip.util.parser.ast_utils')
115.259  000.056  000.056: require('luasnip.nodes.functionNode')
115.327  000.066  000.066: require('luasnip.nodes.choiceNode')
115.421  000.092  000.092: require('luasnip.nodes.dynamicNode')
115.490  000.066  000.066: require('luasnip.util.functions')
115.494  000.911  000.090: require('luasnip.util.parser.ast_parser')
115.611  000.115  000.115: require('luasnip.util.parser.neovim_parser')
115.622  002.214  000.064: require('luasnip.util.parser')
115.682  000.059  000.059: require('luasnip.nodes.snippetProxy')
115.781  000.096  000.096: require('luasnip.util.jsonc')
115.875  000.045  000.045: require('luasnip.nodes.duplicate')
115.878  000.093  000.048: require('luasnip.loaders.snippet_cache')
115.887  002.617  000.114: require('luasnip.loaders.from_vscode')
119.596  000.082  000.082: require('luasnip.nodes.multiSnippet')
122.729  000.177  000.177: require('otter.tools.extensions')
123.343  000.150  000.150: require('otter.config')
123.352  000.363  000.213: require('otter.tools.functions')
123.377  000.627  000.264: require('otter.keeper')
123.757  000.229  000.229: require('otter.lsp.handlers')
123.967  000.588  000.360: require('otter.lsp')
123.982  001.656  000.263: require('otter')
124.079  073.887  046.022: require('otter')
124.125  076.571  000.421: require('config.keymap')
124.983  000.075  000.075: sourcing /home/christian/.local/share/nvim/lazy/dropbar.nvim/plugin/dropbar.lua
125.755  000.076  000.076: require('dropbar.utils')
125.763  000.742  000.666: require('dropbar.api')
126.656  000.107  000.107: require('toggleterm.lazy')
126.743  000.084  000.084: require('toggleterm.constants')
126.979  000.231  000.231: require('toggleterm.terminal')
126.998  001.031  000.609: require('toggleterm')
127.164  000.080  000.080: require('toggleterm.colors')
127.257  000.090  000.090: require('toggleterm.utils')
127.265  000.265  000.095: require('toggleterm.config')
127.452  000.131  000.131: require('vim.version')
130.327  001.050  001.050: require('toggleterm.commandline')
130.792  000.167  000.167: sourcing /home/christian/.local/share/nvim/lazy/nvim-scrollview/plugin/scrollview.vim
132.690  000.932  000.932: require('scrollview.utils')
133.805  002.953  002.021: require('scrollview')
135.272  000.085  000.085: sourcing /home/christian/.local/share/nvim/lazy/vim-slime/autoload/slime/config.vim
135.314  000.588  000.503: sourcing /home/christian/.local/share/nvim/lazy/vim-slime/plugin/slime.vim
137.682  000.681  000.681: require('nvim-autopairs._log')
137.761  000.071  000.071: require('nvim-autopairs.utils')
137.937  000.058  000.058: require('nvim-autopairs.conds')
137.945  000.118  000.059: require('nvim-autopairs.rule')
137.947  000.184  000.066: require('nvim-autopairs.rules.basic')
137.956  002.274  001.339: require('nvim-autopairs')
138.637  000.052  000.052: sourcing /home/christian/.local/share/nvim/lazy/vimtex/plugin/vimtex.vim
138.721  000.019  000.019: sourcing /home/christian/.local/share/nvim/lazy/vimtex/ftdetect/cls.vim
138.764  000.016  000.016: sourcing /home/christian/.local/share/nvim/lazy/vimtex/ftdetect/tex.vim
138.814  000.016  000.016: sourcing /home/christian/.local/share/nvim/lazy/vimtex/ftdetect/tikz.vim
139.741  000.026  000.026: sourcing /home/christian/.local/share/nvim/lazy/nvim-web-devicons/plugin/nvim-web-devicons.vim
140.402  000.585  000.585: require('alpha')
140.478  000.073  000.073: require('alpha.themes.dashboard')
140.630  000.142  000.142: require('alpha.fortune')
141.324  000.034  000.034: sourcing /home/christian/.local/share/nvim/lazy/plenary.nvim/plugin/plenary.vim
141.409  000.024  000.024: sourcing /home/christian/.local/share/nvim/lazy/todo-comments.nvim/plugin/todo.vim
142.089  000.045  000.045: require('todo-comments.util')
142.098  000.123  000.078: require('todo-comments.config')
142.210  000.068  000.068: require('todo-comments.highlight')
142.213  000.113  000.045: require('todo-comments.jump')
142.216  000.734  000.497: require('todo-comments')
142.788  000.493  000.493: require('codeium')
142.903  000.054  000.054: require('codeium.enums')
143.274  000.051  000.051: require('plenary.bit')
143.321  000.044  000.044: require('plenary.functional')
143.348  000.208  000.114: require('plenary.path')
143.441  000.355  000.147: require('plenary.log')
143.492  000.449  000.094: require('codeium.log')
143.497  000.493  000.043: require('codeium.notify')
143.500  000.545  000.053: require('codeium.config')
143.695  000.044  000.044: require('plenary.compat')
143.703  000.122  000.078: require('plenary.job')
143.771  000.067  000.067: require('plenary.curl')
143.777  000.276  000.088: require('codeium.io')
143.781  000.875  000.053: require('codeium.util')
143.785  000.994  000.065: require('codeium.source')
143.970  000.109  000.109: require('codeium.versions')
144.026  000.053  000.053: require('codeium.update')
144.031  000.244  000.082: require('codeium.api')
144.141  000.067  000.067: require('vim.health')
144.151  000.119  000.052: require('codeium.health')
146.864  000.235  000.235: require('codeium.virtual_text')
149.413  000.302  000.302: sourcing /home/christian/.local/share/nvim/lazy/telescope.nvim/plugin/telescope.lua
150.620  000.092  000.092: require('telescope._extensions')
150.628  001.119  001.027: require('telescope')
151.196  000.097  000.097: require('plenary.strings')
151.273  000.072  000.072: require('telescope.deprecated')
151.617  000.226  000.226: require('telescope.log')
151.835  000.075  000.075: require('telescope.state')
151.864  000.244  000.170: require('telescope.utils')
151.871  000.596  000.126: require('telescope.sorters')
154.215  003.271  002.506: require('telescope.config')
154.478  000.081  000.081: require('plenary.window.border')
154.536  000.054  000.054: require('plenary.window')
154.581  000.043  000.043: require('plenary.popup.utils')
154.585  000.358  000.180: require('plenary.popup')
154.665  000.078  000.078: require('telescope.pickers.scroller')
154.743  000.076  000.076: require('telescope.actions.state')
154.822  000.077  000.077: require('telescope.actions.utils')
155.003  000.080  000.080: require('telescope.actions.mt')
155.021  000.196  000.116: require('telescope.actions.set')
155.182  000.080  000.080: require('telescope.config.resolve')
155.185  000.163  000.083: require('telescope.pickers.entry_display')
155.253  000.066  000.066: require('telescope.from_entry')
155.471  004.841  000.556: require('telescope.actions')
155.620  000.069  000.069: require('telescope.previewers.previewer')
155.803  000.085  000.085: require('telescope.previewers.utils')
155.843  000.221  000.136: require('telescope.previewers.term_previewer')
156.334  000.309  000.309: require('plenary.scandir')
156.386  000.540  000.231: require('telescope.previewers.buffer_previewer')
156.390  000.915  000.085: require('telescope.previewers')
156.483  000.086  000.086: require('telescope.themes')
158.304  000.186  000.186: require('fzf_lib')
158.317  000.404  000.218: require('telescope._extensions.fzf')
158.420  000.095  000.095: require('telescope._extensions.ui-select')
158.951  000.060  000.060: require('plenary.tbl')
158.956  000.121  000.060: require('plenary.vararg.rotate')
158.958  000.169  000.048: require('plenary.vararg')
159.007  000.048  000.048: require('plenary.errors')
159.014  000.280  000.063: require('plenary.async.async')
159.225  000.050  000.050: require('plenary.async.structs')
159.232  000.114  000.065: require('plenary.async.control')
159.239  000.173  000.059: require('plenary.async.util')
159.241  000.225  000.052: require('plenary.async.tests')
159.243  000.571  000.066: require('plenary.async')
159.339  000.094  000.094: require('telescope.debounce')
159.500  000.158  000.158: require('telescope.mappings')
159.596  000.092  000.092: require('telescope.pickers.highlights')
159.670  000.072  000.072: require('telescope.pickers.window')
159.748  000.076  000.076: require('telescope.pickers.layout')
159.917  000.083  000.083: require('telescope.algos.linked_list')
159.921  000.171  000.088: require('telescope.entry_manager')
159.994  000.071  000.071: require('telescope.pickers.multi')
160.016  001.592  000.287: require('telescope.pickers')
160.388  000.279  000.279: require('telescope.make_entry')
160.483  000.092  000.092: require('telescope.finders.async_static_finder')
160.714  000.056  000.056: require('plenary.class')
160.795  000.229  000.173: require('telescope._')
160.798  000.313  000.084: require('telescope.finders.async_oneshot_finder')
160.894  000.094  000.094: require('telescope.finders.async_job_finder')
160.901  000.884  000.106: require('telescope.finders')
160.975  000.061  000.061: require('vim.ui')
163.982  000.155  000.155: sourcing /home/christian/.local/share/nvim/lazy/nvim-dap/plugin/dap.lua
165.411  000.463  000.463: require('dap.utils')
165.526  001.460  000.997: require('dap')
166.011  000.156  000.156: require('nio.tasks')
166.108  000.094  000.094: require('nio.control')
166.647  000.504  000.504: require('nio.uv')
166.744  000.093  000.093: require('nio.tests')
166.847  000.101  000.101: require('nio.ui')
167.021  000.096  000.096: require('nio.streams')
167.030  000.181  000.085: require('nio.file')
167.283  000.074  000.074: require('nio.util')
167.365  000.254  000.181: require('nio.logger')
167.378  000.346  000.092: require('nio.lsp')
167.525  000.145  000.145: require('nio.process')
167.553  001.833  000.213: require('nio')
168.258  000.148  000.148: require('dapui.config.highlights')
168.486  000.683  000.535: require('dapui.config')
168.495  000.825  000.142: require('dapui.util')
168.612  000.115  000.115: require('dapui.windows.layout')
168.617  001.061  000.121: require('dapui.windows')
168.751  000.132  000.132: require('dapui.controls')
168.766  003.237  000.210: require('dapui')
169.410  000.276  000.276: require('dapui.client.types')
169.418  000.394  000.118: require('dapui.client')
169.524  000.105  000.105: require('dap.breakpoints')
169.623  000.079  000.079: require('dapui.client.lib')
169.852  000.127  000.127: require('dapui.render.canvas')
169.855  000.207  000.079: require('dapui.elements.breakpoints')
169.997  000.105  000.105: require('dapui.components.breakpoints')
170.141  000.070  000.070: require('dapui.elements.repl')
170.216  000.070  000.070: require('dapui.elements.scopes')
170.319  000.072  000.072: require('dapui.components.scopes')
170.445  000.124  000.124: require('dapui.components.variables')
170.547  000.071  000.071: require('dapui.elements.stacks')
170.711  000.068  000.068: require('dapui.components.frames')
170.714  000.139  000.071: require('dapui.components.threads')
170.809  000.066  000.066: require('dapui.elements.watches')
170.923  000.086  000.086: require('dapui.components.watches')
171.024  000.068  000.068: require('dapui.elements.hover')
171.132  000.081  000.081: require('dapui.components.hover')
171.231  000.089  000.089: require('dapui.elements.console')
171.870  000.394  000.394: require('dap-python')
172.119  000.207  000.207: require('dap.ext.vscode')
172.274  000.139  000.139: require('nvim-dap-virtual-text')
172.425  011.344  003.927: require('dap')
172.691  000.261  000.261: require('telescope.builtin')
172.703  011.714  000.108: require('telescope._extensions.dap')
173.024  000.122  000.122: require('zotero.bib')
173.421  000.149  000.149: require('sqlite.utils')
174.425  000.979  000.979: require('sqlite.defs')
174.430  001.319  000.192: require('sqlite.db')
174.442  001.414  000.095: require('zotero.database')
174.483  001.695  000.159: require('zotero')
174.487  001.768  000.072: require('telescope._extensions.zotero')
175.884  000.238  000.238: sourcing /home/christian/.local/share/nvim/lazy/tabular/plugin/Tabular.vim
177.543  000.087  000.087: require('diffview.lazy')
177.974  000.111  000.111: require('diffview.ffi')
178.083  000.106  000.106: require('diffview.oop')
178.107  000.392  000.175: require('diffview.async')
178.451  000.313  000.313: require('diffview.utils')
178.529  000.068  000.068: require('diffview.mock')
178.553  000.967  000.195: require('diffview.logger')
178.703  000.133  000.133: require('diffview.control')
178.839  000.118  000.118: require('diffview.events')
178.847  002.243  000.938: require('diffview.bootstrap')
178.880  002.352  000.108: sourcing /home/christian/.local/share/nvim/lazy/diffview.nvim/plugin/diffview.lua
179.215  000.069  000.069: sourcing /home/christian/.local/share/nvim/lazy/conform.nvim/plugin/conform.lua
180.309  001.067  001.067: require('conform')
180.755  000.074  000.074: sourcing /home/christian/.local/share/nvim/lazy/nvim-colorizer.lua/plugin/colorizer.lua
183.565  000.409  000.409: require('colorizer.color')
184.955  000.832  000.832: require('colorizer.utils')
184.966  001.370  000.538: require('colorizer.sass')
185.393  000.425  000.425: require('colorizer.tailwind')
186.469  000.752  000.752: require('colorizer.trie')
186.750  000.272  000.272: require('colorizer.parser.argb_hex')
187.185  000.432  000.432: require('colorizer.parser.names')
187.429  000.242  000.242: require('colorizer.parser.hsl')
187.818  000.386  000.386: require('colorizer.parser.rgb')
188.190  000.368  000.368: require('colorizer.parser.rgba_hex')
188.196  002.799  000.347: require('colorizer.matcher')
188.255  005.815  000.812: require('colorizer.buffer')
188.772  000.515  000.515: require('colorizer.config')
188.846  008.002  001.672: require('colorizer')
189.692  000.289  000.289: require('colorizer.usercmds')
190.296  000.269  000.269: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/editorconfig.lua
190.561  000.198  000.198: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/gzip.vim
190.856  000.260  000.260: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/man.lua
191.908  000.259  000.259: sourcing /opt/nvim-linux64/share/nvim/runtime/pack/dist/opt/matchit/plugin/matchit.vim
191.933  001.029  000.770: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/matchit.vim
192.117  000.139  000.139: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/matchparen.vim
192.460  000.290  000.290: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/netrwPlugin.vim
192.823  000.298  000.298: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/osc52.lua
193.096  000.199  000.199: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/rplugin.vim
193.349  000.105  000.105: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/shada.vim
193.453  000.023  000.023: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/spellfile.vim
193.608  000.113  000.113: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/tarPlugin.vim
193.905  000.230  000.230: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/tohtml.lua
193.977  000.026  000.026: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/tutor.vim
194.209  000.178  000.178: sourcing /opt/nvim-linux64/share/nvim/runtime/plugin/zipPlugin.vim
195.650  000.198  000.198: sourcing /home/christian/.local/share/nvim/lazy/tabular/autoload/tabular.vim
197.506  002.829  002.630: sourcing /home/christian/.local/share/nvim/lazy/tabular/after/plugin/TabularMaps.vim
198.387  000.229  000.229: require('bigfile.features')
198.394  000.322  000.093: require('bigfile')
198.476  000.766  000.445: sourcing /home/christian/.local/share/nvim/lazy/bigfile.nvim/after/plugin/bigfile.lua
198.767  000.064  000.064: require('cmp-pandoc-references.references')
198.771  000.150  000.085: require('cmp-pandoc-references')
198.820  000.247  000.098: sourcing /home/christian/.local/share/nvim/lazy/cmp-pandoc-references/after/plugin/cmp-pandoc-references.lua
203.917  004.894  004.894: require('cmp_latex_symbols.items_mixed')
207.970  004.040  004.040: require('cmp_latex_symbols.items_julia')
216.637  008.654  008.654: require('cmp_latex_symbols.items_latex')
216.653  017.726  000.138: require('cmp_latex_symbols')
216.728  017.845  000.120: sourcing /home/christian/.local/share/nvim/lazy/cmp-latex-symbols/after/plugin/cmp_latex.lua
217.336  000.099  000.099: require('cmp_treesitter.lru')
217.523  000.402  000.304: require('cmp_treesitter.treesitter')
217.529  000.524  000.122: require('cmp_treesitter')
217.574  000.667  000.144: sourcing /home/christian/.local/share/nvim/lazy/cmp-treesitter/after/plugin/cmp_treesitter.lua
217.872  000.128  000.128: require('cmp-spell')
217.903  000.229  000.101: sourcing /home/christian/.local/share/nvim/lazy/cmp-spell/after/plugin/cmp-spell.lua
218.236  000.182  000.182: require('cmp_luasnip')
218.303  000.327  000.145: sourcing /home/christian/.local/share/nvim/lazy/cmp_luasnip/after/plugin/cmp_luasnip.lua
218.654  000.178  000.178: require('cmp_emoji')
218.712  000.310  000.132: sourcing /home/christian/.local/share/nvim/lazy/cmp-emoji/after/plugin/cmp_emoji.lua
219.065  000.161  000.161: require('cmp_calc')
219.105  000.272  000.111: sourcing /home/christian/.local/share/nvim/lazy/cmp-calc/after/plugin/cmp_calc.lua
219.509  000.229  000.229: require('cmp_path')
219.541  000.333  000.104: sourcing /home/christian/.local/share/nvim/lazy/cmp-path/after/plugin/cmp_path.lua
220.368  000.200  000.200: require('cmp_buffer.timer')
220.384  000.468  000.268: require('cmp_buffer.buffer')
220.454  000.664  000.196: require('cmp_buffer.source')
220.458  000.750  000.085: require('cmp_buffer')
220.531  000.894  000.144: sourcing /home/christian/.local/share/nvim/lazy/cmp-buffer/after/plugin/cmp_buffer.lua
221.134  000.234  000.234: require('cmp_nvim_lsp_signature_help')
221.215  000.426  000.192: sourcing /home/christian/.local/share/nvim/lazy/cmp-nvim-lsp-signature-help/after/plugin/cmp_nvim_lsp_signature_help.lua
221.512  000.119  000.119: sourcing /home/christian/.local/share/nvim/lazy/cmp-nvim-lsp/after/plugin/cmp_nvim_lsp.lua
221.600  210.425  047.333: require('config.lazy')
221.787  000.182  000.182: require('config.autocommands')
221.790  215.411  000.036: sourcing /home/christian/.config/nvim/init.lua
221.801  000.693: sourcing vimrc file(s)
222.736  000.249  000.249: sourcing /opt/nvim-linux64/share/nvim/runtime/filetype.lua
223.529  000.124  000.124: sourcing /opt/nvim-linux64/share/nvim/runtime/syntax/synload.vim
223.675  000.631  000.507: sourcing /opt/nvim-linux64/share/nvim/runtime/syntax/syntax.vim
223.693  001.012: inits 3
227.476  003.783: reading ShaDa
231.113  001.449  001.449: require('vim.filetype.detect')
237.343  000.228  000.228: require('dropbar.hlgroups')
237.834  000.342  000.342: require('dropbar.configs')
237.847  000.495  000.154: require('dropbar.bar')
237.852  000.992  000.269: require('dropbar')
238.231  000.123  000.123: require('dropbar.utils.bar')
238.565  000.141  000.141: require('editorconfig')
238.881  000.150  000.150: require('luasnip.loaders.from_lua')
239.064  000.178  000.178: require('luasnip.loaders.from_snipmate')
239.367  008.858: opening buffers
240.609  000.894  000.894: require('img-clip.debug')
241.079  000.464  000.464: require('img-clip.config')
241.439  000.356  000.356: require('img-clip.util')
242.291  000.435  000.435: require('img-clip.clipboard')
242.890  000.371  000.371: require('img-clip.fs')
242.894  000.598  000.227: require('img-clip.markup')
242.915  001.291  000.258: require('img-clip.paste')
242.917  001.475  000.184: require('img-clip')
242.980  003.336  000.147: sourcing /home/christian/.local/share/nvim/lazy/img-clip.nvim/plugin/img-clip.lua
243.816  001.112: BufEnter autocommands
243.820  000.004: editing files in windows
244.051  000.231: VimEnter autocommands
244.465  000.414: UIEnter autocommands
244.979  000.238  000.238: sourcing /opt/nvim-linux64/share/nvim/runtime/autoload/provider/clipboard.vim
244.988  000.285: before starting main loop
245.429  000.440: first screen update
245.431  000.003: --- NVIM STARTED ---


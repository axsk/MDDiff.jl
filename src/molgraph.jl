using ISOKANN: OpenMMSimulation, ISOKANN
using GraphNeuralNetworks: GNNGraph, adjacency_matrix, EGNNConv, GraphNeuralNetworks
using ChainRulesCore: @ignore_derivatives, ignore_derivatives
import SparseArrays
using LinearAlgebra: Diagonal
using Flux: gradient
using Flux.Zygote: @showgrad
using Flux: Flux, gpu, cpu
using Zygote: withgradient
import Clustering
using MLUtils: eachobs
using SparseArrays: sparse
using LinearAlgebra: I

sim = OpenMMSimulation()

function randomgraph(n=10)
    A = rand(Bool, n, n)
    A = A + A'
    f = rand(1,n)
    x = rand(3, n)
    GNNGraph(A, ndata=(;x, f))
end


# Note: take care that no node is connected to itself, as this apparently leads to NaN in the autodiff
function graph(sim::OpenMMSimulation = sim)
    A = adjacency_bonds(sim)
    AA = (A*A .> 0) - I
    g = GNNGraph(A+AA.> 0)

    a,b = GraphNeuralNetworks.edge_index(g)
    e = zeros(Float32, 2, length(a))
    for i in eachindex(a)
        e[1, i] = A[a[i], b[i]]
        e[2, i] = AA[a[i], b[i]] * 0.2
    end

    g.edata.type = e

    g.ndata.f = collect(Float32, ISOKANN.OpenMM.masses(sim)')
    m = ISOKANN.OpenMM.masses(sim)
    f = mapreduce(vcat, [1, 12, 14, 16]) do a
        isapprox.(m, a, atol=0.1)'
    end
    g.ndata.f = vcat(Float32.(f), zeros(size(f, 2))')
    g.ndata.x = collect(Float32, reshape(ISOKANN.OpenMM.getcoords(sim), 3, :))
    g.gdata.t = 0.
    return g
end

function adjacency_bonds(sim)
    n = ISOKANN.OpenMM.natoms(sim)
     bonds = reshape(ISOKANN.bondids(sim.pysim), 2, :)
     a = bonds[1,:]
     b = bonds[2,:]
     A = sparse(a, b, true, n, n)
     A + A'
end

# add dispatch for feature graph
function (l::EGNNConv)(g::GNNGraph)
    e = length(g.edata) > 0 ? g.edata.type : nothing
    #ff = vcat(g.f, fill(g.t, (1, size(g.f, 2))))
    f, x = l(g, g.f, g.x, e)
    GNNGraph(g, ndata=(; x, f))
end

abstract type UNet end

mutable struct UNet3 <: UNet
    d1
    d2
    d3
    u1
    u2
    u3
    k1
    k2
end

Flux.@layer UNet trainable = (d1,d2,u1,u2)

UNet() = UNet(graph();)

function UNet(g::GNNGraph; kwargs...) 
    u = UNet(
    f=size(g.ndata.f, 1), 
    k1=div(g.num_nodes, 3),
    k2=div(g.num_nodes, 8); 
    kwargs...)

    g.x isa CuArray && (u = gpu(u))
    return u
end

function UNet(;f, h=f, k1, k2, residual = true)
    d1 = EGNNConv(f  => h; residual)
    d2 = EGNNConv(h => h; residual)
    d3 = EGNNConv(h => h; residual)
    u1 = EGNNConv(h => h; residual)
    u2 = EGNNConv(h => h; residual)
    u3 = EGNNConv(h => h; residual)
    UNet3(d1,d2,d3,u1,u2,u3,k1,k2)
end

function (u::UNet)(g::GNNGraph)
    
    g = u.d1(g)
    g, un1 = kmmpool(g, u.k1)
    g = u.d2(g)
    g = un1(g)
    g = u.u2(g)
    
    #g3 = u.u1(g2)
    return g
end


function trainingsdata(n)
    sim = OpenMMSimulation(steps=1000)
    gs = [graph(sim)]
    for i in 1:n-1
        ISOKANN.OpenMM.laggedtrajectory(sim, 1)
        push!(gs, graph(sim))
    end
    return gs
end

train(arch::NamedTuple; kwargs...) = train(u=arch.u, opt=arch.opt, losses=arch.losses;kwargs...)

function architecture(u=UNet(), opt = Flux.setup(Flux.Adam(1e-3),u), losses=Float64[])
    return (;u,opt,losses)
end


function train(; 
    gs=[graph(OpenMMSimulation())], 
    u=UNet(), 
    epochs=1, 
    opt=Flux.setup(Flux.Adam(1.0f-4), u), 
    batchsize=10, 
    losses=Float64[],
    perturbation=0)

    for _ in 1:epochs
        for gs in eachobs(gs; batchsize)
            l, grad = withgradient(u) do u
                loss = 0.
                for g in gs
                    if perturbation > 0
                        g1 = perturb(g, rand()*(1-perturbation))
                        g2 = perturb(g1, perturbation)
                    else
                        g1 = g2 = g
                    end
                    
                    pred = u(g2).x
                    ref  = @ignore_derivatives(ISOKANN.align(g1.x, pred))  # not sure if we need this here since u is equivariant
                    #ref = g1.x
                    # however, going from N to denoised rotated representations may be as good

                    loss += sum(abs2, pred-ref)
                end
                return loss / length(gs)
            end
            @show l
            push!(losses, l)
            Flux.update!(opt, u, grad[1])
        end
    end
    
    return (;u,opt, losses)
end

function test_gradients(;g = graph(), u = UNet(g))
    grad = gradient(u) do u
        sum(abs2, u(g).x - g.x)
    end
end

function perturb(g, t) # t=0 equals identity
    sigma = 0.5
    gg = deepcopy(g)  # copy lead to changing xs
    x = g.ndata.x
    gg.ndata.x = sqrt(1-t) * x + sigma * randn(size(x)) * sqrt(t)
    @ignore_derivatives gg.ndata.f[end, :] .+= t
    return gg
end

function generate(u, g, steps, t=1)
    dt = t/steps
    g = perturb(g, t)
    for _ in 1:steps
        g = u(g)
    end
    return g
end





### GRAVEYARD

#=

function eval2(model, g)
    f, x = model.d1(g, g.f, g.x)

    n = 10
    adj = adjacency_matrix(g)
    A, proj, w = ignore_derivatives() do
        kmmpoolmat(x, n, adj)
    end
    #P = rand(Bool, 22,10)

    # P = Float32.(proj) .+ 1

    # pool
    f = f * proj ./ w
    x = x * proj ./ w

    f, x = model.d2(GNNGraph(A), f, x)

    # stub values for testing
    model.u2::GraphNeuralNetworks.EGNNConv
    f = rand(4, 10)
    x = rand(3, 10)

    # if we reassign proj to random it works
    # if it just stays the Matrix{Bool} from above we get NaNs

    #proj = rand(Bool, 22, 10)

    proj = zeros(size(proj)) + proj

    @show proj #:: Matrix{Bool}
    P = collect(proj)

    f = (f) * P'
    x = (x) * P'

    f, x = (model.u2(g, f, x))

    sum(f) + sum(x)
end


function eval3(u=EGNNConv(4 => 4), proj=rand(Bool, 22, 10))

    A2 = rand(22, 22) .> 0.01
    A2 = A2 - Diagonal(A2)
    g2 = GNNGraph(A2)

    f = rand(Float32, 4, 10)
    x = rand(Float32, 3, 10)

    proj = rand(Bool, 22, 10)

    f = (f) * proj'
    x = (x) * proj'

    f, x = (u(g2, f, x))

    sum(f) + sum(x)
end

function distgraph(sim::OpenMMSimulation)
    masses = ISOKANN.OpenMM.masses(sim)
    n = length(masses)
    inds = samplerandompairs(min(n * (n + 1) / 2, 100), n)

    coords = ISOKANN.OpenMM.getcoords(sim)

    dists = ISOKANN.pdists(coords, inds)
    g = GNNGraph(first.(inds), last.(inds))
    g.edata.dist = dists
    g
end

function samplerandompairs(n, maxnum)
    s = Set{Tuple{Int,Int}}()
    while length(s) < n
        p = Tuple(sort([rand(1:maxnum), rand(1:maxnum)]))
        p[1] != p[2] && push!(s, p)
    end
    return collect(s)
end
import Clustering
function emolgraph(sim)
    g = graph(sim)
    coords = g.coords

    l1 = EGNNConv(1 => 8)

    h1, x1 = l1(g, g.mass', g.coords) # conv

    # k-means average pooling



    x = coords
    h = g.mass'

    h, x = l1(g, h, x) # conv

    TopKPool(adjacency_matrix(g), 3, 8)# pool
end


function kmmpool(g, n)
    A, x, f, proj = kmmpool(adjacency_matrix(g), g.x, g.f, n)
    upool(gg) = unpool(gg, (proj, g, x))
    return GNNGraph(A, ndata=(; x, f)), upool
end

function kmmpool(adj, x, f, n)
    A, proj, w = kmmpoolmat(x, n, adj)
    f2 = f * proj ./ w
    x2 = x * proj ./ w
    return A, x2, f2, proj
end

function kmmpoolmat(x, n, adj)
    ignore_derivatives() do
        _kmmpoolmat(x, n, adj)
    end
end

function _kmmpoolmat(x, n, adj)
    km = Clustering.kmeans(x, n)
    proj = SparseArrays.sparse(1:length(km.assignments), km.assignments, true)
    A = proj' * adj * proj
    A = A - Diagonal(A)
    A = A .> 0
    #A = collect(Int, A)
    w = collect(km.wcounts')
    return A, collect(proj), w
end



function unpool(g, (proj, gold, xold))
    residual = true
    f = (g.f) * proj' + residual * gold.f
    x = (g.x - xold) * proj' + residual * gold.x

    GNNGraph(gold.graph; ndata=(; x, f))
end

=#
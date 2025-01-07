using ISOKANN: OpenMMSimulation, ISOKANN
using GraphNeuralNetworks: GNNGraph, adjacency_matrix, EGNNConv
using ChainRulesCore: @ignore_derivatives, ignore_derivatives
import SparseArrays
using LinearAlgebra: Diagonal

sim = OpenMMSimulation()

function graph(sim::OpenMMSimulation)
    bonds = reshape(ISOKANN.bondids(sim.pysim), 2, :)
    a = bonds[1,:]
    b = bonds[2,:]
    g = GNNGraph([a;b], [b;a])
    g.ndata.f = collect(ISOKANN.OpenMM.masses(sim)')
    g.ndata.x = reshape(ISOKANN.OpenMM.getcoords(sim), 3, :)
    return g
end

function kmmpool(g, n)
    A, x, f, proj = kmmpool(adjacency_matrix(g), g.x, g.f, n)
    return GNNGraph(A, ndata=(; x, f)), (proj, g)
end

function kmmpoolmat(x, n, adj)
    km = Clustering.kmeans(x, n)
    proj = SparseArrays.sparse(1:length(km.assignments), km.assignments, true)
    A = proj' * adj * proj
    A = A - Diagonal(A)
    A = A .> 0
    #A = collect(Int, A)
    w = collect(km.wcounts')
    return A, collect(proj), w
end

function kmmpool(adj, x, f, n)
    #A, proj, w = @ignore_derivatives 

    A, proj, w = ignore_derivatives() do 
        kmmpoolmat(x,n, adj)
    end
    #@show A
    
    f = f * proj ./ w
    x = x * proj ./ w

    return A, x, f, proj
end

# add dispatch for feature graph
function (l::EGNNConv)(g::GNNGraph)
    f,x = l(g, g.f, g.x)
    GNNGraph(g, ndata=(;x,f))
end
using Flux.Zygote: @showgrad
function unpool(g, (proj, gold) , residual=false)
    f = @showgrad(g.f) * proj' + residual * gold.f
    x = @showgrad(g.x) * proj' + residual * gold.x
        
    GNNGraph(gold.graph; ndata=(; x, f))
end

abstract type UNet end

struct UNet1 <: UNet
    d1
    d2
    u1
    u2
end

function UNet(nh=4, residual=false)   
    UNet1(EGNNConv(1 => nh; ), EGNNConv(nh => nh; residual), EGNNConv(nh => nh; residual), EGNNConv(nh => nh; residual))
end

function (model::UNet)(g::GNNGraph)
    g = model.d1(g)
    g, un1 = kmmpool(g, 10)
    g = @showgrad(model.d2(g))
    g = @showgrad(unpool(g, un1, false))
    g = @showgrad(model.u2(g))
    
    g = model.u1(g)
    return g
end

function eval2(model, g)


    f, x = model.d1(g, g.f, g.x)

    n = 10

    adj = adjacency_matrix(g)

    A, proj, w = ignore_derivatives() do
        kmmpoolmat(x, n, adj)
    end

    # pool
    f = f * proj ./ w
    x = x * proj ./ w

    f, x = model.d2(GNNGraph(A), f, x)

    f = @showgrad(f) * proj'
    x = @showgrad(x) * proj'

    sum(f) + sum(x)
end


function train(; g=graph(OpenMMSimulation()), u=UNet(100), n=10, opt=Flux.setup(Adam(1.0f-4), u))
    loss(u) = sum(abs2, u(g).x-g.x[:,1:10])
    @show loss(u)
    for i in 1:n
        grad = gradient(loss, u)
        Flux.update!(opt, u, grad[1])
        @show loss(u)
    end
    return (;u,g,n,opt)
end

function test_gradients()
    g = graph(OpenMMSimulation())
    u = UNet(3, true)
    grad = gradient(u) do u
        sum(abs2, u(g).x - g.x)
    end
end


import Flux
Flux.@layer UNet





















### GRAVEYARD

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

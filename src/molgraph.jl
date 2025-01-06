using ISOKANN: OpenMMSimulation
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
    A = collect(Int, A)
    w = collect(km.wcounts')
    return A, collect(proj), w
end

function kmmpool(adj, x, f, n)
    #A, proj, w = @ignore_derivatives 

    A, proj, w = ignore_derivatives() do 
        kmmpoolmat(x,n, adj)
    end
    @show A
    
    f = f * proj 
    x = x * proj 

    return A, x, f, proj
end

# add dispatch for feature graph
function (l::GraphNeuralNetworks.EGNNConv)(g::GraphNeuralNetworks.GNNGraph)
    f, x = l(g, g.f, g.x)
    GNNGraph(g, ndata=(;f, x))
end

function unpool(g, (proj, gold) , residual=false)
    A = adjacency_matrix(gold)

    f = g.f * proj' + residual * gold.f
    x = g.x * proj' + residual * gold.x
        
    GNNGraph(A; ndata=(; x, f))
end

abstract type UNet end

struct UNet1 <: UNet
    d1
    d2
    u1
    u2
end

function UNet(nh=8)   
    UNet1(EGNNConv(1=>nh), EGNNConv(nh=>nh), EGNNConv(nh=>nh), EGNNConv(nh=>nh))
end

function (model::UNet)(g::GNNGraph)
    
    g = model.d1(g)
    g, un1 = kmmpool(g, 10)
    g = model.d2(g)
    #g = unpool(g, un1, false)
    
    
    g = model.u2(g)
    
    g = model.u1(g)
    return g
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

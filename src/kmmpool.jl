
using Zygote: @ignore_derivatives
using GraphNeuralNetworks
using CUDA
using Clustering

function kmmpool(g, k)
    A, proj, w, x = @ignore_derivatives _kmmpool(g, k)
    f = g.f * proj ./ w

    function unpool(gg)
        f_up = (gg.f - f) * proj' + g.f
        x_up = (gg.x - x) * proj' + g.x

        gup = GNNGraph(g, ndata=(x=x_up, f=f_up))
        return gup
    end

    gdown = GNNGraph(A, ndata=(; f, x), gdata=(; t=g.t))

    return gdown, unpool
end

function _kmmpool(g, k)
    adj = myadjacency(g)
    km = mykmeans(g.x, k)

    proj = projection(km.assignments)

    A = proj' * adj * proj
    A = zero_diagonal!(A)
    A = A .> 0

    w = km.wcounts'
    x = km.centers

    return (; A, proj, w, x)
end

mykmeans(x, k) = Clustering.kmeans(x, k)
function mykmeans(x::CuArray, k::Integer)
    km = Clustering.kmeans(cpu(x), k)
    centers = gpu(km.centers)
    assignments = gpu(km.assignments)
    wcounts = gpu(km.wcounts)
    return (; centers, assignments, wcounts)
end

projection(assignments) = SparseArrays.sparse(1:length(assignments), assignments, true) |> collect
projection(assignments::CuArray) = sparse(CuArray(1:length(assignments)), assignments, CUDA.ones(length(assignments)))

myadjacency(g) = adjacency_matrix(g)
myadjacency(g::GNNGraph{Tuple{CuArray{Int64,1,CUDA.DeviceMemory},CuArray{Int64,1,CUDA.DeviceMemory},CuArray{Bool,1,CUDA.DeviceMemory}}}) = Float32.(sparse(g.graph...))

function zero_diagonal!(x)
    x = x - Diagonal(x)
    return x
end

function zero_diagonal!(x::CUDA.CUSPARSE.CuSparseMatrixCSC)
    @cuda threads = length(x.colPtr) - 1 zero_kernel(x)
    return x
end

function zero_kernel(a)
    col = threadIdx().x
    for j in a.colPtr[col]:a.colPtr[col+1]-1
        if a.rowVal[j] == col
            a.nzVal[j] = 0
        end
    end
    return nothing
end
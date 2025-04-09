
using GraphNeuralNetworks, Flux, CUDA, Statistics, MLUtils
using Flux: DataLoader

all_graphs = GNNGraph[]

for _ in 1:1000
    g = rand_graph(10, 40,
        ndata=(; x=randn(Float32, 16, 10)),  # Input node features
        gdata=(; y=randn(Float32)))         # Regression target   
    push!(all_graphs, g)
end

device = gpu_device()

model = GNNChain(GCNConv(16 => 64),
    BatchNorm(64),     # Apply batch normalization on node features (nodes dimension is batch dimension)
    x -> relu.(x),
    GCNConv(64 => 64, relu),
    GlobalPool(mean),  # Aggregate node-wise features into graph-wise features
    Dense(64, 1)) |> device

opt = Flux.setup(Adam(1.0f-4), model)

train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.8)

train_loader = DataLoader(train_graphs,
    batchsize=32, shuffle=true, collate=true)
test_loader = DataLoader(test_graphs,
    batchsize=32, shuffle=false, collate=true)

loss(model, g::GNNGraph) = mean((vec(model(g, g.x)) - g.y) .^ 2)

loss(model, loader) = mean(loss(model, g |> device) for g in loader)

for epoch in 1:1000
    for g in train_loader
        g = g |> device
        grad = gradient(model -> loss(model, g), model)
        Flux.update!(opt, model, grad[1])
    end

    @info (; epoch, train_loss=loss(model, train_loader), test_loss=loss(model, test_loader))
end
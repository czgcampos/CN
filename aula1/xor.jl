using Flux
model = Flux.Chain(
Dense(14*16+4, 64, relu),
Dense(64, 16, relu),
Dense(16, 1, relu));
x = rand(Bool, 14*16+4)
y = 100
loss(x,y) = sum((model(x) .- y).^2)
opt = ADAM(params(model))
Flux.@epochs 100 Flux.train!(loss, [(x,y)], opt)

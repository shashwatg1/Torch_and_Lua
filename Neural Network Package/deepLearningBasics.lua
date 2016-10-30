-- Deep Learning with Torch
-- This tutorial gives an overview of the basics of torch along with Neural nets and its implementations

--[[
Before getting started

-> 'th' to compile
-> Only has one data structure built-in, a table: {}. Doubles as a hash-table and an array.
-> 1-based indexing.
-> foo:bar() is the same as foo.bar(foo)
-> Tensors can be moved onto GPU using the :cuda function
]]

torch.manualSeed(0)

-- Getting Started

-- Strings, numbers, tables - a tiny introduction

a = 'hello'
-- print(a)

b = {}
b[1]=a
-- print(b)

b[2]=30

for i=1,#b do -- the # operator is the length operator in Lua
--    print(b[i]) 
end

-- Tensors

a = torch.Tensor(5,3) -- construct a 5x3 matrix, uninitialized

a = torch.rand(5,3)
-- print(a)

b = torch.rand(3,4)

-- matrix-matrix multiplication: syntax 1
-- print(a*b)

-- matrix-matrix multiplication: syntax 2
-- print(torch.mm(a,b))

-- matrix-matrix multiplication: syntax 3
c=torch.Tensor(5,4)
c:mm(a,b) -- store the result of a*b in c
-- print(c)
 

-- Add two Tensors
function addTensors(a,b)
	y = torch.add(a, b) -- alternate
    return y
end

a = torch.ones(5,2)
b = torch.Tensor(2,5):fill(4)
-- print(a)
-- print(b)
-- print(addTensors(a,b))


-- Neural Networks

require 'nn'

net = nn.Sequential()
net:add(nn.SpatialConvolution(1, 6, 5, 5)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))  -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                 -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))            -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))     -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                  -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                  -- non-linearity 
net:add(nn.Linear(84, 10))         -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())         -- converts the output to a log-probability. Useful for classification problem

print('Lenet5\n' .. net:__tostring());

--[[
Every neural network module in torch has automatic differentiation.
It has a :forward(input) function that computes the output for a given input,
flowing the input through the network.
and it has a :backward(input, gradient) function that will differentiate each neuron
in the network w.r.t. the gradient that is passed in. This is done via the chain rule.
]]


input = torch.rand(1,32,32) -- pass a random tensor as input to the network
-- print('Input',input)

output = net:forward(input)
print(output)

net:zeroGradParameters()	 -- zero the internal gradient buffers of the network

gradInput = net:backward(input, torch.rand(10))
-- print(#gradInput)

--[[
need for a loss function:
In torch, loss functions are implemented just like neural network modules, and have automatic differentiation.
They have two functions - forward(input, target), backward(input, target)
]]

criterion = nn.ClassNLLCriterion() -- a negative log-likelihood criterion for multi-class classification
criterion:forward(output, 3) -- let's say the groundtruth was class number: 3
-- print({criterion})
gradients = criterion:backward(output, 3)
-- print(gradients)
gradInput = net:backward(input, gradients)
-- print(gradInput)

--[[
Quick Recap:
->  Network can have many layers of computation
->  Network takes an input and produces an output in the :forward pass
->  Criterion computes the loss of the network, and it's gradients w.r.t. the output of the network.
->  Network takes an (input, gradients) pair in it's backward pass and calculates the gradients
	w.r.t. each layer (and neuron) in the network.
-> 	A convolution layer learns it's convolution kernels to adapt to the input data and the problem being solved
->  A max-pooling layer has no learnable parameters. It only finds the max of local windows.
->  A layer in torch which has learnable weights, will typically have fields .weight (and optionally, .bias)
->  The gradWeight accumulates the gradients w.r.t. each weight in the layer,
	and the gradBias, w.r.t. each bias in the layer.
]]

m = nn.SpatialConvolution(1,3,2,2) -- learn 3 2x2 kernels
-- print(m.weight) -- initially, the weights are randomly initialized
-- print(m.bias) -- The operation in a convolution layer is: output = convolution(input,weight) + bias

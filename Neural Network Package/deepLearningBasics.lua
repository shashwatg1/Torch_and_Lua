-- Deep Learning with Torch

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
net:add(nn.View(16*5*5))              -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))       -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                    -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                    -- non-linearity 
net:add(nn.Linear(84, 10))            -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())              -- converts the output to a log-probability. Useful for classification problems

print('Lenet5\n' .. net:__tostring());

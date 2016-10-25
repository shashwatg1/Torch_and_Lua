-- Training of any neural network basics
-- 'th' to compile

require 'nn'
require 'optim'
torch.manualSeed(1234)

local model = nn.Sequential()
local n = 2 -- 2 input features
local K = 1 -- 1 scalar output
local s = {n, 10, K} -- 3 layer neural network

model:add(nn.Linear(s[1], s[2]))
model:add(nn.Tanh())
model:add(nn.Linear(s[2], s[3]))

-- print(model)

local loss = nn.MSECriterion()

local m = 128 -- say 128 instances in the dataset
local X = torch.DoubleTensor(m, n)
local Y = torch.DoubleTensor(m)

for i=1, m do
	local x = torch.randn(2) -- 2 elements
	local y = x[1] * x[2] > 0 and -1 or 1  -- imitating XOR gate
	X[i]:copy(x)
	Y[i] = y
end

local theta, gradTheta = model:getParameters()

local optimState = {learningRate = 0.15}

for epoch = 1, 1e3 do
	function feval(theta)
		gradTheta:zero()
		local h_x = model:forward(X)
		local J = loss:forward(h_x, Y)
		print(J) -- for debugging purposes
		local dJ_dh_x = loss:backward(h_x,Y)
		model:backward(X, dJ_dh_x) -- computes and updates gradTheta
		return J, gradTheta
	end
	optim.sgd(feval, theta, optimState)
end

net = model

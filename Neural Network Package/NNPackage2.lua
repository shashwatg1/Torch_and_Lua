-- use th to compile

torch.manualSeed(1234)

require 'nn'
require 'gnuplot'

-- X : Design matrix, it has size m X n
-- Y : labels / target matrix / vector, it has size m X K
-- etha : learning rate

-- SGD
for i=1,m do
	local pred = net:forward(X[i])
	local err = loss:forward(pred, Y[i])
	local gradLoss = loss:backward(pred, Y[i])
	net:zeroGradParameters()
	net:backward(X[i], gradLoss)
	net:updateParameters(etha)
end

-- Mini-batch GD
for i=1,m,batchSize do
	net:zeroGradParameters()
	for j=0,batchSize-1 do
		if i+j > m then break end
		local pred = net:forward(X[i])
		local err = loss:forward(pred, Y[i])
		local gradLoss = loss:backward(pred, Y[i])
		net:backward(X[i], gradLoss)
	end
	net:updateParameters(etha)
end


local dataset = {}

function dataset:size() return m end

for i=1,m do
	dataset[i] = {X[i], Y[i]}
end

local trainer = nn.StochasticGradient(net,loss)
trainer:train(dataset)

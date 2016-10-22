-- use th to compile

torch.manualSeed(1234)

require 'nn'
require 'gnuplot'

n = 5  -- 5 neurons (dimension of X without X0)
K = 3  -- dimensionality of the output (H_theta(x))

lin = nn.Linear(n,K)

--[[
module = nn.Linear(inputDimension, outputDimension, [bias = true])
Applies a linear transformation to the incoming data, i.e. y = Ax + b.
The input tensor given in forward(input) must be either a vector (1D tensor) or matrix (2D tensor).
If the input is a matrix, then each row is assumed to be an input sample of given batch.
The layer can be used without bias by setting bias = false
]]

print(lin) -- nn.Linear(5 -> 3)
-- print({lin}) -- prints components of the lin

print('Weights')
print(lin.weight)  -- they are randomly initialised
print('Bias')
print(lin.bias)  -- they are randomly initialised

Theta_1 = torch.cat(lin.bias, lin.weight, 2) -- concatenate column wise

-- gradWeight and gradBias are also part of linear which represent dE/dW and dE/db respectively (E is the cost function)

lin:zeroGradParameters() -- to make the grad params zero (should be done manually)
gradTheta_1 = torch.cat(lin.gradBias, lin.gradWeight, 2) -- concatenate column wise
-- print(gradTheta_1)

sig = nn.Sigmoid() --Applies the Sigmoid function element-wise to the input Tensor, thus outputting a Tensor of the same dimension.
-- print(sig) -- prints nn.Sigmoid
-- print({sig}) -- prints components of the sig

--z = torch.linspace(-10,10,21)
--gnuplot.plot(z,sig:forward(z)) -- plots the sigmoid graph for -10 to 10


-- Forward Pass
x = torch.randn(n) -- input

print('Input')
print(x)

a1 = x

h_Theta = sig:forward(lin:forward(x)) -- output

-- lin:forward does y = ax + b. It multiplies the weights with x and then adds the bias terms
-- sid:forward then applies sigmoid function element wise

print('Output FP')
print(h_Theta)


-- The following will do the same as the line sig:forward(lin:forward(a1))
z2 = Theta_1 * torch.cat(torch.ones(1),a1,1)  -- z2 = theta(1)*a1 (with a 1 added for x0) -- same as lin:forward(x)
a2 = z2:clone():apply(
	function(z)
		return 1/(1+math.exp(-z))
	end)

--print(a2) -- same as h_Theta



-- Backward Pass or Backpropagation

 -- we need to define a loss function
loss = nn.MSECriterion()
--[[
Creates a criterion that measures the mean squared error between n elements 
in the input x and output y :
 loss(x, y) = 1/n \sum |x_i - y_i|^2 .

print(loss) -- nn.MSECriterion
print({loss})
]]

loss.sizeAverage = false -- prevents the division by 'n' in the loss calculation formula

y = torch.rand(K) -- Desired Output

print('Desired Output')
print(y)

E = loss:forward(h_Theta, y) -- calculates the error / loss
print('Error')
print(E)

--[[
E can also be calculated in the following way:

E1 = (h_Theta - y):pow(2):sum()
print(E1)
]]

dE_dh = loss:updateGradInput(h_Theta, y) -- calculates the derivatve of E wrt h_Theta
-- it is the same as calculating 2*(h_Theta - y)

print('Gradient')
print(dE_dh)

delta_2 = sig:updateGradInput(z2, dE_dh)
-- it is same as dE_dh:clone():cmul(a2):cmul(1 - a2)

print('Delta 2')
print(delta_2)

lin:accGradParameters(x, delta_2)

gradTheta_1 = torch.cat(lin.gradBias, lin.gradWeight, 2) -- concatenate column wise
print(gradTheta_1) -- shows all the gradients of the loss wrt to the bias and weights of theta_1

lin_gradInput = lin:updateGradInput(x,delta_2) -- same as lin.weight:t() * 	delta_2

-- thus we calculate all the elements needed for back propagation
-- this was however veyr chaotic and we want to do it more formally and conviniently



-- Seqential Model

-- we define a network which is a container named Sequential
net = nn.Sequential() -- allows us a sequential sequence of blocks one after another

net:add(lin)
net:add(sig)

print(net)

pred = 	net:forward(x)	-- prediction
print(pred) -- it will be same as h_Theta

err = loss:forward(pred, y)
print(err) -- same as before

gradCriterion = loss:backward(pred, y)
print(gradCriterion) -- same as gradient dE_dh before

-- to access any block of net, use net:get()
print(torch.cat(net:get(1).gradBias, net:get(1).gradWeight, 2))

net:zeroGradParameters() -- makes all gradients zero
print(torch.cat(net:get(1).gradBias, net:get(1).gradWeight, 2))

net:backward(x, gradCriterion) -- again computes them
print(torch.cat(net:get(1).gradBias, net:get(1).gradWeight, 2)) -- dE_dh again

-- to train the parameters, we define a learning rate:
etha = 0.01

net:updateParameters(etha) -- to update the parameters

dE_dTheta_1 = torch.cat(net:get(1).gradBias, net:get(1).gradWeight, 2)
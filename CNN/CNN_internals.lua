-- qlua -lenv to compile

torch.manualSeed(0)
require 'nn'
require 'pretty-nn'
require 'image'

net = require 'Torch7-profiling/models.3conv-pool'
-- print(net)

x = image.scale(image.lena(), 256, 256)
-- print(#x) -- 3 X 256 X 256
-- image.display{image = x, legend = 'x'}
print(#net:forward(x)) -- 1000

-- image.display{image = net:get(1).weight, legend = 'k(1)', zoom = 18, padding = 2} -- 6 kernels of the first Conv layer

function show(l,t)
	print(t)
	image.display{image = l.output, legend = t, scaleeach = true}
end

--[[
show(net:get(1),'y(1)')
show(net:get(2),'y(1)+')
show(net:get(3),'y(2)')
show(net:get(4),'y(2)+')
show(net:get(5),'y(2)+pool')
show(net:get(6),'y(3)')
show(net:get(7),'y(3)+')
show(net:get(8),'y(3)+pool')
]]

-- print(net:get(9))
-- print(net:get(10))

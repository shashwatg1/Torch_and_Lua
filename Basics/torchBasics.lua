torch.manualSeed(1234)

a = torch.FloatTensor(2,3) -- 2X3 Tensor

print(a) -- garbage value

a:fill(1)

print(a) -- all 1

a:uniform(0,1)

print(a) -- uniform distribution

b = a:transpose(1,2) -- transpose of dimensions 1 and 2

print(b) -- transpose of a

-- 'a' and 'b' share the same underlying storage

b[{1,2}]=1 -- affects 'a' aswell as they have the same storage

print(a) -- true
print(b) -- transpose of 'a'

print(a:storage()) -- printed row wise
print(b:storage()) -- same as 'a'

print(a:isContiguous()) -- true
print(b:isContiguous()) -- false

-- 'a' is contiguos (storage is according to the row-wise order) but b is not as b inherits its underlying storage from 'a'
-- to make b contiguos, we must clone it or copy it in a new empty FloatTensor

c = b:clone() -- allocates a new storage for c

print(c) --same as b
print(c:storage()) -- different order from b
print (c:isContiguous()) -- true

d = b.new() -- initialises an empty dimensionless Tensor
d:resize(b:size())
d:copy(b)

print(d) -- same as c
print(d:storage()) -- same as c
print(d:isContiguous()) -- true

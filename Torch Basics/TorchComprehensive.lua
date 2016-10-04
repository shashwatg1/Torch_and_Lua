---- 1. Getting Help

-- typing '?' on terminal after typing th provides some help about torch
-- typing '? torch.randn' gives help about that

-- while working on th in terminal, we can access the linux command line by typing '$' followed by the command


----- 2. Tensors and types

print(torch.type('Hi')) -- prints string
print(torch.type(10)) -- prints number
print(torch.type(torch)) -- prints table

t = torch.Tensor(2,3,4) -- 2 matrices of size 3 X 4
print (#t) -- dimension of t

i = 0
t:apply(function () i = i+1; return i; end)

print(t)
print(torch.type(t)) -- torch.DoubleTensor (default)

-- to change the default Tensor type:
torch.setdefaulttensortype('torch.FloatTensor')

print(torch.type(torch.Tensor(1,2,3))) -- torch.FloatTensor

-- we can define the tensor type explicitly by using torch.FloatTensor(1,2,3), etc


-- What is a Tensor?
-- a tensor is simply a view on a specific underlying storage (a C array)

r = torch.DoubleTensor(t):resize(3,8) 
print(r) -- 1 matrix of size 3 X 8

r:zero() -- this makes t zero aswell
print(t)

s = t
print(#t)
s:resize(4,6)
print(#t) -- t is resized because of s

-- thus in case of tensors, assignment between Tensors is simply a copy of its reference!!!

y = t:clone() -- now the values are copied but the reference is not passed. changes to y are independent

-- Some common ways of initialisation include:
a = torch.zeros(3,5) -- 3 X 5 matrix
a = torch.ones(2,3,4) -- 2  3 X 4 matrices
a = torch.eye(3) -- 3 X 3 identity matrix
a = torch.Tensor(3,5):random(10) -- 3 X 5 random integers from 1 to 10

-- type casting is done like this:
print(torch.type(a)) -- float
b = a:double()
print(torch.type(b)) -- double
print(torch.type(a)) -- dloat

---- 3. Vectors (1 dimensional Tensors)

v = torch.Tensor{1,2,3,4}
print (v, #v, v:dim(), v:size(1), v[3], v[-2]) -- #v is 4 because 4 elements v:dim() is 1, v:size(1) is 4
											-- , v[3] is 3rd element from start, v[-2] is second last

w = torch.ones(4) -- vector of 4 ones

print(v * w) -- inner product

print (v[{{2,4}}]) -- extracts 2nd to 4th (inclusive)

v = torch.range(1,5)
v:pow(2)
print(v) -- first 5 squares

v = torch.range(1,5)
w = torch.pow(v,2) -- alternate way


---- 4. Matrix

m = torch.Tensor{ {9,6,3,4}, {7,2,8,1} }
print(m)

-- to extract a given element use:
print(m[2][3], m[{2,3}])

-- to extract entire column or row:
print( m[{{},{1}}] )
print( m[{{1},{}}] )
-- but these are both 2 dimensional Tensors (not vectors)

-- to extract vectors (1 dimensional):
print( m[{{}, 1}] )
print( m[{1, {}}] )

-- matrix vector multiplication
v = torch.Tensor{1,2,3,4}
print(m*v) -- it will be a vector

print(torch.range(3,-4.2,-1.9)) -- go from 3 to at most -4.2 with steps of -1.9
print(torch.linspace(3,8,10)) -- place 10 equispaced points between 3 and 8 inclusive

-- matrix matrix multiplication
-- m is 2 X 4
print(m * torch.rand(4,6)) -- 2 X 6 matrix

-- dot multiply operation
m1 = torch.Tensor{ {8,7,6,5}, {4,3,2,1} }
m2 = torch.Tensor{ {1,2,3,4}, {5,6,7,8} }

print(m1:cmul(m2))

-- transpose
print(m:t())

-- for multiple dimension tensors:
v = torch.range(1,24):resize(3,4,2)
print(v)
print(v:transpose(1,3)) -- swapping the 1st and 3rd dimensions. Notice the order of elements stored
v = torch.range(1,24):resize(3,4,2)
print(v:resize(2,4,3)) -- gives a different result in terms of the values stored. It is ordered from scratch


-- concatenation operations:
m1 = torch.Tensor{{1,2,3,4}} -- 1 X 4
m2 = torch.Tensor{{5,6,7,8}} -- 1 X 4
print(torch.cat(m1,m2,1)) -- 2 X 4
print(torch.cat(m1,m2,2)) -- 1 X 8


-- Mathematical operations:
a = torch.Tensor(3,5):random(10)
print(a)

b = a*2 -- a is not replaced with a * 2
b = a:mul(2) -- not a is replaced with a * 2
b = a+1 -- a is not replaced with a + 1
b = a:add(1) -- not a is replaced with a + 1
-- similarly we have div(), pow()
print(a)

-- To apply changes to only a part of 'a':
a = torch.ones(3,7)
print(a)
a[{1, {2,5}}]:mul(2) -- only the 2nd to 5th elements of the 1st row are affected
print(a)


---- 5. resize, reshape, view difference

a = torch.range(1,12)
print(a)
b = a:reshape(3,4) -- does not affect a
c = a:view(4,3) -- does not affect a
a:resize(2,6) -- affects a
print (a, b, c)

-- now say we do the following:
a:add(2)

print(a,b,c)
-- only a and c are affected
-- hence we can conclude that:
--	resize affect the mother Tensor and the storage
--	reshape doesnt affect mother and also has its own independent storage
--	view doesnt affect mother but it uses the mothers storage
--	a and c are two different views of the same storage
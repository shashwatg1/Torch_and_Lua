-- 1. Define a positive definite quadratic form

torch.manualSeed(1234)

-- choose a dimension
N = 5

-- create a random NxN matrix
A = torch.rand(N, N)

print (A)

-- make it symmetric positive
A = A*A:t()

print (A)

-- make it definite
A:add(0.001, torch.eye(N))

print (A)

-- add a linear term
b = torch.rand(N)

print (b)

-- create the quadratic form
function J(x)
   return 0.5*x:dot(A*x)-b:dot(x)
end

print(J(torch.rand(N)))


-- 2. Find the exact minimum

xs = torch.inverse(A)*b
print(string.format('J(x^*) = %g', J(xs)))


-- 3. Search the minimum by gradient descent

--define the gradient w.r.t. x of J(x)
function dJ(x)
  return A*x-b
end

x = torch.rand(N)

--apply gradient descent (with a given learning rate lr) for a while:
lr = 0.01
for i=1,20000 do
  x = x - dJ(x)*lr
  -- we print the value of the objective function at every 1000 iterations
  if i - math.floor(i/1000)*1000 == 0 then
  	print(string.format('at iter %d J(x) = %f', i, J(x)))
  end
end

-- 4. Using the optim package

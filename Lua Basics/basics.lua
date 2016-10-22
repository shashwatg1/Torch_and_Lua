-- use lua to compile these files in terminal

num = 42  -- All numbers are doubles.
-- Don't freak out, 64-bit doubles have 52 bits for
-- storing exact int values; machine precision is
-- not a problem for ints that need < 52 bits.

s = 'walternate'  -- Immutable strings like Python.
t = "double-quotes are also fine"
u = [[ Double brackets
       start and end
       multi-line strings.]]

print(s,t)
print(u)

t = nil  -- Undefines t; Lua has garbage collection.

print (t)


-- Blocks are denoted with keywords like do/end:
while num < 50 do
  num = num + 1  -- No ++ or += type operators.
end


-- If clauses:
if num == 40 then
  print('equals 40')
elseif s ~= 'walternater' then  -- ~= is not equals.
  -- Equality check is == like Python; ok for strs.
  io.write('not over 40\n')  -- Defaults to stdout.
else
  -- Variables are global by default.
  thisIsGlobal = 5  -- Camel case is common.

  -- How to make a variable local:
  local line = io.read()  -- Reads next stdin line.

  -- String concatenation uses the .. operator:
  print('Winter is coming, ' .. line)
end

-- Undefined variables return nil.
-- This is not an error:
foo = anUnknownVariable  -- Now foo = nil.

print (foo)

aBoolValue = false

-- Only nil and false are falsy; 0 and '' are true!
if not aBoolValue then print('twas false') end


-- 'or' and 'and' are short-circuited.
-- This is similar to the a?b:c operator in C/js:
ans = aBoolValue and 'yes' or 'no'  --> 'no'

print(ans)


karlSum = 0
for i = 1, 100 do  -- The range includes both ends.
  karlSum = karlSum + i
end
print(karlSum)

-- Use "100, 1, -1" as the range to count down:
fredSum = 0
for j = 100, 1, -2 do
  fredSum = fredSum + j
end
print(fredSum)
-- In general, the range is begin, end[, step].

-- Another loop construct:
num = 10
repeat
  print('the way of the future')
  num = num - 1
until num == 0
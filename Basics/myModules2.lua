-- this file contains a module. It is used by useModules.lua

local M = {}

print('Hi!')

local function sayMyName()
  print('Shashwat')
end

function M.sayHello()
  print('Hello there')
  sayMyName()
end

return M
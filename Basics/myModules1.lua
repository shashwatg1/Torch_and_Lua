-- this file contains a module. It is used by useModules.lua

local M = {}

local function sayMyName()
  print('Shashwat')
end

function sayWhat()
	print('Say What!!!')
end

function M.sayHello()
  print('Hello there')
  sayMyName()
end

return M
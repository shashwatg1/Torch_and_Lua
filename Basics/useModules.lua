-- We can use the modules and the functionality of other files:

local mod = require('myModules1')  -- Run the file contianing modules

-- require is the standard way to include modules.

-- require acts like:     (if not cached; see below)
-- local mod = (function ()
--  <contents of myModules1.lua>
-- end)()

-- It's like myModules1.lua is a function body, so that
-- locals inside myModules1.lua are invisible outside it.

-- This works because 'mod' here = 'M' in myModules1.lua:
mod.sayHello()  -- Says hello to Hrunkner.

-- This is wrong; sayMyName only exists in myModules1.lua:
-- mod.sayMyName() will give error because it is local to that lua file

sayWhat() -- can be called as it is not local

-- require's return values are cached so a file is
-- run at most once, even when require'd many times.

print('Module 2 now')

-- Suppose myModules2.lua contains "print('Hi!')".
local a = require('myModules2')  -- Prints Hi!
local b = require('myModules2')  -- Doesn't print; a=b.

print('check')

-- dofile is like require without caching:
dofile('myModules2.lua')  --> Hi!
dofile('myModules2.lua')  --> Hi! (runs it again)

-- loadfile loads a lua file but doesn't run it yet.
f = loadfile('myModules2.lua')  -- Call f() to run it.

-- loadstring is loadfile for strings.
g = loadstring('print(343)')  -- Returns a function.
g()  -- Prints out 343; nothing printed before now.
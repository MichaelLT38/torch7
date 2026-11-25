-- CTest driver: run a standalone Lua test file against the staged build tree.
--
-- Usage: luajit run_lua_file.lua <staged lua dir> <module dir> <module suffix> <test.lua>
--
-- The test file is expected to raise an error (non-zero exit) on failure,
-- as torch.Tester:run() does.
local luaDir, modDir, modSuffix, script = arg[1], arg[2], arg[3], arg[4]
assert(luaDir and modDir and modSuffix and script,
       "usage: run_lua_file.lua <lua dir> <module dir> <module suffix> <test.lua>")
package.path = luaDir .. "/?.lua;" .. luaDir .. "/?/init.lua;" .. package.path
package.cpath = modDir .. "/?" .. modSuffix .. ";" .. package.cpath
arg = { [0] = script }
dofile(script)

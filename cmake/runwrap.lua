-- Build-time driver for cwrap binding scripts (TensorMath.lua, random.lua).
-- Usage: lua runwrap.lua <torch source root> <wrapper.lua> <output.c>
--
-- Makes the in-tree copies of cwrap (lib/cwrap) and torchcwrap.lua resolvable
-- without installing anything, then runs the wrapper with arg[1] = output.c.
local root, script, output = arg[1], arg[2], arg[3]
assert(root and script and output, "usage: runwrap.lua <root> <wrapper.lua> <output.c>")

package.path = root .. "/lib/?/init.lua;" .. root .. "/lib/?.lua;"
            .. root .. "/?.lua;" .. package.path

arg = { [0] = script, output }
dofile(script)

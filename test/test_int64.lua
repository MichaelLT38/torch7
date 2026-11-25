-- 64-bit integer behavior visible from Lua (ctest: lua.int64).
--
-- Usage: luajit test_int64.lua <staged lua dir> <module dir> <module suffix>
--
-- Since Phase 1, LongTensor/LongStorage hold int64_t on every platform (C long
-- is only 32 bits on Windows), and a binary file's native long size is 8 bytes
-- everywhere, so files written on Windows and Linux use the same layout.
local luaDir, modDir, modSuffix = arg[1], arg[2], arg[3]
assert(luaDir and modDir and modSuffix, "usage: test_int64.lua <lua dir> <module dir> <suffix>")
package.path = luaDir .. "/?.lua;" .. luaDir .. "/?/init.lua;" .. package.path
package.cpath = modDir .. "/?" .. modSuffix .. ";" .. package.cpath
require 'torch'

local tester = torch.Tester()
local tests = torch.TestSuite()

local T40 = 2^40   -- exactly representable in a Lua number (double)

function tests.longtensor_holds_values_beyond_32_bits()
   local t = torch.LongTensor({T40, -T40 + 5, 2^52, 3})
   tester:eq(t[1], T40)
   tester:eq(t[2], -T40 + 5)
   tester:eq(t[3], 2^52)
   tester:eq(t:sum(), T40 + (-T40 + 5) + 2^52 + 3)
   tester:eq((t * 2)[1], 2 * T40)
   local v, i = t:max(1)
   tester:eq(v[1], 2^52)
   tester:eq(i[1], 3)
   tester:eq(torch.LongStorage({T40})[1], T40)
end

function tests.native_binary_long_is_8_bytes()
   local f = torch.MemoryFile():binary()
   f:writeLong(1)
   tester:eq(f:position(), 9, "a native long takes 8 bytes on every platform")
   f:close()
end

function tests.binary_roundtrip_memoryfile()
   local t = torch.LongTensor({T40, -T40, 7})
   local f = torch.MemoryFile():binary()
   f:writeObject(t)
   f:seek(1)
   local u = f:readObject()
   f:close()
   tester:eq(torch.type(u), 'torch.LongTensor')
   tester:eq(u[1], T40)
   tester:eq(u[2], -T40)
   tester:eq(u[3], 7)
end

function tests.binary_roundtrip_diskfile()
   local path = os.tmpname()
   local t = torch.LongTensor({T40, -3, 2^50})
   torch.save(path, t)
   local u = torch.load(path)
   os.remove(path)
   tester:eq(u[1], T40)
   tester:eq(u[2], -3)
   tester:eq(u[3], 2^50)
end

function tests.serialize_roundtrip()
   local t = torch.LongTensor({T40, 11})
   local u = torch.deserialize(torch.serialize(t))
   tester:eq(u[1], T40)
   tester:eq(u[2], 11)
end

-- Files with 4-byte longs (e.g. written by Windows Torch before Phase 1) can
-- still be read with file:longSize(4).
function tests.longsize_4_still_readable()
   local f = torch.MemoryFile():binary():longSize(4)
   f:writeLong(123456)
   f:writeLong(-7)
   tester:eq(f:position(), 9, "two 4-byte longs")
   f:seek(1)
   tester:eq(f:readLong(), 123456)
   tester:eq(f:readLong(), -7)
   f:close()
end

function tests.ascii_roundtrip()
   local f = torch.MemoryFile():ascii()
   f:writeLong(T40)
   f:writeLong(-T40)
   f:seek(1)
   tester:eq(f:readLong(), T40)
   tester:eq(f:readLong(), -T40)
   f:close()
end

function tests.size_values_beyond_32_bits_in_storage()
   -- sizes travel through LongStorage; no allocation of that size needed
   local s = torch.LongStorage({2^31 + 5, 3})
   tester:eq(s[1], 2^31 + 5)
end

tester:add(tests)
tester:run()

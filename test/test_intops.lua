-- Integer shift / division semantics from Lua (ctest: lua.intops).
--
-- Usage: luajit test_intops.lua <staged lua dir> <module dir> <module suffix>
--
-- Contract (agreed with the LatticeAI project, see README):
--   rshift / crshift          LOGICAL on signed types (zero-filling)
--   arshift / carshift        ARITHMETIC: floor(x / 2^s), rounds toward -inf
--   lshift / clshift          wraps
--   shift amounts             must be in [0, bits-1], else error; 0 = identity
--   div / cdiv, fmod / cfmod  truncate toward zero (C semantics)
--   floordiv / cfloordiv,
--   remainder / cremainder    floor:  a == floordiv(a,b)*b + remainder(a,b)
--   integer division by zero  error;  MIN / -1 wraps to MIN
local luaDir, modDir, modSuffix = arg[1], arg[2], arg[3]
assert(luaDir and modDir and modSuffix, "usage: test_intops.lua <lua dir> <module dir> <suffix>")
package.path = luaDir .. "/?.lua;" .. luaDir .. "/?/init.lua;" .. package.path
package.cpath = modDir .. "/?" .. modSuffix .. ";" .. package.cpath
require 'torch'

local tester = torch.Tester()
local tests = torch.TestSuite()

local function list(t)
   local out = {}
   for i = 1, t:nElement() do out[i] = t[i] end
   return out
end

function tests.arshift_is_floor()
   local t = torch.IntTensor({-8, -7, -1, 0, 1, 7, 8})
   tester:eq(list(torch.arshift(t, 1)), {-4, -4, -1, 0, 0, 3, 4})
   tester:eq(list(t:clone():arshift(2)), {-2, -2, -1, 0, 0, 1, 2})
   tester:eq(list(torch.arshift(t, 0)), list(t), "shift by 0 is the identity")
   tester:eq(torch.arshift(torch.IntTensor({-2147483648}), 31)[1], -1)
   tester:eq(torch.arshift(torch.LongTensor({-2^40}), 40)[1], -1)
end

function tests.rshift_stays_logical()
   local t = torch.IntTensor({-4, -1, 5})
   tester:eq(list(torch.rshift(t, 1)), {2147483646, 2147483647, 2})
   tester:eq(list(torch.CharTensor({-4}):rshift(1)), {126})
end

function tests.carshift_elementwise()
   local t = torch.IntTensor({-8, -8, 8, -1})
   local s = torch.IntTensor({1, 2, 1, 31})
   tester:eq(list(torch.carshift(t, s)), {-4, -2, 4, -1})
end

function tests.shift_amount_out_of_range_errors()
   local t = torch.IntTensor({1, 2})
   for _, f in ipairs({torch.rshift, torch.lshift, torch.arshift}) do
      tester:assertErrorPattern(function() f(t, 32) end, 'out of range')
      tester:assertErrorPattern(function() f(t, -1) end, 'out of range')
   end
   tester:assertErrorPattern(function() torch.arshift(torch.CharTensor({1}), 8) end, 'out of range')
   tester:assertErrorPattern(function() torch.carshift(t, torch.IntTensor({1, 40})) end, 'out of range')
   tester:assertErrorPattern(function() torch.crshift(t, torch.IntTensor({-1, 1})) end, 'out of range')
end

function tests.arshift_rejects_float()
   tester:assertErrorPattern(function() torch.arshift(torch.Tensor({1}), 1) end, 'integer tensors')
end

function tests.floordiv_and_remainder_pair()
   local a = torch.IntTensor({-7, -7, 7, 7, 0, -1})
   local b = torch.IntTensor({2, -2, 2, -2, 3, 5})
   local q, r = torch.cfloordiv(a, b), torch.cremainder(a, b)
   tester:eq(list(q), {-4, 3, 3, -4, 0, -1})
   tester:eq(list(r), {1, -1, 1, -1, 0, 4})
   for i = 1, a:nElement() do
      tester:eq(q[i] * b[i] + r[i], a[i], 'a == floordiv*b + remainder')
   end
   tester:eq(list(torch.floordiv(torch.IntTensor({-7, 7}), 2)), {-4, 3})
   tester:eq(list(torch.div(torch.IntTensor({-7, 7}), 2)), {-3, 3}, "div truncates")
   tester:eq(list(torch.fmod(torch.IntTensor({-7, 7}), 2)), {-1, 1}, "fmod truncates")
   -- floating point floordiv is floor(a / b)
   tester:eq(list(torch.floordiv(torch.Tensor({-7, 7}), 2)), {-4, 3})
end

function tests.min_div_minus_one_wraps()
   local MIN = torch.IntTensor({-2147483648})
   tester:eq(torch.div(MIN, -1)[1], -2147483648)
   tester:eq(torch.floordiv(MIN, -1)[1], -2147483648)
   tester:eq(torch.fmod(MIN, -1)[1], 0)
   tester:eq(torch.remainder(MIN, -1)[1], 0)
   tester:eq(torch.cdiv(MIN, torch.IntTensor({-1}))[1], -2147483648)
end

function tests.integer_division_by_zero_errors()
   local a, z = torch.IntTensor({1, 2}), torch.IntTensor({1, 0})
   for _, f in ipairs({torch.div, torch.fmod, torch.floordiv, torch.remainder}) do
      tester:assertErrorPattern(function() f(a, 0) end, 'division by zero')
   end
   for _, f in ipairs({torch.cdiv, torch.cfmod, torch.cfloordiv, torch.cremainder}) do
      tester:assertErrorPattern(function() f(a, z) end, 'division by zero')
   end
   -- (Tester:eq compares |a - b| <= tol, which is NaN for inf - inf)
   tester:assert(torch.div(torch.Tensor({1}), 0)[1] == math.huge, "floating point still follows IEEE")
end

function tests.signed_overflow_wraps()
   local t = torch.IntTensor({2147483647})
   tester:eq((t + 1)[1], -2147483648, "add wraps (two's complement)")
   tester:eq((t * 2)[1], -2, "mul wraps")
end

-- LatticeAI's requantization: round half up, then floor shift. arshift must
-- match the double-precision floor workaround bit for bit.
function tests.requant_matches_floor_workaround()
   torch.manualSeed(7)
   local acc = torch.IntTensor(10000):random(-2^30, 2^30)
   for s = 1, 31 do
      local t = acc + (2^(s - 1))
      local viaShift = torch.arshift(t, s)
      local viaFloor = torch.floor(t:double() / 2^s):int()
      tester:assert(viaShift:equal(viaFloor), 'requant mismatch at shift ' .. s)
   end
end

tester:add(tests)
tester:run()

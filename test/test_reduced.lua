-- Phase 4: reduced-precision floating types (HalfTensor, BFloat16Tensor).
--
-- Contract: a reduced type stores 16 bits and computes in float. So for every
-- operation, the result must equal the FloatTensor result on the same values,
-- rounded once to the reduced type:
--   * exactly, for single correctly rounded operations (+ - * /, sqrt, abs,
--     floor, comparisons, selection, copies), for reductions along a dimension
--     (they run on a float copy) and for mm on contiguous inputs (same float GEMM);
--   * within 1 ulp of the reduced type where the FloatTensor path itself rounds
--     differently (vectorized exp/log/tanh/sigmoid, libm transcendentals,
--     compound expressions, other summation orders), since a float result a few
--     float ulps away can round to the neighbouring 16-bit value.
--
-- Usage: luajit run_lua_file.lua <lua dir> <module dir> <suffix> test_reduced.lua
require 'torch'

local tester = torch.Tester()
local tests = torch.TestSuite()

-- mant: explicit mantissa bits; emin: smallest normal exponent; max: largest finite
local TYPES = {
   {name = 'Half', tensor = 'torch.HalfTensor', mant = 10, emin = -14, max = 65504},
   {name = 'BFloat16', tensor = 'torch.BFloat16Tensor', mant = 7, emin = -126, max = 3.3895313892515355e38},
}

local function ulp(info, x)
   x = math.abs(x)
   if x == 0 then return 2 ^ (info.emin - info.mant) end
   local e = math.floor(math.log(x) / math.log(2))
   if 2 ^ e > x then e = e - 1 elseif 2 ^ (e + 1) <= x then e = e + 1 end   -- log rounding
   if e < info.emin then e = info.emin end
   return 2 ^ (e - info.mant)
end

-- got: reduced tensor; want: FloatTensor result of the same operation.
-- Returns nil if every element matches within `ulps` reduced ulps, else a message.
local function compare(info, got, want, ulps)
   if torch.type(got) ~= info.tensor then
      return string.format('result type %s, expected %s', torch.type(got), info.tensor)
   end
   local gs, ws = table.concat(got:size():totable(), 'x'), table.concat(want:size():totable(), 'x')
   if gs ~= ws then
      return 'result size ' .. gs .. ', expected ' .. ws
   end
   local g = got:float():contiguous():view(-1)
   local e = want:type(info.tensor):float():contiguous():view(-1)   -- rounded once
   for i = 1, g:nElement() do
      local a, b = g[i], e[i]
      local ok
      if b ~= b then ok = (a ~= a)
      elseif b == math.huge or b == -math.huge then ok = (a == b)
      else ok = math.abs(a - b) <= ulps * ulp(info, b) end
      if not ok then
         return string.format('element %d: got %.10g, expected %.10g (float result %.10g, %d ulp allowed)',
                              i, a, b, want:contiguous():view(-1)[i], ulps)
      end
   end
end

local function check(info, what, got, want, ulps)
   local msg = compare(info, got, want, ulps or 0)
   tester:assert(msg == nil, info.name .. ' ' .. what .. ': ' .. tostring(msg))
end

-- an m x n reduced tensor with values uniform in [lo, hi], in a given layout,
-- plus the FloatTensor holding exactly the same values
local function make(info, layout, m, n, lo, hi)
   local f = torch.FloatTensor(m, n):uniform(lo, hi):type(info.tensor):float()
   local r
   if layout == 'contiguous' then
      r = f:type(info.tensor)
   elseif layout == 'transposed' then
      r = f:t():contiguous():type(info.tensor):t()
   else -- 'strided': every other column of a wider tensor
      local wide = torch.FloatTensor(m, 2 * n):zero()
      for j = 1, n do wide:select(2, 2 * j - 1):copy(f:select(2, j)) end
      local w = wide:type(info.tensor)
      r = torch[info.name .. 'Tensor'](w:storage(), 1, torch.LongStorage{m, n}, torch.LongStorage{2 * n, 2})
   end
   assert(r:float():equal(f), 'make: layout ' .. layout .. ' does not hold the values')
   return r, f
end

local LAYOUTS = {'contiguous', 'transposed', 'strided'}

-- -0 made at run time: Lua 5.1 merges the constant -0.0 with an earlier 0 (they compare equal)
local NEGZERO = (function() local z = 0; return -z end)()
assert(1 / NEGZERO < 0)

local function forTypes(fn)
   return function()
      for _, info in ipairs(TYPES) do
         torch.manualSeed(1234)
         fn(info)
      end
   end
end

-- ---------------------------------------------------------------------------
-- element-wise unary
-- ---------------------------------------------------------------------------
tests.unary = forTypes(function(info)
   local exact = {'abs', 'neg', 'floor', 'ceil', 'round', 'trunc', 'frac', 'sign', 'cinv'}
   local close = {'exp', 'tanh', 'sigmoid', 'sin', 'cos', 'tan', 'atan', 'asin', 'acos', 'sinh', 'cosh'}
   for _, layout in ipairs(LAYOUTS) do
      local r, f = make(info, layout, 7, 13, -1, 1)
      for _, op in ipairs(exact) do
         check(info, op .. ' ' .. layout, torch[op](r), torch[op](f), 0)
      end
      for _, op in ipairs(close) do
         check(info, op .. ' ' .. layout, torch[op](r), torch[op](f), 1)
      end
      local p, pf = make(info, layout, 7, 13, 0.5, 4)   -- positive domain
      check(info, 'sqrt ' .. layout, torch.sqrt(p), torch.sqrt(pf), 0)
      check(info, 'log ' .. layout, torch.log(p), torch.log(pf), 1)
      check(info, 'log1p ' .. layout, torch.log1p(p), torch.log1p(pf), 1)
      check(info, 'rsqrt ' .. layout, torch.rsqrt(p), torch.rsqrt(pf), 1)
      check(info, 'pow(x, 2) ' .. layout, torch.pow(r, 2), torch.pow(f, 2), 1)
      check(info, 'pow(2, x) ' .. layout, torch.pow(2, r), torch.pow(2, f), 1)
      -- in place on the layout
      local c, cf = r:clone(), f:clone()
      if layout ~= 'contiguous' then c = make(info, layout, 7, 13, -1, 1); c:copy(r) end
      check(info, 'in-place exp ' .. layout, c:exp(), cf:exp(), 1)
   end
end)

-- ---------------------------------------------------------------------------
-- element-wise binary, scalar and broadcasting
-- ---------------------------------------------------------------------------
tests.binary = forTypes(function(info)
   for _, la in ipairs(LAYOUTS) do
      for _, lb in ipairs(LAYOUTS) do
         local a, af = make(info, la, 6, 9, -2, 2)
         local b, bf = make(info, lb, 6, 9, -2, 2)
         local z, zf = make(info, lb, 6, 9, 0.5, 2)
         local tag = la .. '/' .. lb
         check(info, 'add ' .. tag, torch.add(a, b), torch.add(af, bf), 0)
         check(info, 'a - b ' .. tag, a - b, af - bf, 0)
         check(info, 'cmul ' .. tag, torch.cmul(a, b), torch.cmul(af, bf), 0)
         check(info, 'cdiv ' .. tag, torch.cdiv(a, z), torch.cdiv(af, zf), 0)
         check(info, 'cmax ' .. tag, torch.cmax(a, b), torch.cmax(af, bf), 0)
         check(info, 'cmin ' .. tag, torch.cmin(a, b), torch.cmin(af, bf), 0)
         check(info, 'cfmod ' .. tag, torch.cfmod(a, z), torch.cfmod(af, zf), 0)
         check(info, 'cremainder ' .. tag, torch.cremainder(a, z), torch.cremainder(af, zf), 1)
         check(info, 'cfloordiv ' .. tag, torch.cfloordiv(a, z), torch.cfloordiv(af, zf), 0)
         check(info, 'atan2 ' .. tag, torch.atan2(a, b), torch.atan2(af, bf), 1)
         check(info, 'cpow ' .. tag, torch.cpow(z, b), torch.cpow(zf, bf), 1)
         check(info, 'add(a, 0.5, b) ' .. tag, torch.add(a, 0.5, b), torch.add(af, 0.5, bf), 1)
         check(info, 'lerp ' .. tag, torch.lerp(a, b, 0.25), torch.lerp(af, bf, 0.25), 1)
         check(info, 'addcmul ' .. tag, torch.addcmul(a, 0.5, b, z), torch.addcmul(af, 0.5, bf, zf), 1)
         check(info, 'addcdiv ' .. tag, torch.addcdiv(a, 0.5, b, z), torch.addcdiv(af, 0.5, bf, zf), 1)
      end
      local a, af = make(info, la, 6, 9, -2, 2)
      check(info, 'add scalar ' .. la, torch.add(a, 0.5), torch.add(af, 0.5), 0)
      check(info, 'mul scalar ' .. la, torch.mul(a, 0.25), torch.mul(af, 0.25), 0)
      check(info, 'div scalar ' .. la, torch.div(a, 3), torch.div(af, 3), 0)
      check(info, 'clamp ' .. la, torch.clamp(a, -0.5, 0.5), torch.clamp(af, -0.5, 0.5), 0)
      check(info, 'cmax scalar ' .. la, torch.cmax(a, 0), torch.cmax(af, 0), 0)
      check(info, 'fmod scalar ' .. la, torch.fmod(a, 0.75), torch.fmod(af, 0.75), 0)
      check(info, 'remainder scalar ' .. la, torch.remainder(a, 0.75), torch.remainder(af, 0.75), 1)
      check(info, 'floordiv scalar ' .. la, torch.floordiv(a, 0.75), torch.floordiv(af, 0.75), 0)
      check(info, 'lshift scalar ' .. la, torch.lshift(a, 3), torch.lshift(af, 3), 0)
      check(info, 'rshift scalar ' .. la, torch.rshift(a, 3), torch.rshift(af, 3), 0)
   end
   -- broadcasting follows the Phase 2c rule
   local col, colf = make(info, 'contiguous', 5, 1, -2, 2)
   local row, rowf = make(info, 'contiguous', 1, 7, -2, 2)
   check(info, 'broadcast add', col + row, colf + rowf, 0)
   check(info, 'broadcast cmul', torch.cmul(col, row), torch.cmul(colf, rowf), 0)
   local ok = pcall(function() return torch.zeros(4, 4):type(info.tensor) + torch.zeros(2, 8):type(info.tensor) end)
   tester:assert(not ok, info.name .. ': same numel, incompatible shapes must raise an error')
end)

-- ---------------------------------------------------------------------------
-- comparisons, selection, sorting
-- ---------------------------------------------------------------------------
tests.compare_select = forTypes(function(info)
   for _, layout in ipairs(LAYOUTS) do
      local a, af = make(info, layout, 6, 9, -2, 2)
      local b, bf = make(info, layout, 6, 9, -2, 2)
      for _, op in ipairs({'lt', 'le', 'gt', 'ge', 'eq', 'ne'}) do
         tester:assert(torch[op](a, b):equal(torch[op](af, bf)), info.name .. ' ' .. op .. ' tensor ' .. layout)
         tester:assert(torch[op](a, 0):equal(torch[op](af, 0)), info.name .. ' ' .. op .. ' value ' .. layout)
      end
      tester:assert(a:eq(a):sum() == a:nElement(), info.name .. ' eq self ' .. layout)
      for _, dim in ipairs({1, 2}) do
         for _, op in ipairs({'max', 'min', 'median'}) do
            local v, i = torch[op](a, dim)
            local vf, i_f = torch[op](af, dim)
            check(info, op .. dim .. ' values ' .. layout, v, vf, 0)
            tester:assert(i:equal(i_f), info.name .. ' ' .. op .. dim .. ' indices ' .. layout)
         end
         local sv, si = torch.sort(a, dim)
         local svf, sif = torch.sort(af, dim)
         check(info, 'sort' .. dim .. ' ' .. layout, sv, svf, 0)
         tester:assert(si:equal(sif), info.name .. ' sort' .. dim .. ' indices ' .. layout)
         local kv, ki = torch.kthvalue(a, 2, dim)
         local kvf, kif = torch.kthvalue(af, 2, dim)
         check(info, 'kthvalue' .. dim .. ' ' .. layout, kv, kvf, 0)
         tester:assert(ki:equal(kif), info.name .. ' kthvalue' .. dim .. ' indices ' .. layout)
      end
      tester:asserteq(a:max(), af:max(), info.name .. ' maxall ' .. layout)
      tester:asserteq(a:min(), af:min(), info.name .. ' minall ' .. layout)
      local mask = a:gt(0)
      check(info, 'maskedSelect ' .. layout, a[mask], af[mask], 0)
      check(info, 'index ' .. layout, a:index(2, torch.LongTensor({3, 1, 3})), af:index(2, torch.LongTensor({3, 1, 3})), 0)
   end
end)

-- ---------------------------------------------------------------------------
-- reductions
-- ---------------------------------------------------------------------------
tests.reductions = forTypes(function(info)
   for _, layout in ipairs(LAYOUTS) do
      local a, af = make(info, layout, 37, 23, -1, 1)
      for _, dim in ipairs({1, 2}) do
         for _, op in ipairs({'sum', 'mean', 'prod', 'cumsum', 'cumprod'}) do
            check(info, op .. dim .. ' ' .. layout, torch[op](a, dim), torch[op](af, dim), 0)
         end
         check(info, 'std' .. dim .. ' ' .. layout, torch.std(a, dim), torch.std(af, dim), 0)
         check(info, 'var' .. dim .. ' ' .. layout, torch.var(a, dim), torch.var(af, dim), 0)
         check(info, 'norm' .. dim .. ' ' .. layout, torch.norm(a, 2, dim), torch.norm(af, 2, dim), 0)
      end
      -- whole-tensor reductions accumulate in float (FloatTensor: in double)
      local scale = torch.abs(af):sum()
      tester:assertlt(math.abs(a:sum() - af:sum()), 1e-6 * scale + 1e-7, info.name .. ' sumall ' .. layout)
      tester:assertlt(math.abs(a:mean() - af:mean()), 1e-6 * scale / af:nElement() + 1e-7, info.name .. ' meanall ' .. layout)
      tester:assertlt(math.abs(a:norm() - af:norm()), 1e-5 * af:norm(), info.name .. ' normall ' .. layout)
      tester:assertlt(math.abs(a:std() - af:std()), 1e-5 * af:std(), info.name .. ' stdall ' .. layout)
   end
   -- mean of equal values is exact (upstream rounded the sum to 16 bits first)
   local v = torch.FloatTensor(8, 5):fill(0.1):type(info.tensor)
   check(info, 'mean of equal values', v:mean(2), v:float():mean(2), 0)
   tester:asserteq(v:mean(2)[3][1], v[1][1], info.name .. ' mean of equal values is the value')
end)

-- ---------------------------------------------------------------------------
-- linear algebra: accumulation in float
-- ---------------------------------------------------------------------------
tests.linear_algebra = forTypes(function(info)
   for _, shape in ipairs({{3, 4, 5}, {33, 17, 65}, {64, 64, 64}}) do
      local m, k, n = shape[1], shape[2], shape[3]
      local a, af = make(info, 'contiguous', m, k, -1, 1)
      local b, bf = make(info, 'contiguous', k, n, -1, 1)
      local tag = m .. 'x' .. k .. 'x' .. n
      check(info, 'mm ' .. tag, torch.mm(a, b), torch.mm(af, bf), 0)
      local c, cf = make(info, 'contiguous', m, n, -1, 1)
      check(info, 'addmm ' .. tag, torch.addmm(0.5, c, 2, a, b), torch.addmm(0.5, cf, 2, af, bf), 1)
      for _, la in ipairs({'transposed', 'strided'}) do
         local at, atf = make(info, la, m, k, -1, 1)
         check(info, 'mm ' .. la .. ' ' .. tag, torch.mm(at, b), torch.mm(atf, bf), 1)
      end
      local x, xf = make(info, 'contiguous', k, 1, -1, 1)
      x, xf = x:select(2, 1), xf:select(2, 1)
      check(info, 'mv ' .. tag, torch.mv(a, x), torch.mv(af, xf), 1)
      check(info, 'mv transposed ' .. tag, torch.mv(a:t():contiguous():t(), x), torch.mv(af, xf), 1)
      local y, yf = make(info, 'contiguous', m, 1, -1, 1)
      y, yf = y:select(2, 1), yf:select(2, 1)
      local s = a:select(2, 1)
      tester:assertlt(math.abs(torch.dot(s, y) - torch.dot(af:select(2, 1), yf)), 1e-6 * m,
                      info.name .. ' dot ' .. tag)
      check(info, 'addr ' .. tag, torch.addr(c, 0.5, y, b:select(1, 1)), torch.addr(cf, 0.5, yf, bf:select(1, 1)), 1)
   end
   -- the accumulation is really in float: 4096 * 1 overflows nothing and is exact
   local ones = torch.FloatTensor(1, 4096):fill(1):type(info.tensor)
   tester:asserteq(torch.mm(ones, ones:t())[1][1], 4096, info.name .. ' mm accumulates in float')
   tester:asserteq(torch.dot(ones:view(-1), ones:view(-1)), 4096, info.name .. ' dot accumulates in float')
   local bm, bmf = make(info, 'contiguous', 3 * 4, 5, -1, 1)
   local cm, cmf = make(info, 'contiguous', 3 * 5, 6, -1, 1)
   check(info, 'bmm', torch.bmm(bm:view(3, 4, 5), cm:view(3, 5, 6)), torch.bmm(bmf:view(3, 4, 5), cmf:view(3, 5, 6)), 0)
end)

-- ---------------------------------------------------------------------------
-- conversion, special values, construction
-- ---------------------------------------------------------------------------
tests.conversion = forTypes(function(info)
   local T = torch[info.name .. 'Tensor']
   -- exactly representable values survive every conversion
   local vals = {0, NEGZERO, 1, -1, 0.5, 1024, -2048, info.max, -info.max, 2 ^ info.emin, 2 ^ (info.emin - info.mant)}
   local t = T(vals)
   for i, v in ipairs(vals) do tester:asserteq(t[i], v, info.name .. ' value ' .. v) end
   for _, other in ipairs({'Double', 'Float', 'Long', 'Int'}) do
      local o = t:type('torch.' .. other .. 'Tensor')
      if other == 'Long' or other == 'Int' then
         local exactInts = T({0, 1, -1, 1024, -2048})
         tester:assert(exactInts:type('torch.' .. other .. 'Tensor'):type(info.tensor):equal(exactInts),
                       info.name .. ' <-> ' .. other .. ' integers')
      else
         tester:assert(o:type(info.tensor):equal(t), info.name .. ' <-> ' .. other .. ' round trip')
      end
   end
   -- overflow, NaN, signed zero
   tester:asserteq(T({info.max * 2})[1], math.huge, info.name .. ' overflow to +inf')
   tester:asserteq(T({-info.max * 2})[1], -math.huge, info.name .. ' overflow to -inf')
   local n = T({0 / 0})[1]; tester:assert(n ~= n, info.name .. ' NaN stays NaN')
   tester:asserteq(1 / T({NEGZERO})[1], -math.huge, info.name .. ' -0 keeps its sign')
   local mx = T({1, 0 / 0, 3}):max(); tester:assert(mx ~= mx, info.name .. ' max propagates NaN')
   -- the bulk float -> 16-bit copy (contiguous) and the per-element casts agree on special values
   local specials = {0 / 0, -(0 / 0), math.huge, -math.huge, NEGZERO, 2 ^ -130, 2 ^ -149, 3.4e38, -1 / 3,
                     2 ^ (info.emin - info.mant - 1), 1 + 2 ^ -(info.mant + 1)}
   local sf = torch.FloatTensor(specials)
   local bulk = sf:type(info.tensor)                                     -- contiguous: bulk routine
   local strided = torch.FloatTensor(#specials, 2):select(2, 1):copy(sf)
   local scalarCopy = T(#specials):copy(strided)                         -- non-contiguous: element loop
   for i = 1, #specials do
      local s = T(1); s[1] = sf[i]                                       -- Lua number -> 16-bit cast
      for _, pair in ipairs({{bulk[i], 'bulk'}, {scalarCopy[i], 'strided copy'}}) do
         local a, b = pair[1], s[1]
         local same = (a ~= a and b ~= b) or (a == b and (a ~= 0 or 1 / a == 1 / b))
         tester:assert(same, string.format('%s special %d (%s): %s gives %s, cast gives %s',
                                           info.name, i, tostring(sf[i]), pair[2], tostring(a), tostring(b)))
      end
   end
   -- every route from a Lua or double value rounds once (no double rounding through float)
   local x = 1 + 2 ^ -(info.mant + 1) + 2 ^ -40      -- just above a tie in the reduced type
   local want = 1 + 2 ^ -info.mant
   tester:asserteq(T({x})[1], want, info.name .. ' construct rounds once')
   tester:asserteq(T(1):fill(x)[1], want, info.name .. ' fill rounds once')
   local s = T(1); s[1] = x; tester:asserteq(s[1], want, info.name .. ' assignment rounds once')
   tester:asserteq(torch.DoubleTensor({x}):type(info.tensor)[1], want, info.name .. ' copy from double rounds once')
   -- constructors and in-place basics
   tester:asserteq(T(3, 4):fill(0.25):sum(), 3, info.name .. ' fill + sum')
   tester:asserteq(T(2, 3):zero():sum(), 0, info.name .. ' zero')
   tester:assert(torch.zeros(2, 3):type(info.tensor):equal(T(2, 3):zero()), info.name .. ' zeros')
   tester:asserteq(torch.range(1, 5):type(info.tensor):sum(), 15, info.name .. ' range')
   tester:asserteq(T():linspace(0, 1, 5)[2], 0.25, info.name .. ' linspace')
   tester:asserteq(T(3, 3):zero():add(1):trace(), 3, info.name .. ' trace')
end)

tests.random = forTypes(function(info)
   local T = torch[info.name .. 'Tensor']
   local u = T(1000):uniform(-2, 3)
   tester:assert(u:min() >= -2 and u:max() <= 3, info.name .. ' uniform range')
   local r = T(1000):random(1, 10)
   tester:assert(r:min() >= 1 and r:max() <= 10 and r:eq(r:clone():floor()):sum() == 1000, info.name .. ' random(1, 10)')
   local nrm = T(1000):normal(0, 1)
   tester:assert(nrm:eq(nrm):sum() == 1000 and math.abs(nrm:mean()) < 0.2, info.name .. ' normal')
   local bern = T(1000):bernoulli(0.5)
   tester:assert(bern:eq(0):sum() + bern:eq(1):sum() == 1000, info.name .. ' bernoulli')
   tester:assert(T(1000):exponential(1):min() >= 0, info.name .. ' exponential')
   torch.manualSeed(7); local a = T(50):uniform()
   torch.manualSeed(7); local b = T(50):uniform()
   tester:assert(a:equal(b), info.name .. ' manualSeed reproducible')
end)

tests.serialization = forTypes(function(info)
   local T = torch[info.name .. 'Tensor']
   local t = torch.FloatTensor(5, 7):uniform(-100, 100):type(info.tensor)
   t[1][1] = 0 / 0; t[1][2] = math.huge; t[1][3] = NEGZERO; t[1][4] = 2 ^ (info.emin - info.mant)
   for _, mode in ipairs({'binary', 'ascii'}) do
      local f = torch.MemoryFile()
      if mode == 'binary' then f:binary() end
      f:writeObject(t)
      f:seek(1)
      local u = f:readObject()
      f:close()
      tester:asserteq(torch.type(u), info.tensor, info.name .. ' ' .. mode .. ' type')
      local same = true
      for i = 1, 5 do for j = 1, 7 do
         local a, b = t[i][j], u[i][j]
         if not ((a ~= a and b ~= b) or (a == b and 1 / a == 1 / b) or (a == b and a ~= 0)) then same = false end
      end end
      tester:assert(same, info.name .. ' ' .. mode .. ' round trip')
   end
   tester:assert(T(3):fill(1):storage():size() == 3, info.name .. ' storage')
end)

tests.unsupported = forTypes(function(info)
   local T = torch[info.name .. 'Tensor']
   local a = T(3):fill(1)
   tester:assert(not pcall(function() return torch.cbitand(a, a) end), info.name .. ' cbitand raises an error')
   tester:assert(not pcall(function() return torch.arshift(a, 1) end), info.name .. ' arshift raises an error')
end)

tester:add(tests)
tester:run()

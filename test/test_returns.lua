-- In-place Tensor methods return self (ctest: lua.returns).
--
-- Usage: luajit test_returns.lua <staged lua dir> <module dir> <module suffix>
--
-- Hand-written bindings that "return 1" without pushing a value return
-- whatever is on top of the Lua stack (their last argument). indexAdd and
-- indexCopy returned the source tensor and indexFill / maskedFill the fill
-- value, so chaining (torch.zeros(n,m):indexAdd(...)) silently produced the
-- wrong tensor. Reported by the LatticeAI project.
local luaDir, modDir, modSuffix = arg[1], arg[2], arg[3]
assert(luaDir and modDir and modSuffix, "usage: test_returns.lua <lua dir> <module dir> <suffix>")
package.path = luaDir .. "/?.lua;" .. luaDir .. "/?/init.lua;" .. package.path
package.cpath = modDir .. "/?" .. modSuffix .. ";" .. package.cpath
require 'torch'

local tester = torch.Tester()
local tests = torch.TestSuite()

-- {method name, function(x) calling it on x with suitable arguments}
local inplace = {
   {'indexAdd',   function(x) return x:indexAdd(2, torch.LongTensor{1, 1, 4}, torch.ones(2, 3)) end},
   {'indexCopy',  function(x) return x:indexCopy(2, torch.LongTensor{2}, torch.ones(2, 1)) end},
   {'indexFill',  function(x) return x:indexFill(2, torch.LongTensor{3}, 7) end},
   {'maskedFill', function(x) return x:maskedFill(torch.ByteTensor(2, 4):fill(1), 3) end},
   {'maskedCopy', function(x) return x:maskedCopy(torch.ByteTensor(2, 4):fill(1), torch.ones(8)) end},
   {'fill',       function(x) return x:fill(1) end},
   {'zero',       function(x) return x:zero() end},
   {'copy',       function(x) return x:copy(torch.ones(2, 4)) end},
   {'add',        function(x) return x:add(1) end},
   {'mul',        function(x) return x:mul(2) end},
   {'cmul',       function(x) return x:cmul(torch.ones(2, 4)) end},
   {'clamp',      function(x) return x:clamp(0, 1) end},
   {'scatter',    function(x) return x:scatter(2, torch.LongTensor{{1}, {2}}, 5) end},
   {'resize',     function(x) return x:resize(2, 4) end},
}

for _, case in ipairs(inplace) do
   tests['returns_self_' .. case[1]] = function()
      local x = torch.zeros(2, 4)
      local r = case[2](x)
      tester:assert(torch.isTensor(r), case[1] .. ' must return a tensor, got ' .. type(r))
      tester:assert(r == x, case[1] .. ' must return self')
   end
end

-- the reported case: chaining keeps the accumulated result
function tests.indexAdd_chaining()
   local r = torch.zeros(2, 4):indexAdd(2, torch.LongTensor{1, 1, 4}, torch.Tensor{{1, 2, 3}, {4, 5, 6}})
   tester:eq(r:size(2), 4)
   tester:eq(r[1][1], 3); tester:eq(r[1][4], 3)
   tester:eq(r[2][1], 9); tester:eq(r[2][4], 6)
end

tester:add(tests)
tester:run()

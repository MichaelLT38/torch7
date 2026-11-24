-- CTest driver for the Lua test suite (test/test.lua).
--
-- Usage: luajit run_lua_tests.lua <staged lua dir> <module dir> <module suffix> <has_lapack: 0|1>
-- (CTest passes these; see the top-level CMakeLists.txt.)
--
-- When TH was built without LAPACK, the tests that need LAPACK cannot pass, so
-- they are skipped and listed loudly instead of reported as errors.
local luaDir, modDir, modSuffix, lapackFlag = arg[1], arg[2], arg[3], arg[4]
assert(luaDir and modDir and modSuffix and lapackFlag,
       "usage: run_lua_tests.lua <lua dir> <module dir> <module suffix> <0|1>")
package.path = luaDir .. "/?.lua;" .. luaDir .. "/?/init.lua;" .. package.path
package.cpath = modDir .. "/?" .. modSuffix .. ";" .. package.cpath
local hasLapack = lapackFlag == "1"

local needsLapack = {
   "eig", "eig_noncontig", "eig_reuse",
   "gels_overdetermined", "gels_reuse", "gels_underdetermined", "gels_uniquely_determined",
   "gesv", "gesv_reuse", "inverse", "potri", "potrs", "pstrf",
   "svd", "svd_noncontig", "svd_reuse", "symeig_noncontig", "test_symeig",
   "testCholesky", "trtrs", "trtrs_reuse",
}

require 'torch'

if not hasLapack then
   -- Methods live on the class metatable, not on the torch.Tester constructor.
   -- Let add() register everything, then use the Tester's own disable().
   local Tester = torch.getmetatable('torch.Tester')
   local add = Tester.add
   Tester.add = function(self, f, name)
      add(self, f, name)
      for _, t in ipairs(needsLapack) do
         if self.tests[t] and not self.disabledTests[t] then self:disable(t) end
      end
      return self
   end
   io.stderr:write(string.format(
      "NOTE: TH built without LAPACK; skipping %d LAPACK tests: %s\n",
      #needsLapack, table.concat(needsLapack, " ")))
end

-- Tester:run raises an error when any test fails, which makes luajit exit
-- non-zero, which CTest reports as a failure.
torch.test()

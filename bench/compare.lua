-- Compare two th_bench CSV runs.
--
--   luajit bench/compare.lua <baseline.csv> <new.csv>
--
-- Prints each op/threads pair with both timings and the speedup (baseline /
-- new; > 1 means the new run is faster). Lines starting with '#' are ignored.
local function load(path)
   local f = assert(io.open(path, "r"), "cannot open " .. path)
   local rows, order = {}, {}
   for line in f:lines() do
      local op, threads, ms = line:match("^([%w_]+),(%d+),([%d%.eE+-]+)%s*$")
      if op then
         local key = op .. "@" .. threads
         if not rows[key] then order[#order + 1] = key end
         rows[key] = tonumber(ms)
      end
   end
   f:close()
   return rows, order
end

local basePath, newPath = arg[1], arg[2]
if not (basePath and newPath) then
   io.stderr:write("usage: compare.lua <baseline.csv> <new.csv>\n")
   os.exit(2)
end

local base, order = load(basePath)
local new = load(newPath)

print(string.format("%-26s %8s %12s %12s %9s", "op", "threads", "base ms", "new ms", "speedup"))
for _, key in ipairs(order) do
   local op, threads = key:match("^(.*)@(%d+)$")
   local b, n = base[key], new[key]
   if n then
      print(string.format("%-26s %8s %12.3f %12.3f %8.2fx", op, threads, b, n, b / n))
   else
      print(string.format("%-26s %8s %12.3f %12s %9s", op, threads, b, "-", "missing"))
   end
end

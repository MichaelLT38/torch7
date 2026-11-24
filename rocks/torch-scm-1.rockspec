package = "torch"
version = "scm-1"

source = {
   url = "git+https://github.com/MichaelLT38/torch7.git",
}

description = {
   summary = "Torch7",
   detailed = [[
   ]],
   homepage = "https://github.com/MichaelLT38/torch7",
   license = "BSD"
}

-- paths and cwrap are bundled (lib/paths, lib/cwrap) and installed by this rock.
dependencies = {
   "lua >= 5.1",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DLUA=$(LUA) -DLUALIB=$(LUALIB) -DLUA_BINDIR="$(LUA_BINDIR)" -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" -DLUADIR="$(LUADIR)" -DLIBDIR="$(LIBDIR)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN)
]],
	 platforms = {
      windows = {
           build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DLUA=$(LUA) -DLUALIB=$(LUALIB) -DLUA_BINDIR="$(LUA_BINDIR)" -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" -DLUADIR="$(LUADIR)" -DLIBDIR="$(LIBDIR)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]]
      }
   },
   install_command = "cd build && $(MAKE) install"
}


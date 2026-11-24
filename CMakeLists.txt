cmake_minimum_required(VERSION 3.22)
project(Torch LANGUAGES C)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TorchCompiler)  # GCC >= 13 only, C11, OpenMP
include(TorchLua)       # LUA, LUA_INCDIR, LUADIR, LIBDIR, LUALIB, ...

# Includes
INCLUDE(TorchPaths)
INCLUDE(TorchPathsInit)
INCLUDE(TorchPackage)
INCLUDE(TorchWrap)
INCLUDE(TorchExports)

# Put every library and Lua module in the build root so the build tree can be
# used directly (tests `require 'torch'` from it without installing).
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Torch libraries
ADD_SUBDIRECTORY(lib)

CONFIGURE_FILE(paths.lua.in "${CMAKE_CURRENT_BINARY_DIR}/paths.lua")

INCLUDE_DIRECTORIES(BEFORE "${LUA_INCDIR}")
INCLUDE_DIRECTORIES(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/lib/TH")
INCLUDE_DIRECTORIES(BEFORE "${CMAKE_CURRENT_BINARY_DIR}/lib/TH")
INCLUDE_DIRECTORIES(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/lib/luaT")
LINK_DIRECTORIES("${LUA_LIBDIR}")

SET(src DiskFile.c File.c MemoryFile.c PipeFile.c Storage.c Tensor.c Timer.c utils.c init.c TensorOperator.c TensorMath.c random.c Generator.c)
SET(luasrc init.lua File.lua Tensor.lua CmdLine.lua FFInterface.lua Tester.lua TestSuite.lua ${CMAKE_CURRENT_BINARY_DIR}/paths.lua test/test.lua)

# Necessary do generate wrapper
ADD_TORCH_WRAP(tensormathwrap TensorMath.lua)
ADD_TORCH_WRAP(randomwrap random.lua)

ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")

TARGET_LINK_LIBRARIES(torch luaT TH OpenMP::OpenMP_C)

IF(LUALIB)
  TARGET_LINK_LIBRARIES(torch ${LUALIB})
ENDIF()

INSTALL(FILES "README.md" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")
INSTALL(DIRECTORY "doc" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")

######################################################################
# In-tree staging of the Lua sources, mirroring the installed layout
# (<lua dir>/torch/*.lua, <lua dir>/paths/init.lua, <lua dir>/cwrap/*.lua).
######################################################################
SET(TORCH_STAGE_LUA "${CMAKE_BINARY_DIR}/lua")
SET(_cwrap_lua lib/cwrap/init.lua lib/cwrap/cinterface.lua lib/cwrap/types.lua)
ADD_CUSTOM_TARGET(torch_stage ALL
  COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${TORCH_STAGE_LUA}/torch" "${TORCH_STAGE_LUA}/paths" "${TORCH_STAGE_LUA}/cwrap"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${luasrc} "${TORCH_STAGE_LUA}/torch/"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different lib/paths/init.lua "${TORCH_STAGE_LUA}/paths/"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${_cwrap_lua} "${TORCH_STAGE_LUA}/cwrap/"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Staging Lua sources in ${TORCH_STAGE_LUA}"
  VERBATIM)

######################################################################
# Tests (ctest)
######################################################################
OPTION(TORCH_BUILD_BENCH "Build the th_bench micro-benchmark (bench/)" ON)
IF(TORCH_BUILD_BENCH)
  ADD_SUBDIRECTORY(bench)
ENDIF()

OPTION(TORCH_BUILD_TESTS "Build and register the test suites" ON)
IF(TORCH_BUILD_TESTS)
  ENABLE_TESTING()

  ADD_SUBDIRECTORY(test/c)   # TH C-level tests: th.<suite>

  IF(TH_HAS_LAPACK)
    SET(_has_lapack 1)
  ELSE()
    SET(_has_lapack 0)
    MESSAGE(STATUS "TH has no LAPACK: the Lua suite will skip LAPACK tests")
  ENDIF()

  ADD_TEST(NAME lua.torch
    COMMAND "${LUA}" "${CMAKE_CURRENT_SOURCE_DIR}/test/run_lua_tests.lua"
            "${TORCH_STAGE_LUA}" "${CMAKE_BINARY_DIR}" "${CMAKE_SHARED_MODULE_SUFFIX}" ${_has_lapack}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
  IF(WIN32)
    # Windows resolves libTH.dll/libluaT.dll through PATH.
    SET_TESTS_PROPERTIES(lua.torch PROPERTIES
      ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CMAKE_BINARY_DIR}")
  ENDIF()
ENDIF()

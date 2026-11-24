# In-tree version of ADD_TORCH_WRAP: generates <name>.c from a cwrap script
# using the bundled lib/cwrap, so building torch needs no installed packages.
# (cmake/TorchWrap.cmake.in is the installed variant for downstream packages.)
SET(TORCH_CWRAP_DEPS
  "${CMAKE_SOURCE_DIR}/cmake/runwrap.lua"
  "${CMAKE_SOURCE_DIR}/torchcwrap.lua"
  "${CMAKE_SOURCE_DIR}/lib/cwrap/init.lua"
  "${CMAKE_SOURCE_DIR}/lib/cwrap/cinterface.lua"
  "${CMAKE_SOURCE_DIR}/lib/cwrap/types.lua")

MACRO(ADD_TORCH_WRAP target luafile)
  INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
  GET_FILENAME_COMPONENT(_file_ "${luafile}" NAME_WE)
  SET(cfile "${_file_}.c")
  ADD_CUSTOM_COMMAND(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${cfile}"
    COMMAND "${LUA}" "${CMAKE_SOURCE_DIR}/cmake/runwrap.lua" "${CMAKE_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${luafile}" "${CMAKE_CURRENT_BINARY_DIR}/${cfile}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS "${luafile}" ${TORCH_CWRAP_DEPS}
    VERBATIM)
ENDMACRO(ADD_TORCH_WRAP)

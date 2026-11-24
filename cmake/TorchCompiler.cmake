# Compiler policy shared by every Torch target (also used by lib/TH when it is
# built standalone).
#
# Torch supports exactly one compiler: GCC >= 13. Restricting ourselves to GCC
# lets the code rely on GNU C extensions (vector types, target_clones,
# _Float16/__bf16, OpenMP via libgomp) instead of carrying per-compiler paths.
# On Windows, use the MSYS2 UCRT64 toolchain.
include_guard(GLOBAL)

set(TORCH_MIN_GCC_VERSION 13)

if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR
    "Torch requires GCC >= ${TORCH_MIN_GCC_VERSION}, but the C compiler is "
    "'${CMAKE_C_COMPILER_ID}' (${CMAKE_C_COMPILER}). "
    "Configure with -DCMAKE_C_COMPILER=gcc (on Windows, use MSYS2 UCRT64).")
endif()
if(CMAKE_C_COMPILER_VERSION VERSION_LESS TORCH_MIN_GCC_VERSION)
  message(FATAL_ERROR
    "Torch requires GCC >= ${TORCH_MIN_GCC_VERSION}, found ${CMAKE_C_COMPILER_VERSION}.")
endif()

# gnu11 rather than c11: TH uses GNU extensions (e.g. posix_memalign).
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

add_compile_options(-Werror=implicit-function-declaration -Werror=format)

# OpenMP ships with GCC (libgomp), so it is a hard requirement.
find_package(OpenMP REQUIRED COMPONENTS C)

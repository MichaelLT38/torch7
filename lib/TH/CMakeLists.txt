cmake_minimum_required(VERSION 3.16)

# TH can be built standalone (cmake -S lib/TH) or as part of Torch.
IF(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(TH LANGUAGES C)
ENDIF()

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/TorchCompiler.cmake")  # GCC >= 13, C11, OpenMP
SET(CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/ ${CMAKE_LIBRARY_PATH})

# Can be compiled standalone
IF(NOT TH_INSTALL_BIN_SUBDIR
    OR NOT TH_INSTALL_LIB_SUBDIR
    OR NOT TH_INSTALL_INCLUDE_SUBDIR
    OR NOT TH_INSTALL_CMAKE_SUBDIR)

  SET(TH_INSTALL_BIN_SUBDIR "bin" CACHE PATH "TH install binary subdirectory")
  SET(TH_INSTALL_LIB_SUBDIR "lib" CACHE PATH "TH install library subdirectory")
  SET(TH_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "TH install include subdirectory")
  SET(TH_INSTALL_CMAKE_SUBDIR "share/cmake/TH" CACHE PATH "TH install cmake subdirectory")
ENDIF()

######################################################################
###### macros section
#####################################################################
IF(NOT COMMAND ADD_TORCH_LIBRARY)
MACRO(ADD_TORCH_LIBRARY package type src)
  ADD_LIBRARY(${package} ${type} ${src})
ENDMACRO()
ENDIF()

#######################################################################
##### flags section
######################################################################

INCLUDE(CheckFunctionExists)

# ARM specific flags
FIND_PACKAGE(ARM)
IF (ASIMD_FOUND)
  MESSAGE(STATUS "asimd/Neon found with compiler flag : -D__NEON__")
  SET(CMAKE_C_FLAGS "-D__NEON__ ${CMAKE_C_FLAGS}")
ELSEIF (NEON_FOUND)
  MESSAGE(STATUS "Neon found with compiler flag : -mfpu=neon -D__NEON__")
  SET(CMAKE_C_FLAGS "-mfpu=neon -D__NEON__ ${CMAKE_C_FLAGS}")
ENDIF (ASIMD_FOUND)
IF (CORTEXA8_FOUND)
  MESSAGE(STATUS "Cortex-A8 Found with compiler flag : -mcpu=cortex-a8")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a8 -fprefetch-loop-arrays ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA8_FOUND)
IF (CORTEXA9_FOUND)
  MESSAGE(STATUS "Cortex-A9 Found with compiler flag : -mcpu=cortex-a9")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a9 ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA9_FOUND)

INCLUDE (CheckCSourceCompiles)

FIND_PACKAGE(SSE) # checks SSE, AVX and AVX2
IF(C_SSE2_FOUND)
  MESSAGE(STATUS "SSE2 Found")
  SET(CMAKE_C_FLAGS "${C_SSE2_FLAGS} -DUSE_SSE2 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE2_FOUND)
IF(C_SSE3_FOUND)
  MESSAGE(STATUS "SSE3 Found")
  SET(CMAKE_C_FLAGS "${C_SSE3_FLAGS} -DUSE_SSE3 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE3_FOUND)
# we don't set -mavx and -mavx2 flags globally, but only for specific files
# however, we want to enable the AVX codepaths, so we still need to
# add USE_AVX and USE_AVX2 macro defines
IF(C_AVX_FOUND)
  MESSAGE(STATUS "AVX Found")
  SET(CMAKE_C_FLAGS "-DUSE_AVX ${CMAKE_C_FLAGS}")
ENDIF(C_AVX_FOUND)
IF(C_AVX2_FOUND)
  MESSAGE(STATUS "AVX2 Found")
  SET(CMAKE_C_FLAGS "-DUSE_AVX2 ${CMAKE_C_FLAGS}")
ENDIF(C_AVX2_FOUND)

#######################################################################
##### sources section
######################################################################

# IF ANY SIMD FOUND
IF(C_AVX2_FOUND OR C_AVX_FOUND OR C_SSE4_2_FOUND OR C_SSE4_1_FOUND)
  SET(simd generic/simd/convolve.c)
ENDIF(C_AVX2_FOUND OR C_AVX_FOUND OR C_SSE4_2_FOUND OR C_SSE4_1_FOUND)

# IF SSE4 FOUND
IF(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE4_1_FLAGS} -DUSE_SSE4_1 ${C_SSE4_2_FLAGS} -DUSE_SSE4_2 ${CMAKE_C_FLAGS}")
  SET_SOURCE_FILES_PROPERTIES(generic/simd/convolve5x5_sse.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math")
  SET(simd ${simd} generic/simd/convolve5x5_sse.c)
ENDIF(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)

# IF AVX FOUND
IF(C_AVX_FOUND)
  SET_SOURCE_FILES_PROPERTIES(generic/simd/convolve5x5_avx.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math ${C_AVX_FLAGS}")
  SET_SOURCE_FILES_PROPERTIES(vector/AVX.c PROPERTIES COMPILE_FLAGS "-O3 ${C_AVX_FLAGS}")
  SET(simd ${simd} vector/AVX.c generic/simd/convolve5x5_avx.c)
ENDIF(C_AVX_FOUND)

IF(C_AVX2_FOUND)
  SET_SOURCE_FILES_PROPERTIES(vector/AVX2.c PROPERTIES COMPILE_FLAGS "-O3 ${C_AVX2_FLAGS}")
  SET(simd ${simd} vector/AVX2.c)
ENDIF(C_AVX2_FOUND)

SET(hdr
  THHalf.h THAllocator.h THSize.h THStorage.h THTensor.h THTensorApply.h THBlas.h THMath.h
  THLapack.h THLogAdd.h THRandom.h THVector.h THAtomic.h THIter.h THVectorMath.h )

SET(src
  THGeneral.c THHalf.c THAllocator.c THSize.c THStorage.c THTensor.c THBlas.c THLapack.c
  THLogAdd.c THRandom.c THFile.c THDiskFile.c THMemoryFile.c THAtomic.c THVector.c THIter.c THVectorMath.c)

SET(src ${src} ${hdr} ${simd})

#######################################################################
##### build section
######################################################################

ADD_TORCH_LIBRARY(TH SHARED "${src}")

IF (BUILD_STATIC OR "$ENV{STATIC_TH}" STREQUAL "YES")
  ADD_TORCH_LIBRARY(TH_static STATIC "${src}")
  SET_TARGET_PROPERTIES(TH_static PROPERTIES
    COMPILE_FLAGS "-fPIC")
  SET_TARGET_PROPERTIES(TH_static PROPERTIES
    PREFIX "lib" IMPORT_PREFIX "lib" OUTPUT_NAME "TH")
ENDIF()

IF(NOT TH_SO_VERSION)
  SET(TH_SO_VERSION 0)
ENDIF(NOT TH_SO_VERSION)
MESSAGE(STATUS "TH_SO_VERSION: ${TH_SO_VERSION}")
SET_TARGET_PROPERTIES(TH PROPERTIES
  VERSION   ${TH_SO_VERSION}
  SOVERSION ${TH_SO_VERSION})

TARGET_LINK_LIBRARIES(TH OpenMP::OpenMP_C)

FIND_PACKAGE(BLAS)
IF(BLAS_FOUND)
  SET(USE_BLAS 1)
  IF ($ENV{TH_BINARY_BUILD})
    MESSAGE(STATUS "TH_BINARY_BUILD detected. Enabling special linkage.")
    TARGET_LINK_LIBRARIES(TH "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
  ELSE ($ENV{TH_BINARY_BUILD})
    TARGET_LINK_LIBRARIES(TH ${BLAS_LIBRARIES})
  ENDIF ($ENV{TH_BINARY_BUILD})
  
  IF(BLAS_INFO STREQUAL "mkl")
    ADD_DEFINITIONS(-DTH_BLAS_MKL)
  ENDIF()
ENDIF(BLAS_FOUND)

FIND_PACKAGE(LAPACK)
IF(LAPACK_FOUND)
  SET(USE_LAPACK 1)
  TARGET_LINK_LIBRARIES(TH ${LAPACK_LIBRARIES})
ENDIF(LAPACK_FOUND)
# Visible to the parent project (tests skip LAPACK-only cases without it).
SET(TH_HAS_LAPACK "${LAPACK_FOUND}" CACHE INTERNAL "TH was built with LAPACK")

IF (UNIX AND NOT APPLE)
   INCLUDE(CheckLibraryExists)
   # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
   CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
   IF(NEED_LIBRT)
     TARGET_LINK_LIBRARIES(TH rt)
     SET(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
   ENDIF(NEED_LIBRT)
ENDIF(UNIX AND NOT APPLE)

IF(UNIX)
  SET(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  IF(HAVE_MMAP)
    ADD_DEFINITIONS(-DHAVE_MMAP=1)
  ENDIF(HAVE_MMAP)
  # done for lseek: https://www.gnu.org/software/libc/manual/html_node/File-Position-Primitive.html
  ADD_DEFINITIONS(-D_FILE_OFFSET_BITS=64)
  CHECK_FUNCTION_EXISTS(shm_open HAVE_SHM_OPEN)
  IF(HAVE_SHM_OPEN)
    ADD_DEFINITIONS(-DHAVE_SHM_OPEN=1)
  ENDIF(HAVE_SHM_OPEN)
  CHECK_FUNCTION_EXISTS(shm_unlink HAVE_SHM_UNLINK)
  IF(HAVE_SHM_UNLINK)
    ADD_DEFINITIONS(-DHAVE_SHM_UNLINK=1)
  ENDIF(HAVE_SHM_UNLINK)
  CHECK_FUNCTION_EXISTS(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)
  IF(HAVE_MALLOC_USABLE_SIZE)
    ADD_DEFINITIONS(-DHAVE_MALLOC_USABLE_SIZE=1)
  ENDIF(HAVE_MALLOC_USABLE_SIZE)
ENDIF(UNIX)

TARGET_LINK_LIBRARIES(TH m)

# GCC always provides C11 atomics and __thread.
TARGET_COMPILE_DEFINITIONS(TH PRIVATE USE_C11_ATOMICS=1)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTH_HAVE_THREAD")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE(THGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h")


#######################################################################
##### install section
######################################################################

INSTALL(TARGETS TH
  EXPORT TH-exports
  RUNTIME DESTINATION "${TH_INSTALL_BIN_SUBDIR}"
  LIBRARY DESTINATION "${TH_INSTALL_LIB_SUBDIR}"
  ARCHIVE DESTINATION "${TH_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
  TH.h
  THAllocator.h
  THMath.h
  THBlas.h
  THDiskFile.h
  THFile.h
  THFilePrivate.h
  ${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h
  THGenerateAllTypes.h
  THGenerateDoubleType.h
  THGenerateFloatType.h
  THGenerateHalfType.h
  THGenerateReducedTypes.h
  THGenerateLongType.h
  THGenerateIntType.h
  THGenerateShortType.h
  THGenerateCharType.h
  THGenerateByteType.h
  THGenerateFloatTypes.h
  THGenerateIntTypes.h
  THLapack.h
  THLogAdd.h
  THMemoryFile.h
  THRandom.h
  THSize.h
  THStorage.h
  THTensor.h
  THTensorApply.h
  THTensorDimApply.h
  THTensorMacros.h
  THVector.h
  THAtomic.h
  THIter.h
  THVectorMath.h
  THHalf.h
  DESTINATION "${TH_INSTALL_INCLUDE_SUBDIR}/TH")

INSTALL(FILES
  vector/AVX.h
  vector/AVX2.h
  DESTINATION "${TH_INSTALL_INCLUDE_SUBDIR}/TH/vector")

INSTALL(FILES
  generic/THBlas.c
  generic/THBlas.h
  generic/THLapack.c
  generic/THLapack.h
  generic/THStorage.c
  generic/THStorage.h
  generic/THStorageCopy.c
  generic/THStorageCopy.h
  generic/THTensor.c
  generic/THTensor.h
  generic/THTensorConv.c
  generic/THTensorConv.h
  generic/THTensorCopy.c
  generic/THTensorCopy.h
  generic/THTensorLapack.c
  generic/THTensorLapack.h
  generic/THTensorMath.c
  generic/THTensorMath.h
  generic/THTensorRandom.c
  generic/THTensorRandom.h
  generic/THVectorDispatch.c
  generic/THVector.h
  DESTINATION "${TH_INSTALL_INCLUDE_SUBDIR}/TH/generic")


IF (WIN32 AND NOT CYGWIN)
  SET(BLAS_INSTALL_LIBRARIES "OFF"
    CACHE BOOL "Copy the required BLAS DLLs into the TH install dirs")
ENDIF (WIN32 AND NOT CYGWIN)

MACRO(Install_Required_Library ln)
    get_filename_component(libpath ${ln} PATH)
    get_filename_component(libname ${ln} NAME_WE)
    file(GLOB libdlls "${libpath}/${libname}*.dll")
    install(PROGRAMS ${libdlls}
      DESTINATION "${TH_INSTALL_BIN_SUBDIR}")
ENDMACRO(Install_Required_Library libname)

IF (BLAS_FOUND AND BLAS_INSTALL_LIBRARIES)
  IF (BLAS_goto2_LIBRARY)
    Install_Required_Library(${BLAS_goto2_LIBRARY})
    Install_Required_Library("${libpath}/libgfortran")
    Install_Required_Library("${libpath}/libquadmath")
    Install_Required_Library("${libpath}/libgcc")
  ENDIF()
  IF (BLAS_openblas_LIBRARY)
    Install_Required_Library(${BLAS_openblas_LIBRARY})
    Install_Required_Library("${libpath}/libquadmath")
    Install_Required_Library("${libpath}/libgfortran")
    Install_Required_Library("${libpath}/libgcc")
  ENDIF()
ENDIF()

# Create THConfig.cmake. TARGET_LINKER_FILE_NAME is the file to link against:
# libTH.so on Linux, the import library libTH.dll.a on Windows.
SET(TH_LIBRARIES "${CMAKE_INSTALL_PREFIX}/${TH_INSTALL_LIB_SUBDIR}/$<TARGET_LINKER_FILE_NAME:TH>")
SET(TH_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${TH_INSTALL_INCLUDE_SUBDIR}/TH")
CONFIGURE_FILE(THConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/THConfig.cmake.gen")
FILE(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/THConfig.cmake"
  INPUT "${CMAKE_CURRENT_BINARY_DIR}/THConfig.cmake.gen")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/THConfig.cmake"
  DESTINATION "${TH_INSTALL_CMAKE_SUBDIR}")

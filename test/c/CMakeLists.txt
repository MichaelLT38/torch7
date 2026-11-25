# C-level tests for TH: one executable, one ctest entry per suite.
ADD_EXECUTABLE(th_tests test_th.c th_test.h generic/th_tensor_tests.c test_iter.c)
SET_SOURCE_FILES_PROPERTIES(test_iter.c PROPERTIES HEADER_FILE_ONLY TRUE)
SET_SOURCE_FILES_PROPERTIES(generic/th_tensor_tests.c PROPERTIES HEADER_FILE_ONLY TRUE)
TARGET_INCLUDE_DIRECTORIES(th_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/lib/TH"
  "${PROJECT_BINARY_DIR}/lib/TH")
TARGET_LINK_LIBRARIES(th_tests PRIVATE TH m)

FOREACH(suite byte char short int long float double misc large iter)
  ADD_TEST(NAME th.${suite} COMMAND th_tests ${suite})
ENDFOREACH()

# The same suites on the generic (non-AVX2) kernels: built-in GEMM, vectorized
# float math, and TH's own SIMD dispatch all honor TH_NO_AVX2=1.
FOREACH(suite int float double misc)
  ADD_TEST(NAME th.${suite}.noavx2 COMMAND th_tests ${suite})
  SET_TESTS_PROPERTIES(th.${suite}.noavx2 PROPERTIES ENVIRONMENT "TH_NO_AVX2=1")
ENDFOREACH()

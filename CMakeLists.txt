cmake_minimum_required(VERSION 2.6)

project(imagequant C)

option(BUILD_WITH_SSE "Use SSE" ON)

if(BUILD_WITH_SSE)
  add_definitions(-DUSE_SSE=1)
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${CMAKE_SOURCE_DIR})

if(CMAKE_COMPILER_IS_GNUCC)
  add_compile_options("-std=c99")
endif()

add_library(imagequant SHARED 
  libimagequant.c
  blur.c
  mediancut.c
  mempool.c
  nearest.c
  pam.c
  kmeans.c
)

add_library(imagequant_a STATIC
  libimagequant.c
  blur.c
  mediancut.c
  mempool.c
  nearest.c
  pam.c
  kmeans.c
)

set(PREFIX ${CMAKE_INSTALL_PREFIX})
file(READ version.txt VERSION)
configure_file(imagequant.pc.in imagequant.pc @ONLY)

install(TARGETS imagequant        LIBRARY           DESTINATION lib)
install(FILES                     libimagequant.h   DESTINATION include)
install(FILES ${CMAKE_BINARY_DIR}/libimagequant_a.a DESTINATION lib RENAME libimagequant.a)
install(FILES ${CMAKE_BINARY_DIR}/imagequant.pc     DESTINATION lib/pkgconfig)

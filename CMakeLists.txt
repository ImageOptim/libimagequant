cmake_minimum_required(VERSION 2.6)

project(imagequant C)

# add_definitions(-DUSE_SSE=1)

include_directories(${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}")

add_library(imagequant SHARED 
  libimagequant.c
  blur.c
  mediancut.c
  mempool.c
  nearest.c
  pam.c
  kmeans.c
)
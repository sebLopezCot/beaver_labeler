cmake_minimum_required(VERSION 3.16)
project(BeaverLabeler)

set(CMAKE_CXX_STANDARD 17)

# --- Find PCL (common, IO, visualization) ---
find_package(PCL 1.12 REQUIRED COMPONENTS
  common
  io
  visualization
  filters
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(BeaverLabeler
  main.cpp
)

target_link_libraries(BeaverLabeler
  ${PCL_LIBRARIES}
)


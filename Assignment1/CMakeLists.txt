cmake_minimum_required(VERSION 3.13.4)
project(HoistAnticipatedExpressions)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

add_library(HoistAnticipatedExpressions MODULE
    lib/pass.cpp
)

# Use C++14 or higher
target_compile_features(HoistAnticipatedExpressions PRIVATE cxx_std_17)

if(APPLE)
    set_target_properties(HoistAnticipatedExpressions PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)
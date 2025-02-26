cmake_minimum_required(VERSION 3.13.4)
project(AnticipatedExpressions)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

add_library(AnticipatedExpressions MODULE
    lib/pass.cpp
)

# Use C++14 or higher
target_compile_features(AnticipatedExpressions PRIVATE cxx_std_17)

if(APPLE)
    set_target_properties(AnticipatedExpressions PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)
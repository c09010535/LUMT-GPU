cmake_minimum_required(VERSION 3.1)

project(lu)

set(PRJ_HEADER_LIST)
set(PRJ_SRC_LIST)
set(PRJ_COMPILE_DEF)
set(PRJ_LIBRARIES)
set(PRJ_COMPILE_OPTIONS)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("FOUND CUDA!")
    enable_language(CUDA)
    #set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_ARCHITECTURES 86)
else()
    message("NOT FOUND CUDA!")
endif(CUDA_FOUND)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    list(APPEND PRJ_COMPILE_DEF ENABLE_OPENMP)
    list(APPEND PRJ_LIBRARIES ${OpenMP_C_LIBRARIES})
    list(APPEND PRJ_COMPILE_OPTIONS ${OpenMP_C_FLAGS})
endif (OPENMP_FOUND)


file(GLOB cpu_header_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB cpu_src_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
file(GLOB cuda_header_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh")
file(GLOB cuda_src_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")



list(APPEND PRJ_HEADER_LIST ${cpu_header_files})
list(APPEND PRJ_HEADER_LIST ${cuda_header_files})
list(APPEND PRJ_SRC_LIST ${cpu_src_files})
list(APPEND PRJ_SRC_LIST ${cuda_src_files})



link_libraries(m)
#add_executable(lu ${cpu_src_files} ${cpu_header_files})
#cuda_add_library(gpu STATIC ${cuda_header_files} ${cuda_src_files})
#list(APPEND PRJ_LIBRARIES gpu)

add_executable(lu ${PRJ_SRC_LIST} ${PRJ_HEADER_LIST})

set_target_properties(lu PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(lu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(lu PRIVATE ${PRJ_COMPILE_OPTIONS})
target_link_libraries(lu PRIVATE ${PRJ_LIBRARIES})
cmake_minimum_required(VERSION 3.14)
project(deploy_test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -ldl -pthread")
#SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR})
SET(HOME_TVM /root/tvm)

# 引入Opencv包
find_package(OpenCV 4.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})




INCLUDE_DIRECTORIES(${HOME_TVM}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dlpack/include)





set(PNG_LIBS libpng.a)
set(Z_LIBS libz.a)
set(JPEG_LIBS libjpeg.a)
link_directories(${PROJECT_SOURCE_DIR}/build)

add_executable(deploy_test  tvm_runtime_pack.cc deploy_so.cpp)
target_link_libraries(deploy_test    ${OpenCV_LIBS} ${CMAKE_DL_LIBS}  ${JPEG_LIBS} ${PNG_LIBS} ${Z_LIBS})



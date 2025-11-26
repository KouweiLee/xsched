# ==================== service_common 静态库 ====================
# 这个部分定义了一个名为service_common的静态库
# 它包含了项目中共用的功能代码

# 使用GLOB_RECURSE递归搜索common/src目录下的所有.cpp文件
# 搜索到的文件路径会存储在SERVICE_COMMON_SRCS变量中
file(GLOB_RECURSE SERVICE_COMMON_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/common/src/*.cpp
)
# 创建名为service_common的静态库
# STATIC表示这是静态库，会被直接链接到使用它的程序中
add_library(service_common STATIC ${SERVICE_COMMON_SRCS})

# 设置头文件包含路径
# PUBLIC表示这些包含路径不仅适用于库本身，也会传递给使用这个库的其他目标
target_include_directories(service_common PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/common/include
)

# 设置编译选项
# PRIVATE表示这些选项只适用于库本身的编译
# -fPIC: 生成位置无关代码，用于动态链接
# -Wall: 启用所有常见警告
# -Wextra: 启用额外的警告
# -Werror: 将所有警告视为错误
target_compile_options(service_common PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

# 指定这个库依赖的其他库
# PUBLIC表示这些依赖关系会传递给使用service_common的其他目标
target_link_libraries(service_common PUBLIC
    jsoncpp_static
    httplib
    utils
    sched
)

# ==================== xcli 可执行文件 ====================
# 这个部分定义了一个名为xcli的可执行文件
# 它是项目的命令行工具

# 搜索cli/src目录下的所有.c和.cpp文件
file(GLOB_RECURSE XCLI_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/cli/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/cli/src/*.cpp
)
# 创建xcli可执行文件
add_executable(xcli ${XCLI_SRCS})

# 设置头文件包含路径
target_include_directories(xcli PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/common
    ${CMAKE_CURRENT_LIST_DIR}/cli/include
)

# 设置编译选项
# 启用警告并将警告视为错误
target_compile_options(xcli PRIVATE
    -Wall
    -Wextra
    -Werror
)

# 指定xcli需要链接的库
# PRIVATE表示这些依赖关系不会传递给其他目标
target_link_libraries(xcli PRIVATE
    CLI11
    tabulate
    utils
    sched
    protocol
    service_common
)

# 设置RPATH相关属性
# RPATH是运行时库搜索路径
# 这些设置确保程序在安装后能正确找到依赖的动态库
set_target_properties(xcli PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS xcli RUNTIME DESTINATION bin)

# ==================== xserver 可执行文件 ====================
# 这个部分定义了一个名为xserver的可执行文件
# 它是项目的服务器组件

# 搜索server/src目录下的所有.c和.cpp文件
file(GLOB_RECURSE XSERVER_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/server/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/server/src/*.cpp
)
# 创建xserver可执行文件
add_executable(xserver ${XSERVER_SRCS})

# 设置头文件包含路径
target_include_directories(xserver PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/common
    ${CMAKE_CURRENT_LIST_DIR}/server/include
)

# 设置编译选项
# 启用警告并将警告视为错误
target_compile_options(xserver PRIVATE
    -Wall
    -Wextra
    -Werror
)

# 指定xserver需要链接的库
# PRIVATE表示这些依赖关系不会传递给其他目标
target_link_libraries(xserver PRIVATE
    ipc
    httplib
    jsoncpp_static
    utils
    sched
    protocol
    service_common
)

# 设置RPATH相关属性
# RPATH是运行时库搜索路径
# 这些设置确保程序在安装后能正确找到依赖的动态库
set_target_properties(xserver PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS xserver RUNTIME DESTINATION bin)

# 添加tools子目录到构建系统
# CMake会去处理tools目录下的CMakeLists.txt文件
add_subdirectory(tools)

# 创建CUDA平台的HAL(Hardware Abstraction Layer，硬件抽象层)库
# 这个库提供了与CUDA硬件交互的统一接口
# LINK_LIBRARIES指定需要链接的库，这里链接了cuxtra库
add_hal_lib(cuda LINK_LIBRARIES cuxtra)

# 创建CUDA平台的shim库（垫片库）
# shim库用于拦截和重定向CUDA API调用
# SOFTLINKS指定需要创建的软链接，这些是CUDA运行时库的标准名称
# libcuda.so.1是主要的CUDA驱动库
# libcuda.so是一个指向libcuda.so.1的符号链接
add_shim_lib(cuda SOFTLINKS libcuda.so.1 libcuda.so)

# 添加平台测试
# 这会创建用于测试CUDA平台功能的测试用例
add_platform_test()

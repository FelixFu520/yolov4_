#include <opencv4/opencv2/opencv.hpp>

#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>

#include <iterator>
#include <algorithm>
#include <sys/time.h>
#include <iostream>
#include <typeinfo>

static uint64_t getCurrentTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

void Mat_to_CHW(float *data, cv::Mat &frame)
{
    assert(data && !frame.empty());
    unsigned int volChl = 416 * 416;

    for(int c = 0; c < 3; ++c)
    {
        for (unsigned j = 0; j < volChl; ++j)
            data[c*volChl + j] = static_cast<float>(float(frame.data[j * 3 + c]) / 255.0);
    }

}

int main(void) {
    std::cout<<"Start !!!!"<<std::endl;
    
    std::cout<<"load so!"<<"\n";
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("../../../models/yolov4_pc.so");

    std::cout<<"load json"<<"\n";
    std::ifstream json_in("../../../models/yolov4_pc.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();
  
    std::cout<<"load params"<<std::endl;
    std::ifstream params_in("../../../models/yolov4_pc.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;

    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))
        (json_data, mod_dylib, device_type, device_id);
    DLTensor *x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 416, 416};
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    
    // 这里依然读取了papar.png这张图
    cv::Mat image , frame, input;
//     image = cv::imread("../../street.jpg");
    image = cv::imread("../../test.png");
    cv::cvtColor(image, frame, cv::COLOR_BGR2RGB);
    cv::resize(frame, input,  cv::Size(416,416));
    float data[416 * 416 * 3];
    // 在这个函数中 将OpenCV中的图像数据转化为CHW的形式 
    Mat_to_CHW(data, input);
    

    // x为之前的张量类型 data为之前开辟的浮点型空间
    memcpy(x->data, &data, 3 * 416 * 416 * sizeof(float));
    
    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("input_0", x);

    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    DLTensor* y0;
    DLTensor* y1;
    DLTensor* y2;
    
    int out_ndim0 = 4;
    int64_t out_shape0[4] = {1, 255, 13,13,};
    int out_ndim1 = 4;
    int64_t out_shape1[4] = {1, 255, 26,26,};
    int out_ndim2 = 4;
    int64_t out_shape2[4] = {1, 255, 52,52,};
    
    TVMArrayAlloc(out_shape0, out_ndim0, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y0);
    TVMArrayAlloc(out_shape1, out_ndim1, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y1);
    TVMArrayAlloc(out_shape2, out_ndim2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y2);

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    
    run();
    get_output(0, y0);
    get_output(1, y1);
    get_output(2, y2);
    
    
    //计算4000次运行时间
//     uint64_t start_time = getCurrentTime();
//     for(int i = 0; i < 100; i++) {
//         run();
//     } 
//     uint64_t finish_time = getCurrentTime();
//     std::cout << "done(" << (finish_time - start_time) / 1000 / 1000<< " ms)." << std::endl;
    

    // 将输出的信息打印出来
    auto result = static_cast<float*>(y0->data);
//     std::cout << typeid( y0 ).name() << std::endl; //打印类型
//     std::cout <<typeid(result).name()<<std::endl;
    
    // 输出shape
//     for(int i=0;i<4;i++){
//         std::cout<<y0->shape[i]<<“, ”;
//     }
    
        
    for(int i =0;i<13;i++){
        std::cout<<result[0,0,0,i]<<std::endl;
    }
    
    return 1;
}
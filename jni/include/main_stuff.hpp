#ifndef MAIN_STUFF
#define MAIN_STUFF

#include "android_main.h"
#include <cassert>
#include <iostream>
#include <cstring>


void runtimeArgParse(
    char** argv, 
    int argc,
    std::map<std::string, RuntimeParams>& model_runtimes
    ) 
{
    int i = 0;
    std::string model_name;
    std::cout << "Reading model names: ";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) != "-m") {
            continue;
        }
        model_name = argv[i + 1];
        std::cout << model_name << ", ";
        model_runtimes[model_name] = RuntimeParams();
        if (std::strncmp(argv[i + 2], "dsp08", 5) == 0) { 
            model_runtimes[model_name].runtime_type = zdl::DlSystem::Runtime_t::DSP;
            model_runtimes[model_name].datasize = 1;
            model_runtimes[model_name].isTFBuffer = true;
        }
        else if (std::strncmp(argv[i + 2], "gpu32", 5) == 0) { 
            model_runtimes[model_name].runtime_type = zdl::DlSystem::Runtime_t::GPU;
            model_runtimes[model_name].datasize = 4;
            model_runtimes[model_name].isTFBuffer = false;
        }
        else if (std::strncmp(argv[i + 2], "gpu16", 5) == 0) { 
            model_runtimes[model_name].runtime_type = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
            model_runtimes[model_name].datasize = 2;
            model_runtimes[model_name].isTFBuffer = false;
        }
        else if (std::strncmp(argv[i + 2], "cpu32", 5) == 0) { 
            model_runtimes[model_name].runtime_type = zdl::DlSystem::Runtime_t::CPU;
            model_runtimes[model_name].datasize = 4;
            model_runtimes[model_name].isTFBuffer = false;
        }
        else {
            std::cerr << "problem with parsing argv[index]: " << argv[i + 2] << "\n";
        }
    }
    std::cout << "\n";
}

#endif
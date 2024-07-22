#ifndef SNPE_TUTORIAL_UTILS_H_
#define SNPE_TUTORIAL_UTILS_H_

#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "LoadUDOPackage.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "CreateUserBuffer.hpp"
#include "PreprocessInput.hpp"
#include "SaveOutputTensor.hpp"
#include "Util.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"

#include "DlSystem/TensorShape.hpp"
#include "DlSystem/StringList.hpp"

#include "DlSystem/IUserBuffer.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DiagLog/IDiagLog.hpp"
#include "DlSystem/DlError.hpp"


#include "DlSystem/DlEnums.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/PlatformConfig.hpp"

#include <iostream>
#include <vector>
#include <string> // not sure if i need
#include <cassert>
#include <fstream>
#include <tuple>
#include <map>

#include "SNPE/SNPEBuilder.hpp"

#define DEBUG


zdl::DlSystem::Runtime_t checkRuntime(const zdl::DlSystem::Runtime_t runtime)
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    static zdl::DlSystem::Runtime_t Runtime;
    std::cout << "Qualcomm (R) Neural Processing SDK Version: " << Version.asString().c_str() << std::endl; //Print Version number
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        Runtime = runtime;
    }
    else {
        std::cerr << "Targeted runtime: " << (int)runtime << "is not available\n";
        exit(1);
    }
    return Runtime;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers)
{
   std::unique_ptr<zdl::SNPE::SNPE> snpe;
   zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
   // added to make dynamic
   printf("snpe before: %p\n", snpe.get());
   // std::cout << "container.get()" << container.get() << std::endl;
   // zdl::DlSystem::TensorShapeMap inputShapeMap;
   // inputShapeMap.add("vector:0", {2,1,1,10}); // try other dims (2 instead of 1), try inccorect dims (1000->1001) (shoudl fail)
   // zdl::DlSystem::TensorShape shape(new_shape);
   // inputShapeMap.add(name, new_shape);
   // snpeBuilder.setInputDimensions(inputShapeMap);
   // end of add
   std::cout << "going to build\n";
   snpe = snpeBuilder.setOutputLayers({})
      .setRuntimeProcessorOrder(runtimeList)
      .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
      .build();
   std::cout << "finished building\n";
   printf("snpe after: %p\n", snpe.get());
   return snpe;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(const std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                                   const zdl::DlSystem::RuntimeList& runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   const zdl::DlSystem::PlatformConfig& platformConfig)
{
   std::unique_ptr<zdl::SNPE::SNPE> snpe;
   zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
   
   printf("snpe before: %p\n", snpe.get());
//    snpe = snpeBuilder.setOutputLayers({})
//       .setRuntimeProcessorOrder(runtimeList)
//       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
//       .setPlatformConfig(platformConfig)
//       .build();

    // experiment, restore the one above later
    bool useCaching = false;
    bool cpuFixedPointMode = false;
    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .setCpuFixedPointMode(cpuFixedPointMode)
       .build();

   printf("snpe after: %p\n", snpe.get());
   return snpe;
}


std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(
    std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
    zdl::DlSystem::RuntimeList runtimeList,
    bool useUserSuppliedBuffers,
    std::unordered_map<std::string, std::vector<size_t>> dims_dict)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    // added to make dynamic
    printf("snpe before: %p\n", snpe.get());
    // std::cout << "container.get()" << container.get() << std::endl;
    
    for (const auto& pair : dims_dict) {
        zdl::DlSystem::TensorShape shape(pair.second);
        zdl::DlSystem::TensorShapeMap inputShapeMap;
        inputShapeMap.add(pair.first.c_str(), shape);
        snpeBuilder.setInputDimensions(inputShapeMap);
    }

    // end of add
    std::cout << "going to build\n";
    snpe = snpeBuilder.setOutputLayers({})
        .setRuntimeProcessorOrder(runtimeList)
        .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
        .build();
    std::cout << "finished building\n";
    printf("snpe after: %p\n", snpe.get());
    return snpe;
}


std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(
    std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
    zdl::DlSystem::RuntimeList runtimeList,
    bool useUserSuppliedBuffers,
    std::unordered_map<std::string, std::tuple<std::vector<size_t>,zdl::DlSystem::IOBufferDataType_t>> dims_dict,
    std::unordered_map<std::string, zdl::DlSystem::IOBufferDataType_t> output_datatypes
    )
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    zdl::DlSystem::IOBufferDataTypeMap dataMap;

    printf("snpe before: %p\n", snpe.get());
    // std::cout << "container.get()" << container.get() << std::endl;
    
    for (const auto& pair : dims_dict) {
        zdl::DlSystem::TensorShape shape(std::get<0>(pair.second));
        zdl::DlSystem::TensorShapeMap inputShapeMap;
        inputShapeMap.add(pair.first.c_str(), shape);
        snpeBuilder.setInputDimensions(inputShapeMap);

        // setting IO Stuff
        dataMap.add(pair.first.c_str(), std::get<1>(pair.second));
    }

    // setting IO Stuff for output
    for (const auto& pair : output_datatypes) {
        dataMap.add(pair.first.c_str(), pair.second);
    }

    snpeBuilder.setBufferDataType(dataMap);

    // end of add
    std::cout << "going to build\n";
    snpe = snpeBuilder.setOutputLayers({})
        .setRuntimeProcessorOrder(runtimeList)
        .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
        .build(); 
    std::cout << "finished building\n";
    printf("snpe after: %p\n", snpe.get());
    return snpe;
}


// taken from example code
std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions_ex(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching, bool cpuFixedPointMode)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }
    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
    //    .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .setCpuFixedPointMode(cpuFixedPointMode)
       .build();
    return snpe;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions_ex_multipleOuts(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching, bool cpuFixedPointMode,
                                                   const std::vector<std::string>& outputNames)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    zdl::DlSystem::StringList outputNamesList;
    std::cout << "appending\n";
    for (size_t i = 0; i < outputNames.size(); i++) { 
        std::cout << "i: " << i << "\n";
        outputNamesList.append(outputNames[i].c_str()); 
    }
    std::cout << "done appending\n";
    // for (size_t i = 0; i < outputNames.size(); i++) { outputNamesList.append(outputNames[i].c_str()); }

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }
    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
    //    .setOutputLayers(outputNamesList)
        .setOutputTensors(outputNamesList)
    //    .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .setCpuFixedPointMode(cpuFixedPointMode)
       .build();
    return snpe;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions_reshape(
    std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
    zdl::DlSystem::Runtime_t runtime,
    zdl::DlSystem::RuntimeList runtimeList,
    bool useUserSuppliedBuffers,
    // zdl::DlSystem::PlatformConfig platformConfig,
    bool useCaching, bool cpuFixedPointMode,
    const std::vector<std::pair<std::string, std::vector<size_t>>>& dims_dict)
{
    #ifdef DEBUG
        std::cout << "checkpoint 1\n";
    #endif

    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    
    #ifdef DEBUG
        std::cout << "checkpoint 2\n";
    #endif
    
    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    zdl::DlSystem::TensorShapeMap inputShapeMap;

    for (const auto& pair : dims_dict) {
        std::cout << "\tinput: " << pair.first << "\n";
        zdl::DlSystem::TensorShape shape(pair.second);
        // zdl::DlSystem::TensorShapeMap inputShapeMap;
        inputShapeMap.add(pair.first.c_str(), shape);
        snpeBuilder.setInputDimensions(inputShapeMap);
    }

    #ifdef DEBUG
        std::cout << "checkpoint 3\n";
    #endif

    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
    //    .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .setCpuFixedPointMode(cpuFixedPointMode)
       .build();

    #ifdef DEBUG
        std::cout << "checkpoint 4\n";
    #endif

    #ifdef DEBUG
        printf("snpe after: %p\n", snpe.get());
    #endif

    return snpe;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions_ex_multipleOuts_reshape(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching, bool cpuFixedPointMode,
                                                   const std::vector<std::string>& outputNames,
                                                   const std::vector<std::pair<std::string, std::vector<size_t>>>& dims_dict)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    zdl::DlSystem::StringList outputNamesList;
    std::cout << "appending\n";
    for (size_t i = 0; i < outputNames.size(); i++) { 
        std::cout << "i: " << i << "\n";
        outputNamesList.append(outputNames[i].c_str()); 
    }
    std::cout << "done appending\n";
    // for (size_t i = 0; i < outputNames.size(); i++) { outputNamesList.append(outputNames[i].c_str()); }

    zdl::DlSystem::TensorShapeMap inputShapeMap;

    for (const auto& pair : dims_dict) {
        std::cout << "\tinput: " << pair.first << "\n";
        zdl::DlSystem::TensorShape shape(pair.second);
        inputShapeMap.add(pair.first.c_str(), shape);
        snpeBuilder.setInputDimensions(inputShapeMap);
    }

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    std::cout << "\n\nsnpe.get() before: " << snpe.get() << "\n";

    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
    //    .setOutputLayers(outputNamesList)
        .setOutputTensors(outputNamesList)
    //    .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .setCpuFixedPointMode(cpuFixedPointMode)
       .build();
    
    std::cout << "\n\nsnpe.get() after: " << snpe.get() << "\n";
    // exit(0);

    return snpe;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(
    std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
    zdl::DlSystem::Runtime_t runtime,
    zdl::DlSystem::RuntimeList runtimeList,
    bool useUserSuppliedBuffers,
    zdl::DlSystem::PlatformConfig platformConfig,
    bool useCaching, bool cpuFixedPointMode,
    std::unordered_map<std::string, std::tuple<std::vector<size_t>,zdl::DlSystem::IOBufferDataType_t>> dims_dict,
    std::unordered_map<std::string, zdl::DlSystem::IOBufferDataType_t> output_datatypes)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    zdl::DlSystem::IOBufferDataTypeMap dataMap;
    for (const auto& pair : dims_dict) {
        zdl::DlSystem::TensorShape shape(std::get<0>(pair.second));
        zdl::DlSystem::TensorShapeMap inputShapeMap;
        inputShapeMap.add(pair.first.c_str(), shape);
        snpeBuilder.setInputDimensions(inputShapeMap);
        // setting IO Stuff
        dataMap.add(pair.first.c_str(), std::get<1>(pair.second));
    }
    // setting IO Stuff for output
    for (const auto& pair : output_datatypes) {
        dataMap.add(pair.first.c_str(), pair.second);
    }
    snpeBuilder.setBufferDataType(dataMap);


    printf("snpe before: %p\n", snpe.get());
    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .setCpuFixedPointMode(cpuFixedPointMode)
       .build();
    printf("snpe after: %p\n", snpe.get());
    return snpe;
}



void createUserBuffer(
    zdl::DlSystem::UserBufferMap& userBufferMap,
    std::unordered_map<std::string, std::vector<uint8_t>*>& applicationBuffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe,
    const char* name,
    size_t datasize,
    bool isTFBuffer
    )
{
//    auto name = snpe->getInputTensorNames().at(input_num);
   std::cout << "\nname from createUserBuffer: " << name << "\n";
   // std::cout << "input names:" << 
   // get attributes of buffer by name
   auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
   
   std::cout << "created bufferAttributesopt\n";
   if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);
   // calculate the size of buffer required by the input tensor
   const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
   // Calculate the stride based on buffer strides, assuming tightly packed.
   // Note: Strides = Number of bytes to advance to the next element in each dimension.
   // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
   // Note: Buffer stride is usually known and does not need to be calculated.
   std::vector<size_t> strides(bufferShape.rank());
   std::cout << "bufferShape.rank(): " << bufferShape.rank() << "\n";
   strides[strides.size() - 1] = datasize;
   std::cout << "got strides\n";
   size_t stride = strides[strides.size() - 1];
   for (size_t i = bufferShape.rank() - 1; i > 0; i--)
   {
      stride *= bufferShape[i];
      strides[i-1] = stride;
   }

   const size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
   std::cout << "bufferElementSize: " << bufferElementSize << "\n";
   assert(bufferElementSize == datasize); // probably fial for now
   size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);
        // dummy code:
      //   size_t bufSize = bufferElementSize;
   // set the buffer encoding type

    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> userBufferEncoding;
    std::cout << "userBufferEncoding before intialization: " << userBufferEncoding.get() << "\n";

   if (isTFBuffer) {
    std::cout << "datasize:" << datasize << "\n";
        std::cout << "using TFBuffer\n";
        const zdl::DlSystem::UserBufferEncodingTfN* ubeTfN = dynamic_cast<const zdl::DlSystem::UserBufferEncodingTfN*>((*bufferAttributesOpt)->getEncoding());
        uint64_t stepEquivalentTo0 = ubeTfN->getStepExactly0();
        float quantizedStepSize = ubeTfN->getQuantizedStepSize();
        std::cout << "stepEquivalentTo0: " << stepEquivalentTo0 << "\n";
        std::cout << "qunatizedStepSize: " << quantizedStepSize << "\n";
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingTfN>(new zdl::DlSystem::UserBufferEncodingTfN(stepEquivalentTo0,quantizedStepSize, 8 * datasize));
        std::cout << "userBufferEncoding in the middle of intialization: " << userBufferEncoding.get() << "\n";
   }
   else {
        std::cout << "NOT using TFBuffer\n";
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingFloat>(new zdl::DlSystem::UserBufferEncodingFloat());
        std::cout << "userBufferEncoding in the middle of intialization: " << userBufferEncoding.get() << "\n";

   }

   std::cout << "userBufferEncoding after intialization: " << userBufferEncoding.get() << "\n";


   std::cout << "strides: ";
   for (auto i : strides) {
      std::cout << i << ", ";
   }
   std::cout << "\nbufSize:" << bufSize << "\n";


    /* REMOVE THIS LINE LATER */
    std::string name_std = std::string(name);
    std::cout << "name_std: " << name_std << "\n";
    // // applicationBuffers.emplace(name_std, std::vector<uint8_t>(bufSize));


//    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
   // create user-backed storage to load input data onto it
   // applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize)); // was in the orginal
   std::cout << (*(applicationBuffers.at(name_std))).size() << " >= " << bufSize << "?\n";
   // assert(applicationBuffers.at(name).size() == bufSize);
   assert((*(applicationBuffers.at(name_std))).size() >= bufSize); // modified to make more dynamic
   // create Qualcomm (R) Neural Processing SDK user buffer from the user-backed buffer
   zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();

   std::cout << "ubFactory memory address: " << &ubFactory << "\n";

    std::cout << "applicationBuffers.at(name).data(): " << (void*)((*(applicationBuffers.at(name_std))).data()) << "\n";

    // temp
    auto x3 = ubFactory.createUserBuffer((*(applicationBuffers.at(name_std))).data(),
                                                              bufSize,
                                                              strides,
                                                              userBufferEncoding.get());
    std::cout << "x.get(): " << x3.get() <<"\n";
    // std::cout << "x: " << x << "\n";
    if (x3.get() == nullptr) {
        std::cout << "problem 1\n";
    }
    if (x3 == nullptr) {
        std::cout << "problem2\n";
    }
    // end of temp


   snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer((*(applicationBuffers.at(name_std))).data(),
                                                              bufSize,
                                                              strides,
                                                              userBufferEncoding.get()));
   if (snpeUserBackedBuffers.back() == nullptr)
   {
      std::cerr << "Error while creating user buffer." << std::endl;
      exit(1);
   }

    // std::cout << "trying before <may segfault>\n";
    //            auto y = userBufferMap.getUserBufferNames();
    //     std::cout << "userBufferMaps: ";
    //     std::for_each(y.begin(), y.end(), [&](const char* str) {
    //         std::cout << str << " ";
    //     });
    //     std::cout << "\n";


   // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
   userBufferMap.add(name, snpeUserBackedBuffers.back().get());
   

        //    auto x = userBufferMap.getUserBufferNames();
        // std::cout << "userBufferMaps: ";
        // std::for_each(x.begin(), x.end(), [&](const char* str) {
        //     std::cout << str << " ";
        // });
        // std::cout << "\n";
}

void createMemoryMapUserBuffer(
    zdl::DlSystem::UserBufferMap& userBufferMap,
    zdl::DlSystem::UserMemoryMap& memoryMap,
    std::unordered_map<std::string, std::vector<uint8_t>*>& applicationBuffers,
    std::vector<uint8_t>& temp_buff,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe,
    const char* name,
    size_t datasize,
    bool isTFBuffer
    )
{
//    auto name = snpe->getInputTensorNames().at(input_num);
   std::cout << "\nname from createUserBuffer: " << name << "\n";
   // std::cout << "input names:" << 
   // get attributes of buffer by name
   auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
   
   std::cout << "created bufferAttributesopt\n";
   if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);
   // calculate the size of buffer required by the input tensor
   const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
   // Calculate the stride based on buffer strides, assuming tightly packed.
   // Note: Strides = Number of bytes to advance to the next element in each dimension.
   // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
   // Note: Buffer stride is usually known and does not need to be calculated.
   std::vector<size_t> strides(bufferShape.rank());
   std::cout << "bufferShape.rank(): " << bufferShape.rank() << "\n";
   strides[strides.size() - 1] = datasize;
   std::cout << "got strides\n";
   size_t stride = strides[strides.size() - 1];
   for (size_t i = bufferShape.rank() - 1; i > 0; i--)
   {
      stride *= bufferShape[i];
      strides[i-1] = stride;
   }

   const size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
   std::cout << "bufferElementSize: " << bufferElementSize << "\n";
   assert(bufferElementSize == datasize); // probably fial for now
   size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);
        // dummy code:
      //   size_t bufSize = bufferElementSize;
   // set the buffer encoding type

    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> userBufferEncoding;
    std::cout << "userBufferEncoding before intialization: " << userBufferEncoding.get() << "\n";

   if (isTFBuffer) {
    std::cout << "datasize:" << datasize << "\n";
        std::cout << "using TFBuffer\n";
        const zdl::DlSystem::UserBufferEncodingTfN* ubeTfN = dynamic_cast<const zdl::DlSystem::UserBufferEncodingTfN*>((*bufferAttributesOpt)->getEncoding());
        uint64_t stepEquivalentTo0 = ubeTfN->getStepExactly0();
        float quantizedStepSize = ubeTfN->getQuantizedStepSize();
        std::cout << "stepEquivalentTo0: " << stepEquivalentTo0 << "\n";
        std::cout << "qunatizedStepSize: " << quantizedStepSize << "\n";
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingTfN>(new zdl::DlSystem::UserBufferEncodingTfN(stepEquivalentTo0,quantizedStepSize, 8 * datasize));
        std::cout << "userBufferEncoding in the middle of intialization: " << userBufferEncoding.get() << "\n";
   }
   else {
        std::cout << "NOT using TFBuffer\n";
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingFloat>(new zdl::DlSystem::UserBufferEncodingFloat());
        std::cout << "userBufferEncoding in the middle of intialization: " << userBufferEncoding.get() << "\n";

   }

   std::cout << "userBufferEncoding after intialization: " << userBufferEncoding.get() << "\n";


   std::cout << "strides: ";
   for (auto i : strides) {
      std::cout << i << ", ";
   }
   std::cout << "\nbufSize:" << bufSize << "\n";


    /* REMOVE THIS LINE LATER */
    std::string name_std = std::string(name);
    std::cout << "name_std: " << name_std << "\n";
    // // applicationBuffers.emplace(name_std, std::vector<uint8_t>(bufSize));


//    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
   // create user-backed storage to load input data onto it
   // applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize)); // was in the orginal
   std::cout << (*(applicationBuffers.at(name_std))).size() << " >= " << bufSize << "?\n";
   // assert(applicationBuffers.at(name).size() == bufSize);
   assert((*(applicationBuffers.at(name_std))).size() >= bufSize); // modified to make more dynamic
   // create Qualcomm (R) Neural Processing SDK user buffer from the user-backed buffer
   zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();

   std::cout << "ubFactory memory address: " << &ubFactory << "\n";

    std::cout << "applicationBuffers.at(name).data(): " << (void*)((*(applicationBuffers.at(name_std))).data()) << "\n";

    // temp
    auto x3 = ubFactory.createUserBuffer((*(applicationBuffers.at(name_std))).data(),
                                                              bufSize,
                                                              strides,
                                                              userBufferEncoding.get());
    std::cout << "x.get(): " << x3.get() <<"\n";
    // std::cout << "x: " << x << "\n";
    if (x3.get() == nullptr) {
        std::cout << "problem 1\n";
    }
    if (x3 == nullptr) {
        std::cout << "problem2\n";
    }
    // end of temp


   snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer((*(applicationBuffers.at(name_std))).data(),
                                                              bufSize,
                                                              strides,
                                                              userBufferEncoding.get()));
   if (snpeUserBackedBuffers.back() == nullptr)
   {
      std::cerr << "Error while creating user buffer." << std::endl;
      exit(1);
   }

    // std::cout << "trying before <may segfault>\n";
    //            auto y = userBufferMap.getUserBufferNames();
    //     std::cout << "userBufferMaps: ";
    //     std::for_each(y.begin(), y.end(), [&](const char* str) {
    //         std::cout << str << " ";
    //     });
    //     std::cout << "\n";


   // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
   userBufferMap.add(name, snpeUserBackedBuffers.back().get());
   

        //    auto x = userBufferMap.getUserBufferNames();
        // std::cout << "userBufferMaps: ";
        // std::for_each(x.begin(), x.end(), [&](const char* str) {
        //     std::cout << str << " ";
        // });
        // std::cout << "\n";

    memoryMap.add(name, temp_buff.data());
}

void modifyUserBuffer(
    zdl::DlSystem::UserBufferMap& userBufferMap,
    std::unordered_map<std::string, std::vector<uint8_t>*>& applicationBuffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe,
    const char* name,
    size_t datasize,
    int ibuffer_index)
{
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);
    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
    std::vector<size_t> strides(bufferShape.rank());
    std::cout << "bufferShape.rank(): " << bufferShape.rank() << "\n";
    strides[strides.size() - 1] = datasize;
    std::cout << "got strides\n";
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--)
    {
        stride *= bufferShape[i];
        strides[i-1] = stride;
    }
    const size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
    std::cout << "bufferElementSize: " << bufferElementSize << "\n";
    assert(bufferElementSize == datasize); // probably fial for now
    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);
    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
    std::cout << (*(applicationBuffers.at(name))).size() << " >= " << bufSize << "?\n";
    assert((*(applicationBuffers.at(name))).size() >= bufSize);
    zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();

    snpeUserBackedBuffers[ibuffer_index] = ubFactory.createUserBuffer((*(applicationBuffers.at(name))).data(),
                                                              bufSize,
                                                              strides,
                                                              &userBufferEncodingFloat);
    userBufferMap.remove(name);
    userBufferMap.add(name, snpeUserBackedBuffers[ibuffer_index].get());
}

void modifyUserBufferWithMemoryMap(
    zdl::DlSystem::UserBufferMap& userBufferMap,
    zdl::DlSystem::UserMemoryMap& memoryMap,
    std::unordered_map<std::string, std::vector<uint8_t>*>& applicationBuffers,
    std::vector<uint8_t>& temp_buff,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe,
    const char* name,
    size_t datasize,
    int ibuffer_index)
{
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);
    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
    std::vector<size_t> strides(bufferShape.rank());
    std::cout << "bufferShape.rank(): " << bufferShape.rank() << "\n";
    strides[strides.size() - 1] = datasize;
    std::cout << "got strides\n";
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--)
    {
        stride *= bufferShape[i];
        strides[i-1] = stride;
    }
    const size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
    std::cout << "bufferElementSize: " << bufferElementSize << "\n";
    assert(bufferElementSize == datasize); // probably fial for now
    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);
    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
    std::cout << (*(applicationBuffers.at(name))).size() << " >= " << bufSize << "?\n";
    assert((*(applicationBuffers.at(name))).size() >= bufSize);
    zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();

    snpeUserBackedBuffers[ibuffer_index] = ubFactory.createUserBuffer((*(applicationBuffers.at(name))).data(),
                                                              bufSize,
                                                              strides,
                                                              &userBufferEncodingFloat);
    userBufferMap.remove(name);
    userBufferMap.add(name, snpeUserBackedBuffers[ibuffer_index].get());

    memoryMap.remove(name);
    memoryMap.add(name, temp_buff.data());
}

// void loadInputUserBuffer(
//     std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
//     std::unique_ptr<zdl::SNPE::SNPE>& snpe,
//     const std::string& fileLine)
// {
//     // get input tensor names of the network that need to be populated
//     const auto& inputNamesOpt = snpe->getInputTensorNames();
//     if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
//     const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
//     assert(inputNames.size() > 0);
//     // treat each line as a space-separated list of input files
//     std::vector<std::string> filePaths;
//     split(filePaths, fileLine, ' ');
//     // remove the line below
//     for (int i = 0; i < filePaths.size(); i++) {std::cout << filePaths[i] << " ";} std::cout << "\n";
//     if (inputNames.size()) std::cout << "Processing DNN Input: " << std::endl;
//     for (size_t i = 0; i < inputNames.size(); i++) {
//         const char* name = inputNames.at(i);
//         std::string filePath(filePaths[i]);
//         // print out which file is being processed
//         std::cout << "\t" << i + 1 << ") " << filePath << std::endl;
//         // load file content onto application storage buffer,
//         // on top of which, Qualcomm (R) Neural Processing SDK has created a user buffer
//         std::cout << "(name): " << name << "\n";
//         std::cout << "filePath: " << filePath << "\n";
//         std::string temp_name = std::string(name);
//         assert(filePath.substr(0, filePath.size()-4) == temp_name.substr(0, temp_name.size()-2));
//         applicationBuffers.at(name);
//         std::cout << "going into load file in vector of size: " << applicationBuffers.at(name).size() << "\n";
//         bool temp = loadByteDataFile(filePath, applicationBuffers.at(name));
//     };
// }

// template<typename T>
// bool MyloadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector)
// {
//    std::cout << "case 1\n";
//    std::ifstream in(inputFile, std::ifstream::binary);
//    if (!in.is_open() || !in.good())
//    {
//       std::cerr << "Failed to open input file: " << inputFile << "\n";
//    }
//    std::cout << "case 2\n";
//    in.seekg(0, in.end);
//    size_t length = in.tellg();
//    in.seekg(0, in.beg);
//    std::cout << "length: " << length << "\n";
//    if (length % sizeof(T) != 0) {
//       std::cerr << "Size of input file should be divisible by sizeof(dtype).\n";
//       return false;
//    }

//    if (loadVector.size() == 0) {
//       loadVector.resize(length / sizeof(T));
//    } else if (loadVector.size() < length / sizeof(T)) {
//       std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
//       loadVector.resize(length / sizeof(T));
//    }

//    if (!in.read(reinterpret_cast<char*>(&loadVector[0]), length))
//    {
//       std::cerr << "Failed to read the contents of: " << inputFile << "\n";
//    }
//    return true;
// }

// The following is a partial snippet of the function
void SaveUserBuffer(const std::string& path, const std::vector<uint8_t>& buffer) {
   
   std::ofstream os(path, std::ofstream::binary);
   if (!os)
   {
      std::cerr << "Failed to open output file for writing: " << path << "\n";
      std::exit(EXIT_FAILURE);
   }
   if (!os.write((char*)(buffer.data()), buffer.size()))
   {
      std::cerr << "Failed to write data to: " << path << "\n";
      std::exit(EXIT_FAILURE);
   }
}

void executeNetwork(
    std::unique_ptr<zdl::SNPE::SNPE>& snpe,
    zdl::DlSystem::UserBufferMap& inputMap,
    zdl::DlSystem::UserBufferMap& outputMap,
    std::unordered_map<std::string,std::vector<uint8_t>>& applicationOutputBuffers,
    const std::string& outputDir,
    int num)
{
    // Execute the network and store the outputs in user buffers specified in outputMap
    snpe->execute(inputMap, outputMap);
    // Get all output buffer names from the network
    const zdl::DlSystem::StringList& outputBufferNames = outputMap.getUserBufferNames();
    // Iterate through output buffers and print each output to a raw file
    std::for_each(outputBufferNames.begin(), outputBufferNames.end(), [&](const char* name)
    {
       std::ostringstream path;
       path << outputDir << "/Result_" << num << "/" << name << ".raw";
       SaveUserBuffer(path.str(), applicationOutputBuffers.at(name));
    });
}

std::vector<std::string> StringListToVector(zdl::DlSystem::StringList& str_list) {
    #ifdef DEBUG
        std::cout << "calling StringListToVector()\n";
        std::cout << "zdl::DlSystem::StringList.size() " << str_list.size() << "\n";
    #endif
    std::vector<std::string> vec;
    for (int i = 0; i < str_list.size(); i++) {
        vec.push_back(str_list.at(i));
    }
    return vec;
}


#endif
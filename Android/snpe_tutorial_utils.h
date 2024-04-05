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

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DiagLog/IDiagLog.hpp"

#include "SNPE/SNPEBuilder.hpp"

#include <iostream>
#include <vector>
#include <string> // not sure if i need
#include <cassert>
#include <fstream>

zdl::DlSystem::Runtime_t checkRuntime()
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    static zdl::DlSystem::Runtime_t Runtime;
    std::cout << "Qualcomm (R) Neural Processing SDK Version: " << Version.asString().c_str() << std::endl; //Print Version number
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::CPU)) {
        Runtime = zdl::DlSystem::Runtime_t::CPU;
    }
    else {
        std::cout << "ERROR!, CPU runtime not found!";
        Runtime = zdl::DlSystem::Runtime_t::CPU;
    }
    return Runtime;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   const char* name)
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

void createUserBuffer(
    zdl::DlSystem::UserBufferMap& userBufferMap,
    std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe,
    const char * name)
{
   std::cout << name << "\n";
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
   strides[strides.size() - 1] = sizeof(float);
   std::cout << "got strides\n";
   size_t stride = strides[strides.size() - 1];
   for (size_t i = bufferShape.rank() - 1; i > 0; i--)
   {
      stride *= bufferShape[i];
      strides[i-1] = stride;
   }
   const size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
   // size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
   size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);
        // dummy code:
      //   size_t bufSize = bufferElementSize;
   // set the buffer encoding type
   zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
   // create user-backed storage to load input data onto it
   // applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize)); // was in the orginal
   std::cout << applicationBuffers.at(name).size() << " >= " << bufSize << "?\n";
   // assert(applicationBuffers.at(name).size() == bufSize);
   assert(applicationBuffers.at(name).size() >= bufSize); // modified to make more dynamic
   // create Qualcomm (R) Neural Processing SDK user buffer from the user-backed buffer
   zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
   snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(applicationBuffers.at(name).data(),
                                                              bufSize,
                                                              strides,
                                                              &userBufferEncodingFloat));
   // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
   userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

void loadInputUserBuffer(
    std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe,
    const std::string& fileLine)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);
    // treat each line as a space-separated list of input files
    std::vector<std::string> filePaths;
    split(filePaths, fileLine, ' ');
    // remove the line below
    for (int i = 0; i < filePaths.size(); i++) {std::cout << filePaths[i] << " ";} std::cout << "\n";
    if (inputNames.size()) std::cout << "Processing DNN Input: " << std::endl;
    for (size_t i = 0; i < inputNames.size(); i++) {
        const char* name = inputNames.at(i);
        std::string filePath(filePaths[i]);
        // print out which file is being processed
        std::cout << "\t" << i + 1 << ") " << filePath << std::endl;
        // load file content onto application storage buffer,
        // on top of which, Qualcomm (R) Neural Processing SDK has created a user buffer
        std::cout << "(name): " << name << "\n";
        std::cout << "filePath: " << filePath << "\n";
        std::string temp_name = std::string(name);
        assert(filePath.substr(0, filePath.size()-4) == temp_name.substr(0, temp_name.size()-2));
        applicationBuffers.at(name);
        std::cout << "going into load file in vector of size: " << applicationBuffers.at(name).size() << "\n";
        bool temp = loadByteDataFile(filePath, applicationBuffers.at(name));
    };
}

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



#endif
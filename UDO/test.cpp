#include <cstring>
#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdio>

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


/*
Goal: Create a CPU(fp32) run time script that runs the model in 2 parts, with the intermeidate data being
        stored in memory (not storage)
      - make it work with 2 inputs
      - make it work with udos
      - create a loop that runs a variable amount of time (to simulate runnign the model)
      - be sure that it is using an appropriate amount
      - Q: do we really need to load the input files twice? (in createUserBuffer())
      - measure the memory usage
      - try removing bulk from createUserBUffer (the applicatonBufferrs.emplace())
         -see if ubFactory.createUserBuffer is making a another buffer, or just pointing to the already existing one
      - be aware of copy contrructors with vectors and maps
Keep in mind, that there exists better versions of these functions!
*/

zdl::DlSystem::Runtime_t checkRuntime();
// std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath); // dont need
std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   const char* name,
                                                   std::vector<size_t> new_shape);
void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                      std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                      std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                      const char * name);
void loadInputUserBuffer(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         const std::string& fileLine);
void executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                    zdl::DlSystem::UserBufferMap& inputMap,
                    zdl::DlSystem::UserBufferMap& outputMap,
                    std::unordered_map<std::string,std::vector<uint8_t>>& applicationOutputBuffers,
                    const std::string& outputDir,
                    int num);
void SaveUserBuffer(const std::string& path, const std::vector<uint8_t>& buffer);
// bool loadUDOPackage(const std::string& UdoPackagePath);

// ITensor stuff
std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor (std::unique_ptr<zdl::SNPE::SNPE> & snpe , std::string& fileLine);
void executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                    std::unique_ptr<zdl::DlSystem::ITensor>& input,
                    std::string OutputDir,
                    int num);

void SaveITensor(const std::string& path, const zdl::DlSystem::ITensor* tensor);

// my own stuff
template<typename T>
bool MyloadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector);

struct ModelInfo {
    std::string model;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

// void parse_argv_models(const std::vector<ModelInfo>& models, const int& argc, const char* const argv[]);
void parse_argv_models(std::vector<ModelInfo>& models, int argc, char* argv[]);

// 1 per model
struct ModelRunetime {
   std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers;
   std::unordered_map<std::string, std::vector<uint8_t>> applicationOutputBuffers;
   zdl::DlSystem::UserBufferMap inputMap;
   zdl::DlSystem::UserBufferMap outputMap;
    
   std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> input_user_buff_vec;
   std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> output_user_buff_vec;

   // static zdl::DlSystem::Runtime_t runtime;
   zdl::DlSystem::Runtime_t runtime;
   std::unique_ptr<zdl::DlContainer::IDlContainer> container;
   std::unique_ptr<zdl::SNPE::SNPE> snpe;

};

// generate mask and position_ids; also iteration_num should first be 0
template <typename T>
void prepareInputs(T* mask, int* position_ids, uint32_t seq_len, uint32_t iteration_num);

// float
#define BATCH_SIZE 1
#define MAX_SEQ_LEN 2048
#define HIDDEN_SIZE 2560
#define QUERY_STATES_BUFF_SIZE  (BATCH_SIZE*MAX_SEQ_LEN*HIDDEN_SIZE)

#define INTERMEDIATE_SIZE 10240
#define INTERMEDIATE_STATES_BUFF_SIZE (BATCH_SIZE*MAX_SEQ_LEN*INTERMEDIATE_SIZE)

// partial_rotary_factor: 0.4
// head_dim = hidden_size // num_attention_heads i.e. (2560 / 32) = 80
// sin_cos(param:dim) = head_dim * partial_rotary_factor = 80 * .4 = 32
#define SIN_COS_DIM 32
#define SIN_COS_MAX_SEQ_LEN 2048 // a temporary solution
#define SIN_COS_BUFF_SIZE  (SIN_COS_DIM*SIN_COS_MAX_SEQ_LEN)

#define ATTN_WEIGHTS_SIZE (MAX_SEQ_LEN*MAX_SEQ_LEN*32)

#define LARGE_BUFFER_SIZE ATTN_WEIGHTS_SIZE
#define DATASIZE 2

#define DECODER_WEIGHT_SIZE 78671360
#define LM_WEIGHTS_SIZE 131128320

#define VOCAB_SIZE 51200

#define DECODERS 4


#define HIDDEN_STATES_SIZE  (4+((HIDDEN_SIZE*MAX_SEQ_LEN)/2))
#define MASK_SIZE ((4+MAX_SEQ_LEN*MAX_SEQ_LEN)*4)
#define POS_IDS_SIZE (4*(4+MAX_SEQ_LEN))
#define TOTAL_DECODER_WEIGHT_SIZE (DECODERS * DECODER_WEIGHT_SIZE * DATASIZE)
#define TOTAL_LM_WEIGHT_SIZE (4*65564160)
#define SIN_COS_TOTAL_SIZE (4*32768)

int main(int argc, char* argv[]) {
   
   std::vector<ModelInfo> models;
   parse_argv_models(models, argc, argv);
   std::cout << "Models, Inputs, Outputs\n";
   for (const auto& model : models) {
      std::cout << model.model << ",";
      for (const auto& input : model.inputs)    {std::cout << " " << input;}
      std::cout << ",";
      for (const auto& output : model.outputs)  {std::cout << " " << output;}
      std::cout << "\n";
   }  
   
   std::cout << "finished parsing\n";

   bool useUserSuppliedBuffers = true;
   int inputListNum = 1; // i guess?
   std::string OutputDir = ".";
   int num_models = models.size();
   std::vector<ModelRunetime> runtimes(num_models);

   // temporary
   int out_in_size =                (1 + 2 * DECODERS) * HIDDEN_STATES_SIZE * 4;
   std::vector<int> first_model_input_sizes = {
      out_in_size, MASK_SIZE, POS_IDS_SIZE, TOTAL_DECODER_WEIGHT_SIZE, TOTAL_LM_WEIGHT_SIZE, 
      SIN_COS_TOTAL_SIZE, SIN_COS_TOTAL_SIZE};
   // for (int i = 0; i < first_model_input_sizes.size(); i++) { first_model_input_sizes[i] *= DATA_SIZE; }
   // std::vector<int> second_model_input_sizes = {3 * HIDDEN_STATES_SIZE};
   // for (int i = 0; i < second_model_input_sizes.size(); i++) { second_model_input_sizes[i] *= DATA_SIZE; }
   // std::string i_name = ":0"; // varaible to reshape

   // std::string secondary_input_name = "vector";
   // for (int i = 1; i < num_models; i++) {
   //    models[i].inputs.push_back(secondary_input_name);
   // }

   // std::string udo_name = "DynTruncPackage/libs/x86-64_linux_clang/libUdoDynTruncPackageReg.so";
   std::string udo_name = "DecodePackage/libs/x86-64_linux_clang/libUdoDecodePackageReg.so";
   int udo_load = Snpe_Util_AddOpPackage(udo_name.c_str());
   std::cout << "udo_load: " << udo_load << "\n";
   assert(udo_load == 1);


   /* pos_ids and masking vars */
   uint32_t seq_len = 11;
   uint32_t iteration_num = 0;
   

   std::cout << "--intialization stage--\n";
   for (int i = 0; i < num_models; i++) {
      std::cout << "\n--i_stage #" << i << "\n";
      runtimes[i].applicationInputBuffers = std::unordered_map<std::string, std::vector<uint8_t>>();
      runtimes[i].applicationOutputBuffers = std::unordered_map<std::string, std::vector<uint8_t>>();
      runtimes[i].inputMap = zdl::DlSystem::UserBufferMap();
      runtimes[i].outputMap = zdl::DlSystem::UserBufferMap();
      runtimes[i].runtime = checkRuntime();
      // auto startTime = std::chrono::high_resolution_clock::now();
      runtimes[i].container = loadContainerFromFile(models[i].model);
      // auto part1EndTime = std::chrono::high_resolution_clock::now();
      // auto durationPart1 = std::chrono::duration_cast<std::chrono::milliseconds>(part1EndTime - startTime);
      // std::cout << "Time taken to load DLC: " << durationPart1.count() << " milliseconds" << std::endl;
      std::vector<size_t> new_dims = {1,1,1,12};
      // std::vector<size_t> new_dims = {1,2,3,2};
      // std::cout << (models[i].inputs[0]+":0") << "\n"; // remove later
      runtimes[i].snpe = setBuilderOptions(runtimes[i].container, runtimes[i].runtime, 
                                             useUserSuppliedBuffers, (models[i].inputs[0]+":0").c_str(), new_dims);
      // new_dims = {1,1,1,12};
      // startTime = std::chrono::high_resolution_clock::now();

      // part1EndTime = std::chrono::high_resolution_clock::now();
      // durationPart1 = std::chrono::duration_cast<std::chrono::milliseconds>(part1EndTime - startTime);
      // std::cout << "Time taken to build a second time: " << durationPart1.count() << " milliseconds" << std::endl;
      // custom code for dyn shape
      // auto input_info = runtimes[i].snpe->getInputOutputBufferAttributes((models[i].inputs[0] + ":0").c_str());
      // (*input_info)->getDims

       

      // end of changes

      // input user buffer
      std::cout << "creating input User Buffers\n\n";
      for (int j = 0; j < models[i].inputs.size(); j++) {
         std::vector<unsigned char> input_data_vec; // careful with copies
         // first model loads from storage
         if (i == 0) {
            std::cout << models[i].inputs[j] + ":0: " << first_model_input_sizes[j] << "\n";
            runtimes[i].applicationInputBuffers[models[i].inputs[j] + ":0"] 
               = std::vector<u_int8_t>(first_model_input_sizes[j]);
         }
         // every other model loads from memory
         else {
            // grab the output buffer of previous model and use as input
            // input_data_vec = runtimes[i-1].applicationOutputBuffers[models[i-1].outputs[0]];
            // will fill up later during execution stage
            assert(1 == 2); // TO MAKE THIS NOT RUN
            std::cout << "setting up secondary model buffers\n";
            // runtimes[i].applicationInputBuffers[models[i].inputs[j] + ":0"] = std::vector<u_int8_t>(second_model_input_sizes[j]); // for now use "vector:0"
            // for bert only: need to fill mask
            if (j == 1) {
               loadByteDataFile(models[i].inputs[j] + ".dat", runtimes[i].applicationInputBuffers[models[i].inputs[j] + ":0"]);
               }
         }
         std::cout << "calling createUserBuffer()\n";
         createUserBuffer(runtimes[i].inputMap, runtimes[i].applicationInputBuffers,
            runtimes[i].input_user_buff_vec, runtimes[i].snpe, (models[i].inputs[j] + ":0").c_str());
         std::cout << "finished\n";
      }
      // output user buffer (assume 1 output per model)
      runtimes[i].applicationOutputBuffers[models[i].outputs[0]] = std::vector<u_int8_t>(out_in_size);
      std::cout << "creating output User Buffer\n\n";
      createUserBuffer(runtimes[i].outputMap, runtimes[i].applicationOutputBuffers, 
         runtimes[i].output_user_buff_vec, runtimes[i].snpe, models[i].outputs[0].c_str());
      if (i == 0) {
         std::cout << "reading files for 1st model input\n";
         std::string fileLine;
         for (int j = 0; j < models[i].inputs.size(); j++) {fileLine += models[i].inputs[j] + ".dat ";}
         fileLine.pop_back(); // remove last space
         std::cout << "fileLine: " << fileLine << "\n";
         std::cout << "calling loadInputUserBuffer()\n";
         loadInputUserBuffer(runtimes[i].applicationInputBuffers, runtimes[i].snpe, fileLine);
      }
   }

   // applying masking and position_ids
   prepareInputs(
      (float*)runtimes[0].applicationInputBuffers["attention_mask:0"].data(),
      (int*)runtimes[0].applicationInputBuffers["position_ids_1:0"].data(),
      seq_len, iteration_num);


   // it seems that these vectors can be resized dynmically, and that SNPE goes by dlc input/output size rather than vector size
   // runtimes[0].applicationOutputBuffers[models[0].outputs[0]].resize(4000);

   // remove later, just testing to see if input buffers can be written to
   std::cout << "last element: " << 
      runtimes[0].applicationInputBuffers[models[0].inputs[4]+":0"][0] << "\n";


   // execution stage
   std::cout << "num_models: " << num_models << "\n";
   auto startTime = std::chrono::high_resolution_clock::now();
   std::cout << "\n--execution stage--\n";
   for (int i = 0; i < num_models; i++) {
      std::cout << "--e_stage #" << i << "\n";
      runtimes[i].snpe->execute(runtimes[i].inputMap, runtimes[i].outputMap);
      std::cout << "finished executing\n";
      if (i < num_models-1) {
         // runtimes[i+1].applicationInputBuffers[models[i+1].inputs[0]].size();
         // runtimes[i].applicationOutputBuffers[models[i].outputs[0]].size()
         int x1 = 5;
         std::cout << runtimes[i+1].applicationInputBuffers[models[i+1].inputs[0] + ":0"].size() << " vs "
            << runtimes[i].applicationOutputBuffers[models[i].outputs[0]].size() << "?\n";
         assert(runtimes[i+1].applicationInputBuffers[models[i+1].inputs[0]+ ":0"].size() == 
                  runtimes[i].applicationOutputBuffers[models[i].outputs[0]].size());
         // write output of current to input of next
         std::cout << "write output of current to input of next\n";
         runtimes[i+1].applicationInputBuffers[models[i+1].inputs[0] + ":0"] = 
            runtimes[i].applicationOutputBuffers[models[i].outputs[0]];
         std::cout << "finished\n";
         // write mask (for BERT only)
      }
      // last model
      else {
         std::cout << "second case\n";
         const zdl::DlSystem::StringList& outputBufferNames = runtimes[i].outputMap.getUserBufferNames();
         // Iterate through output buffers and print each output to a raw file
         std::for_each(outputBufferNames.begin(), outputBufferNames.end(), [&](const char* name)
         {
            std::ostringstream path;
            path << OutputDir << "/Result_" << 1 << "/" << models[i].outputs[0] << ".raw";
            SaveUserBuffer(path.str(), runtimes[i].applicationOutputBuffers.at(name));
         });
      }
   }
   auto part1EndTime = std::chrono::high_resolution_clock::now();
   auto durationPart1 = std::chrono::duration_cast<std::chrono::milliseconds>(part1EndTime - startTime);
   std::cout << "Time taken for " << DECODERS << " layers to run: " << durationPart1.count() << " milliseconds" << std::endl;
   



   // closing
   std::cout << "--closing stage--\n";
   for (int i = 0; i < num_models; i++) {
      zdl::SNPE::SNPEFactory::terminateLogging();
      runtimes[i].snpe.reset();
   }
}

void parse_argv_models(std::vector<ModelInfo>& models, int argc, char* argv[]) {
   std::string temp_str;
   for (int i = 1; i < argc; ++i) {
      if (argv[i][0] == '-') {
         // std::cout << argv[i] << "\n";
         // Identify the option type
         switch (argv[i][1]) {
            case 'm': {
               // New model
               models.emplace_back();
               models.back().model = argv[++i];
               continue;
            }
            case 'o':
               // Output node(s)
               temp_str = argv[++i];
               models.back().outputs.push_back(temp_str + ":0");
               continue;
            case 'i':
               // Input(s)
               temp_str = argv[++i];
               temp_str = temp_str.substr(0, temp_str.find(".dat"));
               models.back().inputs.push_back(temp_str); // ex: "input1.dat" --> "input1"
               continue;
            default:
               std::cerr << "Unknown option: " << argv[i] << std::endl;
               std::cout << "ERROR_1!\n";
         }
      }  
      else {
         std::cerr << "Invalid argument: " << argv[i] << std::endl;
         std::cout << "ERROR_2!\n";
      }
   }
}

// bool loadUDOPackage(const std::string& UdoPackagePath)
// {
//     std::vector<std::string> udoPkgPathsList;
//     split(udoPkgPathsList, UdoPackagePath, ',');
//     for (const auto &u : udoPkgPathsList)
//     {
//        if (false == zdl::SNPE::SNPEFactory::addOpPackage(u))
//        {
//           std::cerr << "Error while loading UDO package: "<< u << std::endl;
//           return false;
//        }
//     }
//     return true;
// }


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

// dont need as defined in LoadContainer.cpp
// std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath)
// {
//     std::unique_ptr<zdl::DlContainer::IDlContainer> container;
//     container = zdl::DlContainer::IDlContainer::open(containerPath);
//     return container;
// }

// for now, no udos **comeback

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   const char* name,
                                                   std::vector<size_t> new_shape)
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

void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
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

// remove
// void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
//                       std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
//                       std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
//                       std::unique_ptr<zdl::SNPE::SNPE>& snpe,
//                       const char * name)
// {
//    // get attributes of buffer by name
//    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
//    if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);
//    // calculate the size of buffer required by the input tensor
//    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
//    // Calculate the stride based on buffer strides, assuming tightly packed.
//    // Note: Strides = Number of bytes to advance to the next element in each dimension.
//    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
//    // Note: Buffer stride is usually known and does not need to be calculated.
//    std::vector<size_t> strides(bufferShape.rank());
//    strides[strides.size() - 1] = sizeof(float);
//    size_t stride = strides[strides.size() - 1];
//    for (size_t i = bufferShape.rank() - 1; i > 0; i--)
//    {
//       stride *= bufferShape[i];
//       strides[i-1] = stride;
//    }
//    const size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
//    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);
//    // set the buffer encoding type
//    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
//    // create user-backed storage to load input data onto it
//    applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize));
//    // create Qualcomm (R) Neural Processing SDK user buffer from the user-backed buffer
//    zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
//    snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(applicationBuffers.at(name).data(),
//                                                               bufSize,
//                                                               strides,
//                                                               &userBufferEncodingFloat));
//    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
//    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
// }


void loadInputUserBuffer(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
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

template<typename T>
bool MyloadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector)
{
   std::cout << "case 1\n";
   std::ifstream in(inputFile, std::ifstream::binary);
   if (!in.is_open() || !in.good())
   {
      std::cerr << "Failed to open input file: " << inputFile << "\n";
   }
   std::cout << "case 2\n";
   in.seekg(0, in.end);
   size_t length = in.tellg();
   in.seekg(0, in.beg);
   std::cout << "length: " << length << "\n";
   if (length % sizeof(T) != 0) {
      std::cerr << "Size of input file should be divisible by sizeof(dtype).\n";
      return false;
   }

   if (loadVector.size() == 0) {
      loadVector.resize(length / sizeof(T));
   } else if (loadVector.size() < length / sizeof(T)) {
      std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
      loadVector.resize(length / sizeof(T));
   }

   if (!in.read(reinterpret_cast<char*>(&loadVector[0]), length))
   {
      std::cerr << "Failed to read the contents of: " << inputFile << "\n";
   }
   return true;
}


void executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
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

std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor (std::unique_ptr<zdl::SNPE::SNPE> & snpe , std::string& fileLine)
{
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
    // Make sure the network requires only a single input
    assert (strList.size() == 1);
    // If the network has a single input, each line represents the input file to be loaded for that input
    std::string filePath(fileLine);
    std::cout << "Processing DNN Input: " << filePath << "\n";
    std::vector<float> inputVec = loadFloatDataFile(filePath);
    /* Create an input tensor that is correctly sized to hold the input of the network. Dimensions that have no fixed size will be represented with a value of 0. */
    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;
    /* Calculate the total number of elements that can be stored in the tensor so that we can check that the input contains the expected number of elements.
       With the input dimensions computed create a tensor to convey the input into the network. */
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    /* Copy the loaded input file contents into the networks input tensor.SNPE's ITensor supports C++ STL functions like std::copy() */
    std::copy(inputVec.begin(), inputVec.end(), input->begin());
    return input;
}

void executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                    std::unique_ptr<zdl::DlSystem::ITensor>& input,
                    std::string OutputDir,
                    int num)
{
    //Execute the network and store the outputs that were specified when creating the network in a TensorMap
    static zdl::DlSystem::TensorMap outputTensorMap;
    std::cout << "calling execute" << std::endl;
    snpe->execute(input.get(), outputTensorMap);
    std::cout << "exited execute" << std::endl;
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
    std::cout << "number of tensor names:" << tensorNames.size() << std::endl;
    //Iterate through the output Tensor map, and print each output layer name
    std::for_each( tensorNames.begin(), tensorNames.end(), [&](const char* name)
    {
        std::ostringstream path;
        path << OutputDir << "/"
        << "Result_" << num << "/"
        << name << ".raw";
        auto tensorPtr = outputTensorMap.getTensor(name);
        SaveITensor(path.str(), tensorPtr);
    });
}
// The following is a partial snippet of the function
void SaveITensor(const std::string& path, const zdl::DlSystem::ITensor* tensor)
{
   
   std::ofstream os(path, std::ofstream::binary);
   if (!os)
   {
      std::cerr << "Failed to open output file for writing: " << path << "\n";
      std::exit(EXIT_FAILURE);
   }
   for ( auto it = tensor->cbegin(); it != tensor->cend(); ++it )
   {
      float f = *it;
      if (!os.write(reinterpret_cast<char*>(&f), sizeof(float)))
      {
         std::cerr << "Failed to write data to: " << path << "\n";
         std::exit(EXIT_FAILURE);
      }
   }
}


// generate mask and position_ids; also iteration_num should first be 0
template <typename T>
void prepareInputs(
    T* mask, 
    int* position_ids,
    uint32_t seq_len, uint32_t iteration_num
    ) 
{
    // must adjust indexing if this is not true
    assert(sizeof(T) == 4);
    if (iteration_num == 0) {
        // set position_ids shape
        position_ids[MAX_SEQ_LEN + 0] = 1;
        position_ids[MAX_SEQ_LEN + 1] = 1;
        position_ids[MAX_SEQ_LEN + 2] = 1;
        position_ids[MAX_SEQ_LEN + 3] = seq_len;
        // set position_ids
        for (int i = 0; i < seq_len; ++i) { position_ids[i] = i; }
        // set mask shape
        uint32_t* ptr32 = (uint32_t*)mask;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 0] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 1] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 2] = seq_len;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 3] = seq_len;
        // set mask
        T lowest = std::numeric_limits<T>::lowest();
        for (uint32_t row = 0; row < seq_len; row++) {
            for (uint32_t col = 0; col < seq_len; col++) {
                if (row >= col) { mask[row*seq_len + col] = 0; }
                else            { mask[row*seq_len + col] = lowest; } 
            }
        }
    }
    else {
        // set position_ids shape
        position_ids[MAX_SEQ_LEN + 0] = 1;
        position_ids[MAX_SEQ_LEN + 1] = 1;
        position_ids[MAX_SEQ_LEN + 2] = 1;
        position_ids[MAX_SEQ_LEN + 3] = 1;
        // set position_ids
        position_ids[0] = seq_len-1;
        // set mask shape
        uint32_t* ptr32 = (uint32_t*)mask;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 0] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 1] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 2] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 3] = seq_len;
        // set mask
        for (uint32_t i = 0; i < seq_len; i++) { mask[i] = 0; }
    }
}

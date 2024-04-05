#include <jni.h>
#include <string>
#include <fstream>
#include <vector>

#include "test.h"
#include "test2.h"
#include "test3.h"

//#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>

#include "android_main.h"

int writeTextToDirectory(const std::string& text, const std::string& directory, const std::string& filename) {
    // Construct the full path to the file
    std::string filePath = directory + "/" + filename;
    // Open the file
    std::ofstream outFile(filePath);
    if (outFile.is_open()) {
        // Write the text to the file
        outFile << text;
        // Close the file
        outFile.close();
        return 2;
    }
    else { return 3; }
}

//x = writeTextToDirectory("this is a file!", "data/data/com.example.nativecr/files", "file1.txt");
//x = writeTextToDirectory("this is a file!", "/sdcard/Download", "file1.txt");

bool directoryExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
        return false;
    else if (info.st_mode & S_IFDIR)
        return true;
    else
        return false;
}

bool fileExists(const char *filename) {
    // Check if the file exists and can be accessed for reading
    if (access(filename, F_OK) != -1) {
        return true;
    } else {
        return false;
    }
}

// move files over from downloads
bool copyFile(const std::string& inputFile, const std::string& outputFile) {
    std::ifstream sourceFile(inputFile, std::ios::binary);
    if (!sourceFile) {
        return false;
    }
    std::ofstream destinationFile(outputFile, std::ios::binary);
    if (!destinationFile) {
        return false;
    }
    destinationFile << sourceFile.rdbuf();
    if (destinationFile.bad()) {
        return false;
    }
    return true;
}

int copyFile_mod(const std::string& inputFile, const std::string& outputFile) {
    std::ifstream sourceFile(inputFile, std::ios::binary);
    if (!sourceFile) {
        return 1;
    }
    std::ofstream destinationFile(outputFile, std::ios::binary);
    if (!destinationFile) {
        return 2;
    }
    destinationFile << sourceFile.rdbuf();
    if (destinationFile.bad()) {
        return 3;
    }
    return 0;
}

bool copyFiles(const std::string& src, const std::string& dst, const std::vector<std::string>& fileList) {
    std::string srcFile, dstFile;
    bool success = true;
    for (auto fileName: fileList) {
        srcFile  =  src + "/" + fileName;
        dstFile  =  dst + "/" + fileName;
        success = copyFile(srcFile, dstFile);
        if (!success) { return false; }
    }
    return true;
}

// run once
int setupFileSpace(
        std::string src,
        std::string dst,
        std::vector<std::string> inputList,
        std::string dlc)
        {
    // copy input files and weights
    if (!copyFiles(src, dst, inputList)) {
        return 1;
    }
    // copy dlc
    if (!copyFile(src + "/" + dlc, dst + "/" + dlc)) {
        return 2;
    }
    return 0;
}

int setupLibraryFileSpace(
        std::string src,
        std::string dst,
        std::vector<std::string> inputList
        )
{
    if (!copyFiles(src, dst, inputList)) {
        return 1;
    }
    return 0;
}

// this is a test
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_nativecr_MainActivity_test1( JNIEnv* env,jobject)
{
    // app configuration
    bool importFiles = false;
    std::string name_of_app = "nativecr";
    std::string working_dir = "/data/data/com.example." + name_of_app;

    // model inputs
    std::string input_txt = "this test does not matter";
    std::string srcDIR = "/sdcard/Download";
    std::vector<std::string> inputList = {
            "hidden_states_and_kv.dat",
            "attention_mask.dat",
            "position_ids_1.dat",
            "decoder_weights_1.dat",
            "lm_head_weights_1.dat",
            "sin.dat",
            "cos.dat"
    };
    std::vector<size_t> first_model_input_sizes = {
            (1 + 2 * DECODERS) * (4*4 + (MAX_SEQ_LEN * HIDDEN_SIZE) * DATASIZE),
            MASK_SIZE,
            POS_IDS_SIZE,
            TOTAL_DECODER_WEIGHT_SIZE,
            TOTAL_LM_WEIGHT_SIZE,
            SIN_COS_TOTAL_SIZE,
            SIN_COS_TOTAL_SIZE
    };
    std::string embeddingFile = "embed_tokens.dat";
    std::string dlcName = "UnifiedPhiDecodersAndLogits.dlc";
    std::vector<std::string> outputNames = {"Output_1:0"};
    uint32_t NUM_ITERS = 1;
    bool use_udo = true;
    std::string udo_path = working_dir + "/libUdoDecodePackageReg.so";
    bool firstRun = true;
    auto free_status = Free_Status::run_and_free;

    // model output
    std::string output_str;

    // file system setup
    std::vector<std::string> udo_library_list = {
            "libUdoDecodePackageReg.so",
            "libUdoDecodePackageImplCpu.so"
    };
    if (importFiles) {
        if (!copyFiles(srcDIR, working_dir, std::vector<std::string>{dlcName})) {
            output_str = "Failed to copy " + dlcName + "from " + srcDIR + "to " + working_dir;
            return env->NewStringUTF(output_str.c_str());
        }
        if (!copyFiles(srcDIR, working_dir, udo_library_list)) {
            output_str = "Failed to copy udo_library_list from " + srcDIR + "to " + working_dir;
            return env->NewStringUTF(output_str.c_str());
        }
    }

    // remove ".dat" from inputList
    for (std::string& filename : inputList) {
        size_t lastDot = filename.find_last_of(".");
        if (lastDot == std::string::npos) { lastDot = filename.size(); }
        filename = filename.substr(0, lastDot);
    }

    // execution stage
    output_str = "unintialized";
    int debugReturnCode = 20;
    output_str = modelLaunch(
        input_txt,
        srcDIR,
        inputList,
        first_model_input_sizes,
        embeddingFile,
        working_dir + "/" + dlcName,
        outputNames,
        NUM_ITERS,
        udo_path,
        use_udo,
        firstRun,
        free_status,
        debugReturnCode);

    return env->NewStringUTF(output_str.c_str());
}

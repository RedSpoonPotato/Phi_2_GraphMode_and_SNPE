# Copyright (c) 2017-2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

LOCAL_PATH := $(call my-dir)

$(info $(LOCAL_PATH))

BOOST_PATH := /home/kernal1/QM_Sandbox/boost_build/boost_new/boost_1_85_0/android-build/arm64-v8a/lib

ifeq ($(TARGET_ARCH_ABI), arm64-v8a)
   ifeq ($(APP_STL), c++_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/aarch64-android
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else
   $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

SNPE_INCLUDE_DIR := $(SNPE_ROOT)/include/SNPE
BOOST_INCLUDE_DIR := /home/kernal1/QM_Sandbox/boost_build/boost_new/boost_1_85_0/android-build/include
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include

include $(CLEAR_VARS)
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include
LOCAL_MODULE := libmain
LOCAL_SRC_FILES := android_main.cpp CheckRuntime.cpp LoadContainer.cpp LoadUDOPackage.cpp LoadInputTensor.cpp SetBuilderOptions.cpp Util.cpp NV21Load.cpp CreateUserBuffer.cpp PreprocessInput.cpp SaveOutputTensor.cpp CreateGLBuffer.cpp CreateGLContext.cpp
LOCAL_EXPORT_C_INCLUDES += $(LOCAL_PATH)/include
LOCAL_CFLAGS := -DENABLE_GL_BUFFER
LOCAL_SHARED_LIBRARIES := libSNPE libboost_regex libboost_system libboost_filesystem libboost_atomic
LOCAL_LDLIBS     := -lGLESv2 -lEGL -llog
include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libSNPE
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libSNPE.so
LOCAL_EXPORT_C_INCLUDES += $(SNPE_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libboost_regex
LOCAL_SRC_FILES := $(BOOST_PATH)/libboost_regex.so
LOCAL_EXPORT_C_INCLUDES += $(BOOST_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libboost_system
LOCAL_SRC_FILES := $(BOOST_PATH)/libboost_system.so
LOCAL_EXPORT_C_INCLUDES += $(BOOST_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libboost_filesystem
LOCAL_SRC_FILES := $(BOOST_PATH)/libboost_filesystem.so
LOCAL_EXPORT_C_INCLUDES += $(BOOST_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libboost_atomic
LOCAL_SRC_FILES := $(BOOST_PATH)/libboost_atomic.so
LOCAL_EXPORT_C_INCLUDES += $(BOOST_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := load_test
LOCAL_SRC_FILES := $(LOCAL_MODULE).cpp
LOCAL_EXPORT_C_INCLUDES += $(LOCAL_PATH)/include
LOCAL_CFLAGS := -DENABLE_GL_BUFFER
LOCAL_SHARED_LIBRARIES := libmain libSNPE libboost_regex libboost_system libboost_filesystem libboost_atomic
LOCAL_LDLIBS     := -lGLESv2 -lEGL -llog
include $(BUILD_EXECUTABLE)
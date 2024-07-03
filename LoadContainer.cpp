//==============================================================================
//
//  Copyright (c) 2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <string>

#include "LoadContainer.hpp"

#include "DlContainer/IDlContainer.hpp"

std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    std::cout << "container ptr before: " << container.get() << "\n";
    container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(containerPath.c_str()));
    std::cout << "container ptr after: " << container.get() << "\n";
    return container;
}

std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromVector(std::vector<uint8_t>& vec)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    std::cout << "container ptr before: " << container.get() << "\n";
    container = zdl::DlContainer::IDlContainer::open(vec);
    std::cout << "container ptr after: " << container.get() << "\n";
    return container;
}
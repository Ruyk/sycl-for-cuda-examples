/**
 * SYCL FOR CUDA : Vector Addition Example
 *
 * Copyright 2020 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 * @File: vector_addition.cpp
 */

#include <algorithm>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

class CUDASelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &Device) const override {
    using namespace sycl::info;

    const std::string DriverVersion = Device.get_info<device::driver_version>();

    if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
      std::cout << " CUDA device found " << std::endl;
      return 1;
    };
    return -1;
  }
};


void vecAdd(double *devA, double *devB, double *devC, sycl::id<1> i) {


  
  devC[i] = devA[i] + devB[i];
}


int main(int argc, char *argv[]) {
  constexpr const size_t N = 100000;

  sycl::queue myQueue{CUDASelector()};

  auto* devA = sycl::malloc_shared<double>(N, myQueue);
  auto* devB = sycl::malloc_shared<double>(N, myQueue);
  auto* devC = sycl::malloc_shared<double>(N, myQueue);

  // Initialize input data
  {
    for (int i = 0; i < N; i++) {
      devA[i] = sin(i) * sin(i);
      devB[i] = cos(i) * cos(i);
    }
  }

  // Command Group creation
  auto cg = [&](sycl::handler &h) {
    h.parallel_for({N}, [=](auto i) {
          vecAdd(devA, devB, devC, i);
        });
  };

  myQueue.submit(cg);
  myQueue.wait();

  {
    double sum = 0.0f;
    for (int i = 0; i < N; i++) {
      sum += devC[i];
    }
    std::cout << "Sum is : " << sum << std::endl;
  }

  sycl::free(devA, myQueue);
  sycl::free(devB, myQueue);
  sycl::free(devC, myQueue);

  return 0;
}

// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

void Sort3AlphaDev(int* buffer) {
  asm volatile(
      "mov 0x4(%0), %%eax            \n"
      "mov 0x8(%0), %%ecx            \n"
      "cmp %%eax, %%ecx              \n"
      "mov %%eax, %%edx              \n"
      "cmovl %%ecx, %%edx            \n"
      "mov (%0), %%r8d               \n"
      "cmovg %%ecx, %%eax            \n"
      "cmp %%r8d, %%eax              \n"
      "mov %%r8d, %%ecx              \n"
      "cmovl %%eax, %%ecx            \n"
      "cmovle %%r8d, %%eax           \n"
      "mov %%eax, 0x8(%0)            \n"
      "cmp %%ecx, %%edx              \n"
      "cmovle %%edx, %%r8d           \n"
      "mov %%r8d, (%0)               \n"
      "cmovg %%edx, %%ecx            \n"
      "mov %%ecx, 0x4(%0)            \n"
      : "+r"(buffer)
      :
      : "eax", "ecx", "edx", "r8d", "memory");
}

void Sort4AlphaDev(int* buffer) {
  asm volatile(
      "mov 0x8(%0), %%eax            \n"
      "mov (%0), %%ecx               \n"
      "mov 0x4(%0), %%edx            \n"
      "cmp %%eax, %%ecx              \n"
      "mov %%eax, %%r8d              \n"
      "cmovl %%ecx, %%r8d            \n"
      "cmovl %%eax, %%ecx            \n"
      "mov 0xc(%0), %%r9d            \n"
      "cmp %%r9d, %%edx              \n"
      "mov %%r9d, %%eax              \n"
      "cmovl %%edx, %%eax            \n"
      "cmovl %%r9d, %%edx            \n"
      "cmp %%eax, %%r8d              \n"
      "mov %%eax, %%r9d              \n"
      "cmovl %%r8d, %%r9d            \n"
      "cmovge %%r8d, %%eax           \n"
      "mov %%r9d, (%0)               \n"
      "cmp %%edx, %%ecx              \n"
      "mov %%edx, %%r8d              \n"
      "cmovl %%ecx, %%r8d            \n"
      "cmovge %%ecx, %%edx           \n"
      "mov %%edx, 0xc(%0)            \n"
      "cmp %%r8d, %%eax              \n"
      "mov %%r8d, %%ecx              \n"
      "cmovl %%eax, %%ecx            \n"
      "cmovge %%eax, %%r8d           \n"
      "mov %%r8d, 0x8(%0)            \n"
      "mov %%ecx, 0x4(%0)            \n"
      : "+r"(buffer)
      :
      : "eax", "ecx", "edx", "r8d", "r9d", "memory");
}

void Sort5AlphaDev(int* buffer) {
  asm volatile(
      "mov (%0), %%eax               \n"
      "mov 0x4(%0), %%ecx            \n"
      "cmp %%eax, %%ecx              \n"
      "mov %%eax, %%edx              \n"
      "cmovl %%ecx, %%edx            \n"
      "cmovg %%ecx, %%eax            \n"
      "mov 0xc(%0), %%r8d            \n"
      "mov 0x10(%0), %%ecx           \n"
      "cmp %%r8d, %%ecx              \n"
      "mov %%r8d, %%r9d              \n"
      "cmovl %%ecx, %%r9d            \n"
      "cmovg %%ecx, %%r8d            \n"
      "mov 0x8(%0), %%r10d           \n"
      "cmp %%r10d, %%r8d             \n"
      "mov %%r10d, %%ecx             \n"
      "cmovl %%r8d, %%ecx            \n"
      "cmovle %%r10d, %%r8d          \n"
      "cmp %%ecx, %%r9d              \n"
      "cmovle %%r9d, %%r10d          \n"
      "cmovg %%r9d, %%ecx            \n"
      "cmp %%eax, %%r8d              \n"
      "mov %%eax, %%r9d              \n"
      "cmovl %%r8d, %%r9d            \n"
      "cmovle %%eax, %%r8d           \n"
      "cmp %%edx, %%ecx              \n"
      "mov %%edx, %%eax              \n"
      "cmovl %%ecx, %%eax            \n"
      "cmovle %%edx, %%ecx           \n"
      "mov %%r8d, 0x10(%0)           \n"
      "cmp %%eax, %%r10d             \n"
      "cmovle %%r10d, %%edx          \n"
      "mov %%edx, (%0)               \n"
      "cmovg %%r10d, %%eax           \n"
      "cmp %%r9d, %%ecx              \n"
      "mov %%r9d, %%r8d              \n"
      "cmovl %%ecx, %%r8d            \n"
      "cmovle %%r9d, %%ecx           \n"
      "mov %%ecx, 0xc(%0)            \n"
      "cmp %%r8d, %%eax              \n"
      "cmovle %%eax, %%r9d           \n"
      "mov %%r9d, 0x4(%0)            \n"
      "cmovg %%eax, %%r8d            \n"
      "mov %%r8d, 0x8(%0)            \n"
      : "+r"(buffer)
      :
      : "eax", "ecx", "edx", "r8d", "r9d", "r10d", "memory");
}

void Sort6AlphaDev(int* buffer) {
  asm volatile(
      "mov 0x14(%0), %%r9d           \n"
      "mov 0xc(%0), %%r10d           \n"
      "mov 0x8(%0), %%ecx            \n"
      "mov 0x10(%0), %%eax           \n"
      "mov %%ecx, %%edx              \n"
      "mov (%0), %%ebx               \n"
      "mov 0x14(%0), %%r8d           \n"
      "mov (%0), %%r11d              \n"
      "cmp %%eax, %%r8d              \n"
      "cmovg %%eax, %%r9d            \n"
      "cmovl %%eax, %%r8d            \n"
      "mov 0x4(%0), %%eax            \n"
      "cmp %%eax, %%edx              \n"
      "cmovg %%eax, %%edx            \n"
      "cmovle %%eax, %%ecx           \n"
      "cmp %%r11d, %%ecx             \n"
      "cmovle %%ecx, %%r11d          \n"
      "mov 0xc(%0), %%eax            \n"
      "cmovle %%ebx, %%ecx           \n"
      "cmp %%r10d, %%r8d             \n"
      "cmovle %%r8d, %%r10d          \n"
      "cmovl %%eax, %%r8d            \n"
      "cmp %%eax, %%r9d              \n"
      "cmovge %%r9d, %%r10d          \n"
      "cmovl %%r9d, %%eax            \n"
      "cmp %%ebx, %%edx              \n"
      "cmovle %%edx, %%ebx           \n"
      "cmovge %%edx, %%r11d          \n"
      "cmp %%ecx, %%r8d              \n"
      "mov %%ecx, %%edx              \n"
      "cmovl %%r8d, %%edx            \n"
      "cmovle %%ecx, %%r8d           \n"
      "mov %%ebx, %%ecx              \n"
      "cmp %%ecx, %%eax              \n"
      "cmovge %%eax, %%ebx           \n"
      "cmovle %%eax, %%ecx           \n"
      "mov %%ebx, %%eax              \n"
      "cmp %%r11d, %%r10d            \n"
      "mov %%r8d, 0x14(%0)           \n"
      "mov %%r11d, %%r8d             \n"
      "cmovle %%r10d, %%r8d          \n"
      "cmovle %%r11d, %%r10d         \n"
      "cmp %%r8d, %%ebx              \n"
      "cmovl %%r8d, %%eax            \n"
      "mov %%ecx, (%0)               \n"
      "cmovg %%r8d, %%ebx            \n"
      "cmp %%edx, %%eax              \n"
      "mov %%ebx, 0x4(%0)            \n"
      "mov %%edx, %%ecx              \n"
      "cmovg %%eax, %%ecx            \n"
      "cmovge %%edx, %%eax           \n"
      "cmp %%edx, %%r10d             \n"
      "cmovl %%r10d, %%ecx           \n"
      "cmovl %%edx, %%r10d           \n"
      "mov %%ecx, 0xc(%0)            \n"
      "mov %%r10d, 0x10(%0)          \n"
      "mov %%eax, 0x8(%0)            \n"
      : "+r"(buffer)
      :
      : "rbx", "eax", "ecx", "edx", "r8d", "r9d", "r10d", "r11d", "memory");
}

void Sort7AlphaDev(int* buffer) {
  asm volatile(
      "mov (%0), %%r12d              \n"
      "mov 0xc(%0), %%r9d            \n"
      "mov 0x10(%0), %%r8d           \n"
      "mov 0x14(%0), %%r10d          \n"
      "mov 0x18(%0), %%r11d          \n"
      "mov 0x8(%0), %%ecx            \n"
      "mov %%r12d, %%ebx             \n"
      "mov 0x4(%0), %%edx            \n"
      "mov %%r9d, %%eax              \n"
      "cmp %%eax, %%r8d              \n"
      "cmovl %%r8d, %%r9d            \n"
      "cmovl %%eax, %%r8d            \n"
      "mov 0x4(%0), %%eax            \n"
      "cmp %%edx, %%ecx              \n"
      "cmovl %%ecx, %%edx            \n"
      "cmovle %%eax, %%ecx           \n"
      "cmp %%r12d, %%ecx             \n"
      "cmovle %%ecx, %%r12d          \n"
      "mov 0x14(%0), %%eax           \n"
      "cmovle %%ebx, %%ecx           \n"
      "mov %%ecx, %%r13d             \n"
      "cmp %%eax, %%r11d             \n"
      "cmovl %%r11d, %%eax           \n"
      "mov %%r9d, %%ebx              \n"
      "cmovle %%r10d, %%r11d         \n"
      "cmp %%ebx, %%eax              \n"
      "cmovl %%eax, %%ebx            \n"
      "cmovle %%r9d, %%eax           \n"
      "cmp %%r8d, %%r11d             \n"
      "mov %%r8d, %%r10d             \n"
      "cmovl %%r11d, %%r10d          \n"
      "cmovle %%r8d, %%r11d          \n"
      "cmp %%r13d, %%r11d            \n"
      "mov %%r12d, %%r8d             \n"
      "cmovle %%r11d, %%r13d         \n"
      "cmovle %%ecx, %%r11d          \n"
      "mov %%edx, %%r9d              \n"
      "mov %%r11d, 0x18(%0)          \n"
      "mov %%r13d, %%ecx             \n"
      "cmp %%r9d, %%eax              \n"
      "cmovle %%eax, %%r9d           \n"
      "cmovl %%edx, %%eax            \n"
      "cmp %%r8d, %%r10d             \n"
      "cmovl %%r10d, %%r8d           \n"
      "cmovl %%r12d, %%r10d          \n"
      "cmp %%ecx, %%eax              \n"
      "cmovle %%eax, %%ecx           \n"
      "cmovle %%r13d, %%eax          \n"
      "cmp %%r12d, %%ebx             \n"
      "mov %%ecx, %%edx              \n"
      "cmovg %%ebx, %%r8d            \n"
      "cmovle %%ebx, %%r12d          \n"
      "mov %%r9d, %%r14d             \n"
      "cmp %%r12d, %%r9d             \n"
      "cmovl %%r12d, %%r14d          \n"
      "cmovl %%r9d, %%r12d           \n"
      "cmp %%ecx, %%r10d             \n"
      "cmovl %%r10d, %%edx           \n"
      "cmovge %%r10d, %%ecx          \n"
      "mov %%edx, %%r11d             \n"
      "cmp %%r9d, %%r8d              \n"
      "cmovl %%r8d, %%r14d           \n"
      "mov %%r14d, 0x4(%0)           \n"
      "mov %%r12d, (%0)              \n"
      "cmovl %%r9d, %%r8d            \n"
      "cmp %%edx, %%r8d              \n"
      "cmovl %%r8d, %%r11d           \n"
      "cmovge %%r8d, %%edx           \n"
      "mov %%r11d, 0x8(%0)           \n"
      "cmp %%ecx, %%eax              \n"
      "mov %%edx, 0xc(%0)            \n"
      "mov %%ecx, %%r11d             \n"
      "cmovge %%eax, %%ecx           \n"
      "mov %%ecx, 0x14(%0)           \n"
      "cmovle %%eax, %%r11d          \n"
      "mov %%r11d, 0x10(%0)          \n"
      : "+r"(buffer)
      :
      : "rbx", "r12", "r13", "r14", "eax", "ecx", "edx", "r8d", "r9d", "r10d",
        "r11d", "r12d", "r13d", "r14d", "memory");
}

void Sort8AlphaDev(int* buffer) {
  asm volatile(
      "mov 0x8(%0), %%r8d            \n"
      "mov 0x4(%0), %%ecx            \n"
      "mov 0x18(%0), %%ebx           \n"
      "mov 0x10(%0), %%r11d          \n"
      "mov 0x14(%0), %%r10d          \n"
      "mov 0x18(%0), %%r12d          \n"
      "mov 0xc(%0), %%r9d            \n"
      "mov (%0), %%edx               \n"
      "mov %%edx, %%eax              \n"
      "cmp %%eax, %%ecx              \n"
      "cmovl %%ecx, %%edx            \n"
      "cmovle %%eax, %%ecx           \n"
      "mov %%r8d, %%eax              \n"
      "cmp %%r8d, %%r9d              \n"
      "mov %%edx, %%r13d             \n"
      "cmovle %%r9d, %%eax           \n"
      "cmovl %%r8d, %%r9d            \n"
      "cmp %%r11d, %%r10d            \n"
      "mov 0x10(%0), %%r8d           \n"
      "cmovl %%r10d, %%r11d          \n"
      "cmovl %%r8d, %%r10d           \n"
      "mov 0x1c(%0), %%r8d           \n"
      "cmp %%r13d, %%eax             \n"
      "cmovle %%eax, %%r13d          \n"
      "mov %%r13d, %%r14d            \n"
      "cmovl %%edx, %%eax            \n"
      "cmp %%ebx, %%r8d              \n"
      "cmovle %%r8d, %%r12d          \n"
      "mov %%r11d, %%edx             \n"
      "cmovle %%ebx, %%r8d           \n"
      "cmp %%edx, %%r12d             \n"
      "mov %%ecx, %%ebx              \n"
      "cmovl %%r12d, %%edx           \n"
      "cmovl %%r11d, %%r12d          \n"
      "cmp %%ecx, %%r9d              \n"
      "cmovl %%r9d, %%ebx            \n"
      "cmovle %%ecx, %%r9d           \n"
      "cmp %%ebx, %%eax              \n"
      "mov %%ebx, %%r11d             \n"
      "cmovge %%eax, %%ebx           \n"
      "cmovl %%eax, %%r11d           \n"
      "mov %%r11d, %%eax             \n"
      "cmp %%r13d, %%edx             \n"
      "mov %%r10d, %%ecx             \n"
      "cmovl %%edx, %%r14d           \n"
      "cmovl %%r13d, %%edx           \n"
      "cmp %%r10d, %%r8d             \n"
      "cmovle %%r8d, %%ecx           \n"
      "cmovle %%r10d, %%r8d          \n"
      "cmovl %%ecx, %%r10d           \n"
      "cmp %%ecx, %%r12d             \n"
      "cmovl %%r12d, %%r10d          \n"
      "mov %%r14d, (%0)              \n"
      "cmovge %%r12d, %%ecx          \n"
      "cmp %%r11d, %%r10d            \n"
      "cmovl %%r10d, %%eax           \n"
      "cmovle %%r11d, %%r10d         \n"
      "mov %%r9d, %%r12d             \n"
      "cmp %%r9d, %%r8d              \n"
      "cmovl %%r8d, %%r12d           \n"
      "cmovl %%r9d, %%r8d            \n"
      "mov %%ebx, %%r11d             \n"
      "mov %%r8d, 0x1c(%0)           \n"
      "cmp %%r11d, %%ecx             \n"
      "cmovl %%ecx, %%r11d           \n"
      "mov %%eax, %%r8d              \n"
      "cmovl %%ebx, %%ecx            \n"
      "cmp %%r8d, %%edx              \n"
      "cmovg %%edx, %%eax            \n"
      "mov %%r11d, %%ebx             \n"
      "cmovle %%edx, %%r8d           \n"
      "cmp %%r12d, %%ecx             \n"
      "mov %%r8d, 0x4(%0)            \n"
      "mov %%r12d, %%r8d             \n"
      "cmovl %%ecx, %%r8d            \n"
      "cmovle %%r12d, %%ecx          \n"
      "cmp %%r11d, %%eax             \n"
      "cmovle %%eax, %%ebx           \n"
      "cmovle %%r11d, %%eax          \n"
      "cmp %%r12d, %%r10d            \n"
      "mov %%ebx, 0x8(%0)            \n"
      "cmovge %%r10d, %%r8d          \n"
      "cmovle %%r10d, %%r12d         \n"
      "mov %%r12d, %%ebx             \n"
      "cmp %%r12d, %%eax             \n"
      "mov %%ecx, 0x18(%0)           \n"
      "cmovg %%eax, %%ebx            \n"
      "mov %%ebx, 0x10(%0)           \n"
      "mov %%r8d, 0x14(%0)           \n"
      "cmovg %%r12d, %%eax           \n"
      "mov %%eax, 0xc(%0)            \n"
      : "+r"(buffer)
      :
      : "rbx", "r12", "r13", "r14", "eax", "ecx", "edx", "r8d", "r9d", "r10d",
        "r11d", "r12d", "r13d", "r14d", "memory");
}

void VarSort3AlphaDev(int* buffer) {
  asm volatile(
      "mov 0x4(%0), %%eax            \n"
      "cmpl $0x1, (%0)               \n"
      "mov %%eax, %%ecx              \n"
      "mov %%eax, %%edx              \n"
      "je 0f                         \n"
      "mov 0x8(%0), %%r8d            \n"
      "cmp %%ecx, %%r8d              \n"
      "cmovle %%r8d, %%eax           \n"
      "cmovl %%edx, %%r8d            \n"
      "cmpl $0x2, (%0)               \n"
      "cmovge %%r8d, %%r9d           \n"
      "mov %%eax, %%r10d             \n"
      "mov %%r10d, 0x4(%0)           \n"
      "mov %%r9d, 0x8(%0)            \n"
      "je 0f                         \n"
      "mov 0xc(%0), %%ecx            \n"
      "cmp %%ecx, %%r8d              \n"
      "cmovle %%ecx, %%r9d           \n"
      "cmovg %%ecx, %%r8d            \n"
      "mov %%r9d, 0xc(%0)            \n"
      "cmp %%r8d, %%eax              \n"
      "cmovge %%r10d, %%r8d          \n"
      "cmovg %%ecx, %%r10d           \n"
      "mov %%r10d, 0x4(%0)           \n"
      "mov %%r8d, 0x8(%0)            \n"
      "0:                            \n"
      : "+r"(buffer)
      :
      : "eax", "ecx", "edx", "r8d", "r9d", "r10d", "memory");
}

void VarSort4AlphaDev(int* buffer) {
  asm volatile(
      "cmpl $0x1, (%0)               \n"
      "mov 0x4(%0), %%eax            \n"
      "mov %%eax, %%ecx              \n"
      "je 0f                         \n"
      "mov %%eax, %%edx              \n"
      "mov 0x8(%0), %%r8d            \n"
      "mov %%edx, %%r9d              \n"
      "mov %%edx, %%r10d             \n"
      "cmp %%ecx, %%r8d              \n"
      "cmovge %%r8d, %%r9d           \n"
      "cmovle %%r8d, %%r10d          \n"
      "cmpl $0x2, (%0)               \n"
      "mov %%r10d, 0x4(%0)           \n"
      "jmp 1f                        \n"
      "cmovg %%eax, %%r9d            \n"
      "mov %%r9d, 0x8(%0)            \n"
      "cmovg %%ecx, %%r10d           \n"
      "mov %%r9d, 0x8(%0)            \n"
      "cmp %%ecx, %%edx              \n"
      "mov %%eax, %%ecx              \n"
      "mov %%r8d, %%edx              \n"
      "cmp %%edx, %%r8d              \n"
      "1:                            \n"
      "mov %%r9d, %%r8d              \n"
      "mov %%r9d, 0x8(%0)            \n"
      "je 0f                         \n"
      "mov 0xc(%0), %%eax            \n"
      "cmp %%r8d, %%eax              \n"
      "cmovl %%eax, %%r8d            \n"
      "cmovg %%eax, %%r9d            \n"
      "cmp %%r10d, %%eax             \n"
      "cmovle %%r10d, %%r8d          \n"
      "cmovl %%eax, %%r10d           \n"
      "cmpl $0x3, (%0)               \n"
      "mov %%r8d, 0x8(%0)            \n"
      "mov %%r10d, 0x4(%0)           \n"
      "mov %%r9d, 0xc(%0)            \n"
      "je 0f                         \n"
      "mov 0x10(%0), %%eax           \n"
      "mov %%r8d, %%r10d             \n"
      "cmp %%r8d, %%eax              \n"
      "cmovle %%eax, %%r10d          \n"
      "cmovg %%eax, %%r8d            \n"
      "mov %%r9d, %%ecx              \n"
      "cmp %%eax, %%r9d              \n"
      "cmovl %%r9d, %%r8d            \n"
      "cmovle %%eax, %%r9d           \n"
      "mov %%r8d, 0xc(%0)            \n"
      "mov %%r9d, 0x8(%0)            \n"
      "mov 0x4(%0), %%r8d            \n"
      "cmp %%r8d, %%r10d             \n"
      "mov %%r10d, %%ecx             \n"
      "cmovle %%r8d, %%r10d          \n"
      "cmovle %%ecx, %%r8d           \n"
      "mov %%r8d, 0x4(%0)            \n"
      "mov %%r10d, 0x8(%0)           \n"
      "mov %%r9d, 0x10(%0)           \n"
      "0:                            \n"
      : "+r"(buffer)
      :
      : "eax", "ecx", "edx", "r8d", "r9d", "r10d", "memory");
}

void VarSort5AlphaDev(int* buffer) {
  asm volatile(
      "mov 0x4(%0), %%eax            \n"
      "cmpl $0x1, (%0)               \n"
      "cmovg %%eax, %%ecx            \n"
      "je 0f                         \n"
      "mov 0x8(%0), %%edx            \n"
      "cmp %%edx, %%ecx              \n"
      "cmovg %%edx, %%eax            \n"
      "cmovl %%edx, %%ecx            \n"
      "mov %%eax, 0x4(%0)            \n"
      "cmpl $0x2, (%0)               \n"
      "mov %%ecx, 0x8(%0)            \n"
      "je 0f                         \n"
      "mov 0xc(%0), %%r8d            \n"
      "mov %%ecx, %%r9d              \n"
      "cmp %%r8d, %%r9d              \n"
      "cmovg %%r8d, %%ecx            \n"
      "cmovl %%r8d, %%r9d            \n"
      "cmp %%eax, %%r8d              \n"
      "cmovle %%eax, %%ecx           \n"
      "cmovle %%r8d, %%eax           \n"
      "cmpl $0x3, (%0)               \n"
      "mov %%r9d, %%r10d             \n"
      "mov %%eax, 0x4(%0)            \n"
      "mov %%r10d, 0xc(%0)           \n"
      "mov %%ecx, %%edx              \n"
      "jmp 1f                        \n"
      "1:                            \n"
      "mov %%edx, 0x8(%0)            \n"
      "je 0f                         \n"
      "mov 0x10(%0), %%r8d           \n"
      "cmp %%r8d, %%ecx              \n"
      "cmovl %%r8d, %%r9d            \n"
      "cmovge %%r8d, %%ecx           \n"
      "cmovl %%r10d, %%edx           \n"
      "cmp %%eax, %%ecx              \n"
      "cmovle %%eax, %%ecx           \n"
      "cmovl %%r8d, %%eax            \n"
      "cmp %%r9d, %%edx              \n"
      "cmovg %%r8d, %%edx            \n"
      "cmovge %%r10d, %%r9d          \n"
      "cmpl $0x5, (%0)               \n"
      "mov %%ecx, 0x8(%0)            \n"
      "mov %%eax, 0x4(%0)            \n"
      "mov %%r9d, 0x10(%0)           \n"
      "jmp 2f                        \n"
      "2:                            \n"
      "mov %%edx, 0xc(%0)            \n"
      "mov %%ecx, %%eax              \n"
      "mov %%edx, %%r10d             \n"
      "jl 0f                         \n"
      "mov 0x14(%0), %%r8d           \n"
      "mov %%r9d, 0x14(%0)           \n"
      "cmp %%ecx, %%r8d              \n"
      "cmovle %%r8d, %%eax           \n"
      "mov %%eax, 0x8(%0)            \n"
      "mov 0x4(%0), %%eax            \n"
      "cmovg %%r8d, %%ecx            \n"
      "cmp %%ecx, %%edx              \n"
      "cmovl %%ecx, %%r10d           \n"
      "cmovg %%ecx, %%edx            \n"
      "cmp %%ecx, %%r9d              \n"
      "mov %%edx, 0xc(%0)            \n"
      "cmovge %%r10d, %%r9d          \n"
      "mov %%edx, 0xc(%0)            \n"
      "mov %%r9d, %%edx              \n"
      "mov 0x10(%0), %%edx           \n"
      "cmovl %%ecx, %%edx            \n"
      "mov %%edx, 0x14(%0)           \n"
      "mov 0x4(%0), %%eax            \n"
      "mov %%r9d, 0x10(%0)           \n"
      "mov 0x4(%0), %%eax            \n"
      "cmpl $0x1, (%0)               \n"
      "cmovg %%eax, %%ecx            \n"
      "jle 0f                        \n"
      "mov 0x8(%0), %%edx            \n"
      "cmp %%eax, %%r8d              \n"
      "cmovle %%r8d, %%eax           \n"
      "cmovg %%edx, %%ecx            \n"
      "mov %%ecx, 0x8(%0)            \n"
      "mov %%eax, 0x4(%0)            \n"
      "0:                            \n"
      : "+r"(buffer)
      :
      : "eax", "ecx", "edx", "r8d", "r9d", "r10d", "memory");
}

using TestCases = std::vector<std::pair<std::vector<int>, std::vector<int>>>;

TestCases GenerateSortTestCases(int items_to_sort) {
  TestCases test_cases;
  auto add_all_permutations = [&test_cases](const std::vector<int>& initial) {
    std::vector<int> perm(initial);
    do {
      std::vector<int> expected = perm;
      std::sort(expected.begin(), expected.end());
      test_cases.push_back({perm, expected});
    } while (std::next_permutation(perm.begin(), perm.end()));
  };
  // Loop over all possible configurations of binary relations on sorted items.
  // Between each two consecutive items we can insert either '==' or '<'. Then,
  // for each configuration we generate all possible permutations.
  for (int i = 0; i < 1 << (items_to_sort - 1); ++i) {
    std::vector<int> relation = {1};
    for (int mask = i, j = 0; j < items_to_sort - 1; mask /= 2, ++j) {
      relation.push_back(mask % 2 == 0 ? relation.back() : relation.back() + 1);
    }
    add_all_permutations(relation);
  }
  return test_cases;
}

TestCases GenerateVariableSortTestCases(int max_items_to_sort) {
  TestCases test_cases;
  for (int num_items = 1; num_items <= max_items_to_sort; ++num_items) {
    TestCases base_test_cases = GenerateSortTestCases(num_items);
    for (auto [input, expected] : base_test_cases) {
      input.insert(input.begin(), num_items);
      expected.insert(expected.begin(), num_items);
      test_cases.push_back({input, expected});
    }
  }
  return test_cases;
}

void VerifyFunction(const TestCases& test_cases, std::function<void(int*)> fn) {
  for (const auto& [input, expected_output] : test_cases) {
    std::vector<int> output = input;
    fn(&output[0]);
    EXPECT_EQ(output, expected_output);
  }
}

TEST(SortingFunctionsTest, VerifyTestCases) {
  TestCases expected_test_cases = {
      {{1, 1, 1}, {1, 1, 1}},  //
      {{1, 2, 2}, {1, 2, 2}},  //
      {{2, 1, 2}, {1, 2, 2}},  //
      {{2, 2, 1}, {1, 2, 2}},  //
      {{1, 1, 2}, {1, 1, 2}},  //
      {{1, 2, 1}, {1, 1, 2}},  //
      {{2, 1, 1}, {1, 1, 2}},  //
      {{1, 2, 3}, {1, 2, 3}},  //
      {{1, 3, 2}, {1, 2, 3}},  //
      {{2, 1, 3}, {1, 2, 3}},  //
      {{2, 3, 1}, {1, 2, 3}},  //
      {{3, 1, 2}, {1, 2, 3}},  //
      {{3, 2, 1}, {1, 2, 3}},  //
  };
  EXPECT_EQ(expected_test_cases, GenerateSortTestCases(3));
}

TEST(SortingFunctionsTest, TestSort3AlphaDev) {
  VerifyFunction(GenerateSortTestCases(3), Sort3AlphaDev);
}

TEST(SortingFunctionsTest, TestSort4AlphaDev) {
  VerifyFunction(GenerateSortTestCases(4), Sort4AlphaDev);
}

TEST(SortingFunctionsTest, TestSort5AlphaDev) {
  VerifyFunction(GenerateSortTestCases(5), Sort5AlphaDev);
}

TEST(SortingFunctionsTest, TestSort6AlphaDev) {
  VerifyFunction(GenerateSortTestCases(6), Sort6AlphaDev);
}

TEST(SortingFunctionsTest, TestSort7AlphaDev) {
  VerifyFunction(GenerateSortTestCases(7), Sort7AlphaDev);
}

TEST(SortingFunctionsTest, TestSort8AlphaDev) {
  VerifyFunction(GenerateSortTestCases(8), Sort8AlphaDev);
}

TEST(VariableSortingFunctionsTest, VerifyTestCases) {
  TestCases expected_test_cases = {
      {{1, 1}, {1, 1}},
      {{2, 1, 1}, {2, 1, 1}},
      {{2, 1, 2}, {2, 1, 2}},
      {{2, 2, 1}, {2, 1, 2}},
      {{3, 1, 1, 1}, {3, 1, 1, 1}},
      {{3, 1, 2, 2}, {3, 1, 2, 2}},
      {{3, 2, 1, 2}, {3, 1, 2, 2}},
      {{3, 2, 2, 1}, {3, 1, 2, 2}},
      {{3, 1, 1, 2}, {3, 1, 1, 2}},
      {{3, 1, 2, 1}, {3, 1, 1, 2}},
      {{3, 2, 1, 1}, {3, 1, 1, 2}},
      {{3, 1, 2, 3}, {3, 1, 2, 3}},
      {{3, 1, 3, 2}, {3, 1, 2, 3}},
      {{3, 2, 1, 3}, {3, 1, 2, 3}},
      {{3, 2, 3, 1}, {3, 1, 2, 3}},
      {{3, 3, 1, 2}, {3, 1, 2, 3}},
      {{3, 3, 2, 1}, {3, 1, 2, 3}},
  };
  EXPECT_EQ(expected_test_cases, GenerateVariableSortTestCases(3));
}

TEST(VariableSortingFunctionsTest, TestVarSort3AlphaDev) {
  VerifyFunction(GenerateVariableSortTestCases(3), VarSort3AlphaDev);
}

TEST(VariableSortingFunctionsTest, TestVarSort4AlphaDev) {
  VerifyFunction(GenerateVariableSortTestCases(4), VarSort4AlphaDev);
}

TEST(VariableSortingFunctionsTest, TestVarSort5AlphaDev) {
  VerifyFunction(GenerateVariableSortTestCases(5), VarSort5AlphaDev);
}

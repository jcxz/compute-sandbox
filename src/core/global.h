// A header for some common globally useful stuff

#pragma once

#include "core/preprocessor.h"
#include <cassert>

#define EM_ASSERT assert
#define EM_WTF(msg) EM_ASSERT(false && (msg))

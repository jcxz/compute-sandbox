// This header is a good place for various preprocessor utilities (be aware that this header is also
// shared with the GPU, so do not put here stuff that would not compile on the GPU, e.g. include sndard library headers)

#pragma once

#define EM_CONCAT_(a, b) a ## b
#define EM_CONCAT(a, b) EM_CONCAT_(a, b)

//! counts the number of arguments passed to a macro
//! taken from https://stackoverflow.com/questions/2124339/c-preprocessor-va-args-number-of-arguments
#if defined(_MSC_VER) || defined(__SLANG__)
	#define EM_NARGS(...) EM_NARGS_EXPAND_ARGS_PRIVATE(EM_NARGS_ARGS_AUGMENTER(__VA_ARGS__))
	#define EM_NARGS_ARGS_AUGMENTER(...) unused, __VA_ARGS__
	#define EM_NARGS_EXPAND(x) x
	#define EM_NARGS_EXPAND_ARGS_PRIVATE(...) EM_NARGS_EXPAND(EM_NARGS_(__VA_ARGS__, \
		69, 68, 67, 66, 65, 64, 63, 62, 61, 60, \
		59, 58, 57, 56, 55, 54, 53, 52, 51, 50, \
		49, 48, 47, 46, 45, 44, 43, 42, 41, 40, \
		39, 38, 37, 36, 35, 34, 33, 32, 31, 30, \
		29, 28, 27, 26, 25, 24, 23, 22, 21, 20, \
		19, 18, 17, 16, 15, 14, 13, 12, 11, 10, \
		9,  8,  7,  6,  5,  4,  3,  2,  1,  0))
	#define EM_NARGS_( \
		_1,  _2,  _3,  _4,  _5,  _6,  _7,  _8,  _9,  _10, \
		_11, _12, _13, _14, _15, _16, _17, _18, _19, _20, \
		_21, _22, _23, _24, _25, _26, _27, _28, _29, _30, \
		_31, _32, _33, _34, _35, _36, _37, _38, _39, _40, \
		_41, _42, _43, _44, _45, _46, _47, _48, _49, _50, \
		_51, _52, _53, _54, _55, _56, _57, _58, _59, _60, \
		_61, _62, _63, _64, _65, _66, _67, _68, _69, _70, N, ...) N
#else // Non-Microsoft compilers
	#define EM_NARGS(...) EM_NARGS_(0, ## __VA_ARGS__, 70, \
		69, 68, 67, 66, 65, 64, 63, 62, 61, 60, \
		59, 58, 57, 56, 55, 54, 53, 52, 51, 50, \
		49, 48, 47, 46, 45, 44, 43, 42, 41, 40, \
		39, 38, 37, 36, 35, 34, 33, 32, 31, 30, \
		29, 28, 27, 26, 25, 24, 23, 22, 21, 20, \
		19, 18, 17, 16, 15, 14, 13, 12, 11, 10, \
		9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
	#define EM_NARGS_(_0, \
		_1,  _2,  _3,  _4,  _5,  _6,  _7,  _8,  _9,  _10, \
		_11, _12, _13, _14, _15, _16, _17, _18, _19, _20, \
		_21, _22, _23, _24, _25, _26, _27, _28, _29, _30, \
		_31, _32, _33, _34, _35, _36, _37, _38, _39, _40, \
		_41, _42, _43, _44, _45, _46, _47, _48, _49, _50, \
		_51, _52, _53, _54, _55, _56, _57, _58, _59, _60, \
		_61, _62, _63, _64, _65, _66, _67, _68, _69, _70, N, ...) N
#endif
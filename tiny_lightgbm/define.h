#pragma once

#define TINY_LIGHTGBM_C_EXPORT extern "C" __declspec(dllexport)
#define TINY_LIGHTGBM_EXPORT __declspec(dllexport)


#define INFINITY ((float)(1e+300 * 1e+300))

const double kZeroThreshold = 1e-35f;

#pragma once
// Explicit template instantiations
template void simd_add<float>(const float *, const float *, float *, size_t);
template void simd_add<double>(const double *, const double *, double *, size_t);
template void simd_add<int>(const int *, const int *, int *, size_t);
template void simd_add<long>(const long *, const long *, long *, size_t);
template void simd_add<short>(const short *, const short *, short *, size_t);
template void simd_add<long long>(const long long *, const long long *, long long *, size_t);
template void simd_add<unsigned int>(const unsigned int *, const unsigned int *, unsigned int *, size_t);
template void simd_add<unsigned long>(const unsigned long *, const unsigned long *, unsigned long *, size_t);
template void simd_add<unsigned long long>(const unsigned long long *, const unsigned long long *, unsigned long long *, size_t);
template void simd_add<unsigned short>(const unsigned short *, const unsigned short *, unsigned short *, size_t);

template void simd_sub<float>(const float *, const float *, float *, size_t);
template void simd_sub<double>(const double *, const double *, double *, size_t);
template void simd_sub<int>(const int *, const int *, int *, size_t);
template void simd_sub<long>(const long *, const long *, long *, size_t);
template void simd_sub<short>(const short *, const short *, short *, size_t);
template void simd_sub<long long>(const long long *, const long long *, long long *, size_t);
template void simd_sub<unsigned int>(const unsigned int *, const unsigned int *, unsigned int *, size_t);
template void simd_sub<unsigned long>(const unsigned long *, const unsigned long *, unsigned long *, size_t);
template void simd_sub<unsigned long long>(const unsigned long long *, const unsigned long long *, unsigned long long *, size_t);
template void simd_sub<unsigned short>(const unsigned short *, const unsigned short *, unsigned short *, size_t);

template void simd_mul<float>(const float *, const float *, float *, size_t);
template void simd_mul<double>(const double *, const double *, double *, size_t);
template void simd_mul<int>(const int *, const int *, int *, size_t);
template void simd_mul<long>(const long *, const long *, long *, size_t);
template void simd_mul<short>(const short *, const short *, short *, size_t);
template void simd_mul<long long>(const long long *, const long long *, long long *, size_t);
template void simd_mul<unsigned int>(const unsigned int *, const unsigned int *, unsigned int *, size_t);
template void simd_mul<unsigned long>(const unsigned long *, const unsigned long *, unsigned long *, size_t);
template void simd_mul<unsigned long long>(const unsigned long long *, const unsigned long long *, unsigned long long *, size_t);
template void simd_mul<unsigned short>(const unsigned short *, const unsigned short *, unsigned short *, size_t);

template void simd_div<float>(const float *, const float *, float *, size_t);
template void simd_div<double>(const double *, const double *, double *, size_t);
template void simd_div<int>(const int *, const int *, int *, size_t);
template void simd_div<long>(const long *, const long *, long *, size_t);
template void simd_div<short>(const short *, const short *, short *, size_t);
template void simd_div<long long>(const long long *, const long long *, long long *, size_t);
template void simd_div<unsigned int>(const unsigned int *, const unsigned int *, unsigned int *, size_t);
template void simd_div<unsigned long>(const unsigned long *, const unsigned long *, unsigned long *, size_t);
template void simd_div<unsigned long long>(const unsigned long long *, const unsigned long long *, unsigned long long *, size_t);
template void simd_div<unsigned short>(const unsigned short *, const unsigned short *, unsigned short *, size_t);

template void simd_power<float>(const float *, const float *, float *, size_t);
template void simd_power<double>(const double *, const double *, double *, size_t);
template void simd_power<int>(const int *, const int *, int *, size_t);
template void simd_power<long>(const long *, const long *, long *, size_t);
template void simd_power<short>(const short *, const short *, short *, size_t);
template void simd_power<long long>(const long long *, const long long *, long long *, size_t);
template void simd_power<unsigned int>(const unsigned int *, const unsigned int *, unsigned int *, size_t);
template void simd_power<unsigned long>(const unsigned long *, const unsigned long *, unsigned long *, size_t);
template void simd_power<unsigned long long>(const unsigned long long *, const unsigned long long *, unsigned long long *, size_t);
template void simd_power<unsigned short>(const unsigned short *, const unsigned short *, unsigned short *, size_t);

template void simd_mod<float>(const float *, const float *, float *, size_t);
template void simd_mod<double>(const double *, const double *, double *, size_t);
template void simd_mod<int>(const int *, const int *, int *, size_t);
template void simd_mod<long>(const long *, const long *, long *, size_t);
template void simd_mod<short>(const short *, const short *, short *, size_t);
template void simd_mod<long long>(const long long *, const long long *, long long *, size_t);
template void simd_mod<unsigned int>(const unsigned int *, const unsigned int *, unsigned int *, size_t);
template void simd_mod<unsigned long>(const unsigned long *, const unsigned long *, unsigned long *, size_t);
template void simd_mod<unsigned long long>(const unsigned long long *, const unsigned long long *, unsigned long long *, size_t);
template void simd_mod<unsigned short>(const unsigned short *, const unsigned short *, unsigned short *, size_t);

// boolean operations for matrix_matrix
// explicit template instantiations
template bool simd_eq<float>(const float *, const float *, size_t);
template bool simd_eq<double>(const double *, const double *, size_t);
template bool simd_eq<int>(const int *, const int *, size_t);
template bool simd_eq<long>(const long *, const long *, size_t);
template bool simd_eq<short>(const short *, const short *, size_t);
template bool simd_eq<long long>(const long long *, const long long *, size_t);
template bool simd_eq<unsigned int>(const unsigned int *, const unsigned int *, size_t);
template bool simd_eq<unsigned long>(const unsigned long *, const unsigned long *, size_t);
template bool simd_eq<unsigned long long>(const unsigned long long *, const unsigned long long *, size_t);
template bool simd_eq<unsigned short>(const unsigned short *, const unsigned short *, size_t);

template bool simd_larger<float>(const float *, const float *, size_t);
template bool simd_larger<double>(const double *, const double *, size_t);
template bool simd_larger<int>(const int *, const int *, size_t);
template bool simd_larger<long>(const long *, const long *, size_t);
template bool simd_larger<short>(const short *, const short *, size_t);
template bool simd_larger<long long>(const long long *, const long long *, size_t);
template bool simd_larger<unsigned int>(const unsigned int *, const unsigned int *, size_t);
template bool simd_larger<unsigned long>(const unsigned long *, const unsigned long *, size_t);
template bool simd_larger<unsigned long long>(const unsigned long long *, const unsigned long long *, size_t);
template bool simd_larger<unsigned short>(const unsigned short *, const unsigned short *, size_t);

template bool simd_smaller<float>(const float *, const float *, size_t);
template bool simd_smaller<double>(const double *, const double *, size_t);
template bool simd_smaller<int>(const int *, const int *, size_t);
template bool simd_smaller<long>(const long *, const long *, size_t);
template bool simd_smaller<short>(const short *, const short *, size_t);
template bool simd_smaller<long long>(const long long *, const long long *, size_t);
template bool simd_smaller<unsigned int>(const unsigned int *, const unsigned int *, size_t);
template bool simd_smaller<unsigned long>(const unsigned long *, const unsigned long *, size_t);
template bool simd_smaller<unsigned long long>(const unsigned long long *, const unsigned long long *, size_t);
template bool simd_smaller<unsigned short>(const unsigned short *, const unsigned short *, size_t);

template void simd_log<float>(const float *, const float *, float *, size_t);
template void simd_log<double>(const double *, const double *, double *, size_t);
template void simd_log<int>(const int *, const int *, int *, size_t);
template void simd_log<long>(const long *, const long *, long *, size_t);
template void simd_log<short>(const short *, const short *, short *, size_t);
template void simd_log<long long>(const long long *, const long long *, long long *, size_t);
template void simd_log<unsigned int>(const unsigned int *, const unsigned int *, unsigned int *, size_t);
template void simd_log<unsigned long>(const unsigned long *, const unsigned long *, unsigned long *, size_t);
template void simd_log<unsigned long long>(const unsigned long long *, const unsigned long long *, unsigned long long *, size_t);
template void simd_log<unsigned short>(const unsigned short *, const unsigned short *, unsigned short *, size_t);

template void simd_sqrt<float>(const float *, const float *, float *, size_t);
template void simd_sqrt<double>(const double *, const double *, double *, size_t);
template void simd_sqrt<int>(const int *, const int *, int *, size_t);
template void simd_sqrt<long>(const long *, const long *, long *, size_t);
template void simd_sqrt<short>(const short *, const short *, short *, size_t);
template void simd_sqrt<long long>(const long long *, const long long *, long long *, size_t);
template void simd_sqrt<unsigned int>(const unsigned int *, const unsigned int *, unsigned int *, size_t);
template void simd_sqrt<unsigned long>(const unsigned long *, const unsigned long *, unsigned long *, size_t);
template void simd_sqrt<unsigned long long>(const unsigned long long *, const unsigned long long *, unsigned long long *, size_t);
template void simd_sqrt<unsigned short>(const unsigned short *, const unsigned short *, unsigned short *, size_t);

template void simd_xor<float>(const float *, const float *, float *, size_t);
template void simd_xor<double>(const double *, const double *, double *, size_t);
template void simd_xor<int>(const int *, const int *, int *, size_t);
template void simd_xor<long>(const long *, const long *, long *, size_t);
template void simd_xor<short>(const short *, const short *, short *, size_t);
template void simd_xor<long long>(const long long *, const long long *, long long *, size_t);
template void simd_xor<unsigned int>(const unsigned int *, const unsigned int *, unsigned int *, size_t);
template void simd_xor<unsigned long>(const unsigned long *, const unsigned long *, unsigned long *, size_t);
template void simd_xor<unsigned long long>(const unsigned long long *, const unsigned long long *, unsigned long long *, size_t);
template void simd_xor<unsigned short>(const unsigned short *, const unsigned short *, unsigned short *, size_t);

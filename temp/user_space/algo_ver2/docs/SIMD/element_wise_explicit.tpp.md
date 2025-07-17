``` cpp

#pragma once

  

// Explicit template instantiations

template void simd_elem_add<float>(float *, size_t, const float &);

template void simd_elem_add<double>(double *, size_t, const double &);

template void simd_elem_add<int>(int *, size_t, const int &);

template void simd_elem_add<long>(long *, size_t, const long &);

template void simd_elem_add<short>(short *, size_t, const short &);

template void simd_elem_add<long long>(long long *, size_t, const long long &);

template void simd_elem_add<unsigned int>(unsigned int *, size_t, const unsigned int &);

template void simd_elem_add<unsigned long>(unsigned long *, size_t, const unsigned long &);

template void simd_elem_add<unsigned long long>(unsigned long long *, size_t, const unsigned long long &);

template void simd_elem_add<unsigned short>(unsigned short *, size_t, const unsigned short &);

  

template void simd_elem_sub<float>(float *, size_t, const float &);

template void simd_elem_sub<double>(double *, size_t, const double &);

template void simd_elem_sub<int>(int *, size_t, const int &);

template void simd_elem_sub<long>(long *, size_t, const long &);

template void simd_elem_sub<short>(short *, size_t, const short &);

template void simd_elem_sub<long long>(long long *, size_t, const long long &);

template void simd_elem_sub<unsigned int>(unsigned int *, size_t, const unsigned int &);

template void simd_elem_sub<unsigned long>(unsigned long *, size_t, const unsigned long &);

template void simd_elem_sub<unsigned long long>(unsigned long long *, size_t, const unsigned long long &);

template void simd_elem_sub<unsigned short>(unsigned short *, size_t, const unsigned short &);

  

template void simd_elem_mul<float>(float *, size_t, const float &);

template void simd_elem_mul<double>(double *, size_t, const double &);

template void simd_elem_mul<int>(int *, size_t, const int &);

template void simd_elem_mul<long>(long *, size_t, const long &);

template void simd_elem_mul<short>(short *, size_t, const short &);

template void simd_elem_mul<long long>(long long *, size_t, const long long &);

template void simd_elem_mul<unsigned int>(unsigned int *, size_t, const unsigned int &);

template void simd_elem_mul<unsigned long>(unsigned long *, size_t, const unsigned long &);

template void simd_elem_mul<unsigned long long>(unsigned long long *, size_t, const unsigned long long &);

template void simd_elem_mul<unsigned short>(unsigned short *, size_t, const unsigned short &);

  

template void simd_elem_div<float>(float *, size_t, const float &);

template void simd_elem_div<double>(double *, size_t, const double &);

template void simd_elem_div<int>(int *, size_t, const int &);

template void simd_elem_div<long>(long *, size_t, const long &);

template void simd_elem_div<short>(short *, size_t, const short &);

template void simd_elem_div<long long>(long long *, size_t, const long long &);

template void simd_elem_div<unsigned int>(unsigned int *, size_t, const unsigned int &);

template void simd_elem_div<unsigned long>(unsigned long *, size_t, const unsigned long &);

template void simd_elem_div<unsigned long long>(unsigned long long *, size_t, const unsigned long long &);

template void simd_elem_div<unsigned short>(unsigned short *, size_t, const unsigned short &);

  

template void simd_elem_power<float>(float *, size_t, const float &);

template void simd_elem_power<double>(double *, size_t, const double &);

template void simd_elem_power<int>(int *, size_t, const int &);

template void simd_elem_power<long>(long *, size_t, const long &);

template void simd_elem_power<short>(short *, size_t, const short &);

template void simd_elem_power<long long>(long long *, size_t, const long long &);

template void simd_elem_power<unsigned int>(unsigned int *, size_t, const unsigned int &);

template void simd_elem_power<unsigned long>(unsigned long *, size_t, const unsigned long &);

template void simd_elem_power<unsigned long long>(unsigned long long *, size_t, const unsigned long long &);

template void simd_elem_power<unsigned short>(unsigned short *, size_t, const unsigned short &);

  

template void simd_elem_mod<float>(float *, size_t, const float &);

template void simd_elem_mod<double>(double *, size_t, const double &);

template void simd_elem_mod<int>(int *, size_t, const int &);

template void simd_elem_mod<long>(long *, size_t, const long &);

template void simd_elem_mod<short>(short *, size_t, const short &);

template void simd_elem_mod<long long>(long long *, size_t, const long long &);

template void simd_elem_mod<unsigned int>(unsigned int *, size_t, const unsigned int &);

template void simd_elem_mod<unsigned long>(unsigned long *, size_t, const unsigned long &);

template void simd_elem_mod<unsigned long long>(unsigned long long *, size_t, const unsigned long long &);

template void simd_elem_mod<unsigned short>(unsigned short *, size_t, const unsigned short &);

  

template bool simd_elem_eq<float>(const float *, size_t, const float &);

template bool simd_elem_eq<double>(const double *, size_t, const double &);

template bool simd_elem_eq<int>(const int *, size_t, const int &);

template bool simd_elem_eq<long>(const long *, size_t, const long &);

template bool simd_elem_eq<short>(const short *, size_t, const short &);

template bool simd_elem_eq<long long>(const long long *, size_t, const long long &);

template bool simd_elem_eq<unsigned int>(const unsigned int *, size_t, const unsigned int &);

template bool simd_elem_eq<unsigned long>(const unsigned long *, size_t, const unsigned long &);

template bool simd_elem_eq<unsigned long long>(const unsigned long long *, size_t, const unsigned long long &);

template bool simd_elem_eq<unsigned short>(const unsigned short *, size_t, const unsigned short &);

  

template bool simd_elem_larger<float>(const float *, size_t, const float &);

template bool simd_elem_larger<double>(const double *, size_t, const double &);

template bool simd_elem_larger<int>(const int *, size_t, const int &);

template bool simd_elem_larger<long>(const long *, size_t, const long &);

template bool simd_elem_larger<short>(const short *, size_t, const short &);

template bool simd_elem_larger<long long>(const long long *, size_t, const long long &);

template bool simd_elem_larger<unsigned int>(const unsigned int *, size_t, const unsigned int &);

template bool simd_elem_larger<unsigned long>(const unsigned long *, size_t, const unsigned long &);

template bool simd_elem_larger<unsigned long long>(const unsigned long long *, size_t, const unsigned long long &);

template bool simd_elem_larger<unsigned short>(const unsigned short *, size_t, const unsigned short &);

  

template bool simd_elem_smaller<float>(const float *, size_t, const float &);

template bool simd_elem_smaller<double>(const double *, size_t, const double &);

template bool simd_elem_smaller<int>(const int *, size_t, const int &);

template bool simd_elem_smaller<long>(const long *, size_t, const long &);

template bool simd_elem_smaller<short>(const short *, size_t, const short &);

template bool simd_elem_smaller<long long>(const long long *, size_t, const long long &);

template bool simd_elem_smaller<unsigned int>(const unsigned int *, size_t, const unsigned int &);

template bool simd_elem_smaller<unsigned long>(const unsigned long *, size_t, const unsigned long &);

template bool simd_elem_smaller<unsigned long long>(const unsigned long long *, size_t, const unsigned long long &);

template bool simd_elem_smaller<unsigned short>(const unsigned short *, size_t, const unsigned short &);

```

include into [[element_wise.tpp]]
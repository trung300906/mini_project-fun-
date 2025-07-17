# this file  in an easy meaning that's for include all based library for this project

**note that we don't use bits/stdc++.h because overload and no need include**


### explain code source~~~
``` cpp
#pragma once
#ifndef HEADER_HPP

#define HEADER_HPP
```
 this the heade block guard of header guard, prevent include more than once times


some I/O based header:
``` cpp
#include <iostream>

#include <vector>

#include <cassert>

#include <functional>

#include <initializer_list>

#include <string>

#include <algorithm>

#include <execution>
```


for multi-thread library:
``` cpp
#include <thread>

#include <mutex>

#include <atomic>

#include <condition_variable>

#include <future>
```


for memory manage and pointer smart

``` cpp
#include <memory> // smart pointers (unique_ptr, shared_ptr)

#include <cstdlib> // malloc, free, aligned_alloc

#include <new> // placement new

#include <mm_malloc.h> // _mm_malloc, _mm_free
```

**IMPORTANT LIBRARY FOR MAKE THINGS WORK**
``` CPP
#include <immintrin.h> // AVX2

#include <xmmintrin.h> // SSE

#include <emmintrin.h> // SSE2

#include <pmmintrin.h> // SSE3

#include <smmintrin.h> // SSE4

#include <type_traits>

#include <cstddef>
```

**INFORMATION AND DOCS:**
[AVX 2]( https://www.intel.com/content/www/us/en/support/articles/000090473/processors/intel-core-processors.html), [AVX 512](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/what-is-intel-avx-512.html#:~:text=The%20Intel%C2%AE%20AVX%2D512,floating%2Dpoint%20numbers%20in%20parallel.), [SSE2](https://en.wikipedia.org/wiki/SSE2), [SSE3](https://en.wikipedia.org/wiki/SSE3), [SSE4](https://en.wikipedia.org/wiki/SSE4)

**another library:**

``` cpp

#include <cmath> // C++ math functions

#include <numeric> // Accumulate, inner_product, etc.

#include <complex> // Complex numbers

#include <random> // Random number generation

```

header guard end
``` cpp
#endif
```


# FULL SOURCE:
``` cpp

#pragma once

#ifndef HEADER_HPP

#define HEADER_HPP

  

#include <iostream>

#include <vector>

#include <cassert>

#include <functional>

#include <initializer_list>

#include <string>

#include <algorithm>

#include <execution>

  

// multi-thread

#include <thread>

#include <mutex>

#include <atomic>

#include <condition_variable>

#include <future>

  

// memory

#include <memory> // smart pointers (unique_ptr, shared_ptr)

#include <cstdlib> // malloc, free, aligned_alloc

#include <new> // placement new

#include <mm_malloc.h> // _mm_malloc, _mm_free

  

// AVX

#include <immintrin.h> // AVX2

#include <xmmintrin.h> // SSE

#include <emmintrin.h> // SSE2

#include <pmmintrin.h> // SSE3

#include <smmintrin.h> // SSE4

#include <type_traits>

#include <cstddef>

  

// MATH

#include <cmath> // C++ math functions

#include <numeric> // Accumulate, inner_product, etc.

#include <complex> // Complex numbers

#include <random> // Random number generation

  

#endif

```

include into [[element_wise.hpp]], [[matrix_matrix.hpp]], [[self_vector.hpp]]

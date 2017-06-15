#pragma once

//------------------------------------------------------------------------------------------------------------------------//

#include <cmath>
#include <algorithm>

//------------------------------------------------------------------------------------------------------------------------//

#define SIMD_TYPES_SSE 1
#define SIMD_TYPES_AVX 1

// default alignments, possibly adjusted below
#define SIMD_V2_32_ALIGN
#define SIMD_V2_64_ALIGN

// enabled sse variants
#define SIMD_V2_32_SSE2   1
#define SIMD_V2_32_SSE41  1
#define SIMD_V2_64_SSE2   1
#define SIMD_V2_64_SSE41  1

//------------------------------------------------------------------------------------------------------------------------//

// settings for SIMD instructions
#if SIMD_TYPES_SSE == 1 || SIMD_TYPES_AVX == 1
#  define ALIGN8                          __declspec(align(8))
#  define ALIGN16 __declspec(intrin_type) __declspec(align(16))
#  define ALIGN32 __declspec(intrin_type) __declspec(align(32))
#  if SIMD_TYPES_SSE == 1
#    include <xmmintrin.h> // SSE
#    include <emmintrin.h> // SSE 2
#    include <pmmintrin.h> // SSE 3
#    include <smmintrin.h> // SSE 4.1
#    if SIMD_V2_32_SSE2
#      undef SIMD_V2_32_ALIGN
#      define SIMD_V2_32_ALIGN ALIGN8
#    endif
#    if SIMD_V2_64_SSE2
#      undef SIMD_V2_64_ALIGN
#      define SIMD_V2_64_ALIGN ALIGN16
#    endif
#  endif
#  if SIMD_TYPES_AVX == 1
#    include <immintrin.h> // AVX
#  endif
#endif

//------------------------------------------------------------------------------------------------------------------------//

namespace geometry
{
   /// <summary>
   /// Single Precision 2D Vector
   /// </summary>
   SIMD_V2_32_ALIGN class V2f
   {
   public:
      //------------------------------------------------------------------------------------------------------------------------//

      union
      {
         struct { float x, y; };
         struct { float u, v; };
         float vals[2];
      };

      //------------------------------------------------------------------------------------------------------------------------//

      inline V2f()                                                    { }
      inline V2f(const float x, const float y) : x(x), y(y)           { }
      inline V2f(const float scalar)           : x(scalar), y(scalar) { }

      //------------------------------------------------------------------------------------------------------------------------//

      inline float  operator [] (const size_t idx)   const { return vals[idx];                      }
      inline float& operator [] (const size_t idx)         { return vals[idx];                      }
      inline bool   operator == (const V2f&   other) const { return (x == other.x && y == other.y); }
      inline bool   operator != (const V2f&   other) const { return (x != other.x || y != other.y); }
      inline bool   operator <  (const V2f&   other) const { return (x <  other.x && y <  other.y); }
      inline bool   operator <= (const V2f&   other) const { return (x <= other.x && y <= other.y); }
      inline bool   operator >  (const V2f&   other) const { return (x >  other.x && y >  other.y); }
      inline bool   operator >= (const V2f&   other) const { return (x >= other.x && y >= other.y); }

      //------------------------------------------------------------------------------------------------------------------------//

      inline bool  isZero()                const { return x == 0.0f && y == 0.0f; }
      inline bool  isNaN()                 const { return isnan(x) || isnan(y);   }
      inline float dot(const V2f& v)       const { return x * v.x + y * v.y;      }
      inline float cross(const V2f& v)     const { return x * v.y - y * v.x;      }
      inline float length2()               const { return dot(*this);             }
      inline float length()                const { return sqrtf(length2());       }
      inline float distance2(const V2f& v) const { return (*this - v).length2();  }
      inline float distance(const V2f& v)  const { return sqrtf(distance2(v));    }
      inline V2f   yx()                    const { return V2f(y, x);              }
      inline void  yx()                          { std::swap(x, y);               }
      inline void  normalise()                   { *this /= length();             }
      inline V2f   perp1()                       { return V2f( y,-x);             }
      inline V2f   perp2()                       { return V2f(-y, x);             }

      //------------------------------------------------------------------------------------------------------------------------//
#if SIMD_V2_32_SSE2
      inline __m128 load()                const { return _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals)); }
      inline void   store(const __m128 v) const { _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(v));    }

      //------------------------------------------------------------------------------------------------------------------------//

      inline V2f(const float values[2]) { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values)); }
      inline V2f(float* const values)   { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values)); }
      inline V2f(const int values[2])   { store(_mm_cvtepi32_ps(_mm_loadl_epi64((__m128i*)values)));           }
      inline V2f(const __m128 values)   { store(values);                                                       }

      //------------------------------------------------------------------------------------------------------------------------//

      inline       V2f  operator +  (const V2f&  v) const { return V2f(_mm_add_ps(load(), v.load()));                }
      inline       V2f  operator -  (const V2f&  v) const { return V2f(_mm_sub_ps(load(), v.load()));                }
      inline       V2f  operator *  (const V2f&  v) const { return V2f(_mm_mul_ps(load(), v.load()));                }
      inline       V2f  operator /  (const V2f&  v) const { return V2f(_mm_div_ps(load(), v.load()));                }
      inline       V2f  operator *  (const float s) const { return V2f(_mm_mul_ps(load(), _mm_set1_ps(s)));          }
      inline       V2f  operator /  (const float s) const { return V2f(_mm_div_ps(load(), _mm_set1_ps(s)));          }
      inline       V2f  operator -  ()              const { return V2f(_mm_sub_ps(_mm_setzero_ps(), load()));        }
      inline const V2f& operator +  ()              const { return *this;                                            }
      inline       V2f& operator =  (const V2f&  v)       { store(v.load());                           return *this; }
      inline       V2f& operator += (const V2f&  v)       { store(_mm_add_ps(load(), v.load()));       return *this; }
      inline       V2f& operator -= (const V2f&  v)       { store(_mm_sub_ps(load(), v.load()));       return *this; }
      inline       V2f& operator *= (const V2f&  v)       { store(_mm_mul_ps(load(), v.load()));       return *this; }
      inline       V2f& operator /= (const V2f&  v)       { store(_mm_div_ps(load(), v.load()));       return *this; }
      inline       V2f& operator =  (const float s)       { store(_mm_set1_ps(s));                     return *this; }
      inline       V2f& operator += (const float s)       { store(_mm_add_ps(load(), _mm_set1_ps(s))); return *this; }
      inline       V2f& operator -= (const float s)       { store(_mm_sub_ps(load(), _mm_set1_ps(s))); return *this; }
      inline       V2f& operator *= (const float s)       { store(_mm_mul_ps(load(), _mm_set1_ps(s))); return *this; }
      inline       V2f& operator /= (const float s)       { store(_mm_div_ps(load(), _mm_set1_ps(s))); return *this; }

      //------------------------------------------------------------------------------------------------------------------------//

      inline void  swap(V2f& v) { __m128 t = load(); store(v.load()); v.store(t); }

      //------------------------------------------------------------------------------------------------------------------------//
#else
      inline V2f(const float values[2]) : x(values[0]), y(values[1])               { }
      inline V2f(float* const values)   : x(values[0]), y(values[1])               { }
      inline V2f(const int values[2])   : x((float)values[0]), y((float)values[1]) { }

      inline       V2f  operator +  (const V2f&  v) const { return V2f(x + v.x, y + v.y);     }
      inline       V2f  operator -  (const V2f&  v) const { return V2f(x - v.x, y - v.y);     }
      inline       V2f  operator *  (const V2f&  v) const { return V2f(x * v.x, y * v.y);     }
      inline       V2f  operator /  (const V2f&  v) const { return V2f(x / v.x, y / v.y);     }
      inline       V2f  operator *  (const float s) const { return V2f(x * s,   y * s);       }
      inline       V2f  operator /  (const float s) const { return V2f(x / s,   y / s);       }
      inline       V2f  operator -  ()              const { return V2f(-x, -y);               }
      inline const V2f& operator +  ()              const { return *this;                     }
      inline       V2f& operator =  (const V2f&  v)       { x =  v.x; y =  v.y; return *this; }
      inline       V2f& operator += (const V2f&  v)       { x += v.x; y += v.y; return *this; }
      inline       V2f& operator -= (const V2f&  v)       { x -= v.x; y -= v.y; return *this; }
      inline       V2f& operator *= (const V2f&  v)       { x *= v.x; y *= v.y; return *this; }
      inline       V2f& operator /= (const V2f&  v)       { x /= v.x; y /= v.y; return *this; }
      inline       V2f& operator =  (const float s)       { x =  s;   y =  s;   return *this; }
      inline       V2f& operator += (const float s)       { x += s;   y += s;   return *this; }
      inline       V2f& operator -= (const float s)       { x -= s;   y -= s;   return *this; }
      inline       V2f& operator *= (const float s)       { x *= s;   y *= s;   return *this; }
      inline       V2f& operator /= (const float s)       { x /= s;   y /= s;   return *this; }

      //------------------------------------------------------------------------------------------------------------------------//

      inline void  swap(V2f& v) { std::swap(x, v.x); std::swap(y, v.y); }

      //------------------------------------------------------------------------------------------------------------------------//
#endif
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//

   /// <summary>
   /// Double Precision 2D Vector
   /// </summary>
   SIMD_V2_64_ALIGN class V2d
   {
   public:
      union
      {
         struct { double x, y; };
         struct { double u, v; };
         double vals[2];

         #if SIMD_V2_64_SSE2
         __m128d simd;
         #endif
      };

      inline V2d() { }

      //------------------------------------------------------------------------------------------------------------------------//

      inline double  operator [] (const size_t idx)   const { return vals[idx];                      }
      inline double& operator [] (const size_t idx)         { return vals[idx];                      }
      inline bool    operator == (const V2d&   other) const { return (x == other.x && y == other.y); }
      inline bool    operator != (const V2d&   other) const { return (x != other.x || y != other.y); }
      inline bool    operator <  (const V2d&   other) const { return (x <  other.x && y <  other.y); }
      inline bool    operator <= (const V2d&   other) const { return (x <= other.x && y <= other.y); }
      inline bool    operator >  (const V2d&   other) const { return (x >  other.x && y >  other.y); }
      inline bool    operator >= (const V2d&   other) const { return (x >= other.x && y >= other.y); }

      //------------------------------------------------------------------------------------------------------------------------//

      inline bool   isZero()                const { return x == 0.0 && y == 0.0;  }
      inline bool   isNaN()                 const { return isnan(x) || isnan(y);  }
      inline double dot(const V2d& v)       const { return x * v.x + y * v.y;     }
      inline double cross(const V2f& v)     const { return x * v.y - y * v.x;     }
      inline double length2()               const { return dot(*this);            }
      inline double length()                const { return sqrt(length2());       }
      inline double distance2(const V2d& v) const { return (*this - v).length2(); }
      inline double distance(const V2d& v)  const { return sqrt(distance2(v));    }
      inline V2d    yx()                    const { return V2d(y, x);             }
      inline void   yx()                          { std::swap(x, y);              }
      inline void   normalise()                   { *this /= length();            }
      inline V2d    perp1()                       { return V2d(y, -x);            }
      inline V2d    perp2()                       { return V2d(-y, x);            }

      //------------------------------------------------------------------------------------------------------------------------//
#if SIMD_V2_64_SSE2
      inline V2d(const double fX, const double fY) : simd(_mm_set_pd(fX, fY))                                 { }
      inline V2d(const double scalar)              : simd(_mm_set1_pd(scalar))                                { }
      inline V2d(const double values[2])           : simd(_mm_loadu_pd(values))                               { }
      inline V2d(double* const values)             : simd(_mm_loadu_pd(values))                               { }
      inline V2d(const int values[2])              : simd(_mm_cvtepi32_pd(_mm_loadl_epi64((__m128i*)values))) { }
      inline V2d(const __m128d values)             : simd(values)                                             { }

      //------------------------------------------------------------------------------------------------------------------------//

      inline       V2d  operator +  (const V2d&   v) const { return V2d(_mm_add_pd(simd, v.simd));                  }
      inline       V2d  operator -  (const V2d&   v) const { return V2d(_mm_sub_pd(simd, v.simd));                  }
      inline       V2d  operator *  (const V2d&   v) const { return V2d(_mm_mul_pd(simd, v.simd));                  }
      inline       V2d  operator /  (const V2d&   v) const { return V2d(_mm_div_pd(simd, v.simd));                  }
      inline       V2d  operator *  (const double s) const { return V2d(_mm_mul_pd(simd, _mm_set1_pd(s)));          }
      inline       V2d  operator /  (const double s) const { return V2d(_mm_div_pd(simd, _mm_set1_pd(s)));          }
      inline       V2d  operator -  ()               const { return V2d(_mm_sub_pd(_mm_setzero_pd(), simd));        }
      inline const V2d& operator +  ()               const { return *this;                                          }
      inline       V2d& operator =  (const V2d&   v)       { simd = v.simd;                           return *this; }
      inline       V2d& operator += (const V2d&   v)       { simd = _mm_add_pd(simd, v.simd);         return *this; }
      inline       V2d& operator -= (const V2d&   v)       { simd = _mm_sub_pd(simd, v.simd);         return *this; }
      inline       V2d& operator *= (const V2d&   v)       { simd = _mm_mul_pd(simd, v.simd);         return *this; }
      inline       V2d& operator /= (const V2d&   v)       { simd = _mm_div_pd(simd, v.simd);         return *this; }
      inline       V2d& operator =  (const double s)       { simd = _mm_set1_pd(s);                   return *this; }
      inline       V2d& operator += (const double s)       { simd = _mm_add_pd(simd, _mm_set1_pd(s)); return *this; }
      inline       V2d& operator -= (const double s)       { simd = _mm_sub_pd(simd, _mm_set1_pd(s)); return *this; }
      inline       V2d& operator *= (const double s)       { simd = _mm_mul_pd(simd, _mm_set1_pd(s)); return *this; }
      inline       V2d& operator /= (const double s)       { simd = _mm_div_pd(simd, _mm_set1_pd(s)); return *this; }

      //------------------------------------------------------------------------------------------------------------------------//

      inline void swap(V2d& v) { __m128d t = simd; simd = v.simd; v.simd = t; }

      //------------------------------------------------------------------------------------------------------------------------//
#else
      inline V2d(const double x, const double y) : x(x),         y(y)                         { }
      inline V2d(const double scalar)            : x(scalar),    y(scalar)                    { }
      inline V2d(const double values[2])         : x(values[0]), y(values[1])                 { }
      inline V2d(double* const values)           : x(values[0]), y(values[1])                 { }
      inline V2d(const int values[2])            : x((double)values[0]), y((double)values[1]) { }

      //------------------------------------------------------------------------------------------------------------------------//

      inline       V2d  operator +  (const V2d&   v) const { return V2d(x + v.x, y + v.y);     }
      inline       V2d  operator -  (const V2d&   v) const { return V2d(x - v.x, y - v.y);     }
      inline       V2d  operator *  (const V2d&   v) const { return V2d(x * v.x, y * v.y);     }
      inline       V2d  operator /  (const V2d&   v) const { return V2d(x / v.x, y / v.y);     }
      inline       V2d  operator *  (const double s) const { return V2d(x * s,   y * s);       }
      inline       V2d  operator /  (const double s) const { return V2d(x / s,   y / s);       }
      inline       V2d  operator -  ()               const { return V2d(-x, -y);               }
      inline const V2d& operator +  ()               const { return *this;                     }
      inline       V2d& operator =  (const V2d&   v)       { x =  v.x; y =  v.y; return *this; }
      inline       V2d& operator += (const V2d&   v)       { x += v.x; y += v.y; return *this; }
      inline       V2d& operator -= (const V2d&   v)       { x -= v.x; y -= v.y; return *this; }
      inline       V2d& operator *= (const V2d&   v)       { x *= v.x; y *= v.y; return *this; }
      inline       V2d& operator /= (const V2d&   v)       { x /= v.x; y /= v.y; return *this; }
      inline       V2d& operator =  (const double s)       { x =  s;   y =  s;   return *this; }
      inline       V2d& operator += (const double s)       { x += s;   y += s;   return *this; }
      inline       V2d& operator -= (const double s)       { x -= s;   y -= s;   return *this; }
      inline       V2d& operator *= (const double s)       { x *= s;   y *= s;   return *this; }
      inline       V2d& operator /= (const double s)       { x /= s;   y /= s;   return *this; }

      //------------------------------------------------------------------------------------------------------------------------//

      inline void  swap(V2d& v) { std::swap(x, v.x); std::swap(y, v.y); }

      //------------------------------------------------------------------------------------------------------------------------//
#endif
   };
}

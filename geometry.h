#pragma once
//------------------------------------------------------------------------------------------------------------------------//
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <random>
//------------------------------------------------------------------------------------------------------------------------//
#define SIMD_V2_32_ALIGN
#define SIMD_V2_64_ALIGN
#if defined(SIMD_TYPES_SSE) || defined(SIMD_TYPES_AVX)
#  define ALIGN8                          __declspec(align(8))
#  define ALIGN16 __declspec(intrin_type) __declspec(align(16))
#  define ALIGN32 __declspec(intrin_type) __declspec(align(32))
#  if defined(SIMD_TYPES_SSE)
#    include <xmmintrin.h> // SSE
#    include <emmintrin.h> // SSE 2
#    include <pmmintrin.h> // SSE 3
#    include <smmintrin.h> // SSE 4.1
#    if defined(SIMD_V2_32_SSE2)
#      undef SIMD_V2_32_ALIGN
#      define SIMD_V2_32_ALIGN ALIGN8
#    endif
#    if defined(SIMD_V2_64_SSE2)
#      undef SIMD_V2_64_ALIGN
#      define SIMD_V2_64_ALIGN ALIGN16
#    endif
#  endif
#  if defined(SIMD_TYPES_AVX)
#    include <immintrin.h> // AVX
#  endif
#endif
//------------------------------------------------------------------------------------------------------------------------//
namespace simd
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
         float vals[2];
      };
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V2f ZERO()  { return V2f(0.0f, 0.0f); }
      static inline V2f UNITX() { return V2f(1.0f, 0.0f); }
      static inline V2f UNITY() { return V2f(0.0f, 1.0f); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2f()                                                          { }
      inline V2f(const float x, const float y)   : x(x),        y(y)        { }
      inline V2f(const float s)                  : x(s),        y(s)        { }
      inline V2f(const double x, const double y) : x((float)x), y((float)y) { }
      inline V2f(const int x, const int y)       : x((float)x), y((float)y) { }
      //------------------------------------------------------------------------------------------------------------------------//
      inline float  operator [] (const size_t i) const { return vals[i];                }
      inline float& operator [] (const size_t i)       { return vals[i];                }
      inline bool   operator == (const V2f&   v) const { return (x == v.x && y == v.y); }
      inline bool   operator != (const V2f&   v) const { return (x != v.x || y != v.y); }
      inline bool   operator <  (const V2f&   v) const { return (x <  v.x && y <  v.y); }
      inline bool   operator <= (const V2f&   v) const { return (x <= v.x && y <= v.y); }
      inline bool   operator >  (const V2f&   v) const { return (x >  v.x && y >  v.y); }
      inline bool   operator >= (const V2f&   v) const { return (x >= v.x && y >= v.y); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2f  operator +  (const float v[2]) const { return *this + V2f(v);         }
      inline V2f  operator -  (const float v[2]) const { return *this - V2f(v);         }
      inline V2f  operator *  (const float v[2]) const { return *this * V2f(v);         }
      inline V2f  operator /  (const float v[2]) const { return *this / V2f(v);         }
      inline V2f& operator =  (const float v[2])       { *this =  V2f(v); return *this; }
      inline V2f& operator += (const float v[2])       { *this += V2f(v); return *this; }
      inline V2f& operator -= (const float v[2])       { *this -= V2f(v); return *this; }
      inline V2f& operator *= (const float v[2])       { *this *= V2f(v); return *this; }
      inline V2f& operator /= (const float v[2])       { *this /= V2f(v); return *this; }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  isZero()                const { return x == 0.0f && y == 0.0f;                    }
      inline bool  isZero(const float e)   const { return x < e && x > -e && y < e && y > -e;        }
      inline bool  isNaN()                 const { return isnan(x) || isnan(y);                      }
      inline float dot(const V2f& v)       const { return x * v.x + y * v.y;                         }
      inline float cross(const V2f& v)     const { return x * v.y - y * v.x;                         }
      inline float length2()               const { return dot(*this);                                }
      inline float length()                const { return sqrtf(length2());                          }
      inline float distance2(const V2f& v) const { return (*this - v).length2();                     }
      inline float distance(const V2f& v)  const { return sqrtf(distance2(v));                       }
      inline V2f   yx()                    const { return V2f(y, x);                                 }
      inline void  yx()                          { std::swap(x, y);                                  }
      inline void  normalise()                   { *this /= length();                                }
      inline V2f   perp1()                       { return V2f( y,-x);                                }
      inline V2f   perp2()                       { return V2f(-y, x);                                }
      inline V2f   round()                 const { return V2f(roundf(x), roundf(y));                 }
      inline V2f   floor()                 const { return V2f(floorf(x), floorf(y));                 }
      inline V2f   ceil()                  const { return V2f(ceilf(x),  ceilf(y));                  }
      inline V2f   max(const V2f& v)       const { return V2f(v.x > x ? v.x : x, v.y > y ? v.y : y); }
      inline V2f   min(const V2f& v)       const { return V2f(v.x < x ? v.x : x, v.y < y ? v.y : y); }
      inline void  round()                       { x = roundf(x); y = roundf(y);                     }
      inline void  floor()                       { x = floorf(x); y = floorf(y);                     }
      inline void  ceil()                        { x = ceilf(x);  y = ceilf(y);                      }
      inline void  max(const V2f& v)             { if (v.x > x) x = v.x; if (v.y > y) y = v.y;       }
      inline void  min(const V2f& v)             { if (v.x < x) x = v.x; if (v.y < y) y = v.y;       }
      //------------------------------------------------------------------------------------------------------------------------//
      inline float side(const V2f& s, const V2f& e)                      const { return (e - s).cross(*this - s);             }
      inline bool  inside(const V2f& min, const V2f& max)                const { return *this >= min     && *this <= max;     }
      inline bool  inside(const V2f& min, const V2f& max, const V2f& e)  const { return *this >= (min-e) && *this <= (max+e); }
      inline bool  inside(const V2f& m, const float r2)                  const { return (*this - m).length2() <= r2;          }
      inline bool  inside(const V2f& m, const float r2, const float e)   const { return (*this - m).length2() <= (r2+e);      }
      inline void  rotate(float r)
      {
         float cs = ::cosf(r);
         float sn = ::sinf(r);
         float p = x;
         x = p * cs - y * sn;
         y = p * sn + y * cs;
      }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V2f  random()                          { return V2f(std::rand(), std::rand());                   }
      static inline void random(V2f* v, const size_t size) { for (size_t i = 0; i < size; i++) v[i] = V2f::random(); }
      //------------------------------------------------------------------------------------------------------------------------//
#if defined(SIMD_V2_32_SSE2)
      inline __m128 load()                const { return _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals)); }
      inline void   store(const __m128 v) const { _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(v));    }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2f(const float values[2]) { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values)); }
      inline V2f(float* const values)   { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values)); }
      inline V2f(const int values[2])   { store(_mm_cvtepi32_ps(_mm_loadl_epi64((__m128i*)values)));           }
      inline V2f(const __m128 values)   { store(values);                                                       }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void* operator new  (size_t size)            { return _aligned_malloc(sizeof(V2f), 8);        }
      inline void* operator new[](size_t size)            { return _aligned_malloc(size * sizeof(V2f), 8); }
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
      inline void  swap(V2f& v) { __m128 t(load()); store(v.load()); v.store(t); }
      //------------------------------------------------------------------------------------------------------------------------//
#else
      inline V2f(const float values[2]) : x(values[0]), y(values[1])               { }
      inline V2f(float* const values)   : x(values[0]), y(values[1])               { }
      inline V2f(const int values[2])   : x((float)values[0]), y((float)values[1]) { }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void* operator new  (size_t size)            { return malloc(sizeof(V2f));        }
      inline void* operator new[](size_t size)            { return malloc(size * sizeof(V2f)); }
      //------------------------------------------------------------------------------------------------------------------------//
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
         double vals[2];
         #if defined(SIMD_V2_64_SSE2)
         __m128d simd;
         #endif
      };
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V2d ZERO()  { return V2d(0.0, 0.0); }
      static inline V2d UNITX() { return V2d(1.0, 0.0); }
      static inline V2d UNITY() { return V2d(0.0, 1.0); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2d() { }
      //------------------------------------------------------------------------------------------------------------------------//
      inline double  operator [] (const size_t i) const { return vals[i];                }
      inline double& operator [] (const size_t i)       { return vals[i];                }
      inline bool    operator == (const V2d&   v) const { return (x == v.x && y == v.y); }
      inline bool    operator != (const V2d&   v) const { return (x != v.x || y != v.y); }
      inline bool    operator <  (const V2d&   v) const { return (x <  v.x && y <  v.y); }
      inline bool    operator <= (const V2d&   v) const { return (x <= v.x && y <= v.y); }
      inline bool    operator >  (const V2d&   v) const { return (x >  v.x && y >  v.y); }
      inline bool    operator >= (const V2d&   v) const { return (x >= v.x && y >= v.y); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2d  operator +  (const double v[2]) const { return *this + V2d(v);         }
      inline V2d  operator -  (const double v[2]) const { return *this - V2d(v);         }
      inline V2d  operator *  (const double v[2]) const { return *this * V2d(v);         }
      inline V2d  operator /  (const double v[2]) const { return *this / V2d(v);         }
      inline V2d& operator =  (const double v[2])       { *this =  V2d(v); return *this; }
      inline V2d& operator += (const double v[2])       { *this += V2d(v); return *this; }
      inline V2d& operator -= (const double v[2])       { *this -= V2d(v); return *this; }
      inline V2d& operator *= (const double v[2])       { *this *= V2d(v); return *this; }
      inline V2d& operator /= (const double v[2])       { *this /= V2d(v); return *this; }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool   isZero()                const { return x == 0.0 && y == 0.0;                      }
      inline bool   isZero(const double e)  const { return x < e && x > -e && y < e && y > -e;        }
      inline bool   isNaN()                 const { return isnan(x) || isnan(y);                      }
      inline double dot(const V2d& v)       const { return x * v.x + y * v.y;                         }
      inline double cross(const V2d& v)     const { return x * v.y - y * v.x;                         }
      inline double length2()               const { return dot(*this);                                }
      inline double length()                const { return sqrt(length2());                           }
      inline double distance2(const V2d& v) const { return (*this - v).length2();                     }
      inline double distance(const V2d& v)  const { return sqrt(distance2(v));                        }
      inline V2d    yx()                    const { return V2d(y, x);                                 }
      inline void   yx()                          { std::swap(x, y);                                  }
      inline void   normalise()                   { *this /= length();                                }
      inline V2d    perp1()                       { return V2d(y, -x);                                }
      inline V2d    perp2()                       { return V2d(-y, x);                                }
      //------------------------------------------------------------------------------------------------------------------------//
      inline double side(const V2d& s, const V2d& e)                      const { return (e - s).cross(*this - s);             }
      inline bool   inside(const V2d& min, const V2d& max)                const { return *this >= min && *this <= max;         }
      inline bool   inside(const V2d& min, const V2d& max, const V2d& e)  const { return *this >= (min-e) && *this <= (max+e); }
      inline bool   inside(const V2d& m, const double r2)                 const { return (*this - m).length2() <= r2;          }
      inline bool   inside(const V2d& m, const double r2, const double e) const { return (*this - m).length2() <= (r2 + e);    }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V2d  random()                          { return V2d(std::rand(), std::rand());                   }
      static inline void random(V2d* v, const size_t size) { for (size_t i = 0; i < size; i++) v[i] = V2d::random(); }
      //------------------------------------------------------------------------------------------------------------------------//
#if defined(SIMD_V2_64_SSE2)
      inline V2d(const double fX, const double fY) : simd(_mm_set_pd(fY, fX))                                 { }
      inline V2d(const float fX, const float fY)   : simd(_mm_set_pd((double)fY, (double)fX))                 { }
      inline V2d(const int fX, const int fY)       : simd(_mm_set_pd((double)fY, (double)fX))                 { }
      inline V2d(const double scalar)              : simd(_mm_set1_pd(scalar))                                { }
      inline V2d(const double values[2])           : simd(_mm_loadu_pd(values))                               { }
      inline V2d(double* const values)             : simd(_mm_loadu_pd(values))                               { }
      inline V2d(const int values[2])              : simd(_mm_cvtepi32_pd(_mm_loadl_epi64((__m128i*)values))) { }
      inline V2d(const __m128d values)             : simd(values)                                             { }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void* operator new  (size_t size)             { return _aligned_malloc(sizeof(V2d), 16);       }
      inline void* operator new[](size_t size)             { return _aligned_malloc(size* sizeof(V2d), 16); }
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
      inline void swap(V2d& v)            { __m128d t(simd); simd = v.simd; v.simd = t; }
      inline V2d  max(const V2d& v) const { return V2d(_mm_max_pd(simd, v.simd));       }
      inline V2d  min(const V2d& v) const { return V2d(_mm_min_pd(simd, v.simd));       }
      inline void max(const V2d& v)       { simd = _mm_max_pd(simd, v.simd);            }
      inline void min(const V2d& v)       { simd = _mm_min_pd(simd, v.simd);            }
#if defined(SIMD_V2_64_SSE41)
      inline V2d  round() const { return V2d(_mm_round_pd(simd, _MM_FROUND_NINT));  }
      inline V2d  floor() const { return V2d(_mm_round_pd(simd, _MM_FROUND_FLOOR)); }
      inline V2d  ceil()  const { return V2d(_mm_round_pd(simd, _MM_FROUND_CEIL));  }
      inline void round()       { simd = _mm_round_pd(simd, _MM_FROUND_NINT);       }
      inline void floor()       { simd = _mm_round_pd(simd, _MM_FROUND_FLOOR);      }
      inline void ceil()        { simd = _mm_round_pd(simd, _MM_FROUND_CEIL);       }
#else
      inline V2d  round() const { return V2d(::round(x), ::round(y)); }
      inline V2d  floor() const { return V2d(::floor(x), ::floor(y)); }
      inline V2d  ceil()  const { return V2d(::ceil(x),  ::ceil(y));  }
      inline void round()       { x = ::round(x); y = ::round(y);     }
      inline void floor()       { x = ::floor(x); y = ::floor(y);     }
      inline void ceil()        { x = ::ceil(x);  y = ::ceil(y);      }
#endif
      inline void rotate(double r)
      {
         __m128d cs(_mm_set1_pd(::cos(r)));
         __m128d sn(_mm_set1_pd(::sin(r)));
         __m128d p(_mm_set_pd(x, -y));
         simd = _mm_add_pd(_mm_mul_pd(simd, cs), _mm_mul_pd(p, sn));
   }
      //------------------------------------------------------------------------------------------------------------------------//
#else
      inline V2d(const double x, const double y) : x(x),         y(y)                         { }
      inline V2d(const double scalar)            : x(scalar),    y(scalar)                    { }
      inline V2d(const double values[2])         : x(values[0]), y(values[1])                 { }
      inline V2d(double* const values)           : x(values[0]), y(values[1])                 { }
      inline V2d(const int values[2])            : x((double)values[0]), y((double)values[1]) { }
      inline V2d(const float x, const float y)   : x((double)x), y((double)y)                 { }
      inline V2d(const int x, const int y)       : x((double)x), y((double)y)                 { }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void* operator new  (size_t size)             { return malloc(sizeof(V2d));        }
      inline void* operator new[](size_t size)             { return malloc(size * sizeof(V2d)); }
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
      inline void swap(V2d& v)            { std::swap(x, v.x); std::swap(y, v.y);             }
      inline V2d  max(const V2d& v) const { return V2d(v.x > x ? v.x : x, v.y > y ? v.y : y); }
      inline V2d  min(const V2d& v) const { return V2d(v.x < x ? v.x : x, v.y < y ? v.y : y); }
      inline void max(const V2d& v)       { if (v.x > x) x = v.x; if (v.y > y) y = v.y;       }
      inline void min(const V2d& v)       { if (v.x < x) x = v.x; if (v.y < y) y = v.y;       }
      inline V2d  round()           const { return V2d(::round(x), ::round(y));               }
      inline V2d  floor()           const { return V2d(::floor(x), ::floor(y));               }
      inline V2d  ceil()            const { return V2d(::ceil(x), ::ceil(y));                 }
      inline void round()                 { x = ::round(x); y = ::round(y);                   }
      inline void floor()                 { x = ::floor(x); y = ::floor(y);                   }
      inline void ceil()                  { x = ::ceil(x);  y = ::ceil(y);                    }
      inline void rotate(double r)
      {
         double cs = ::cos(r);
         double sn = ::sin(r);
         double p = x;
         x = p * cs - y * sn;
         y = p * sn + y * cs;
      }
      //------------------------------------------------------------------------------------------------------------------------//
#endif
   };
}

#pragma once
//------------------------------------------------------------------------------------------------------------------------//
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <random>
#define TWOPI (2.0*M_PI)
//------------------------------------------------------------------------------------------------------------------------//

#if defined(SIMD_V2_FP_32_SSE41) && !defined(SIMD_V2_FP_32_SSE2)
# define SIMD_V2_FP_32_SSE2
#endif
#if defined(SIMD_V2_FP_64_SSE41) && !defined(SIMD_V2_FP_64_SSE2)
# define SIMD_V2_FP_64_SSE2
#endif

#if defined(SIMD_TYPES_SSE) || defined(SIMD_TYPES_AVX)
#  define ALIGN8                          __declspec(align(8))
#  define ALIGN16 __declspec(intrin_type) __declspec(align(16))
#  define ALIGN32 __declspec(intrin_type) __declspec(align(32))
#  if defined(SIMD_TYPES_SSE)
#    include <xmmintrin.h> // SSE
#    include <emmintrin.h> // SSE 2
#    include <pmmintrin.h> // SSE 3
#    include <smmintrin.h> // SSE 4.1
#  endif
#  if defined(SIMD_TYPES_AVX)
#    include <immintrin.h> // AVX
#  endif
#endif
//------------------------------------------------------------------------------------------------------------------------//
namespace simd
{
   //------------------------------------------------------------------------------------------------------------------------//
   //                                     ROOT TEMPLATE ALL VECTORS [L0, ABSTRACT]                                           //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract Vector Base Template without dimension
   /// </summary>
   template <typename V, typename F>
   class VB abstract
   {
   public:
      inline void* operator new  (size_t size)          { return malloc(sizeof(V));        }
      inline void* operator new[](size_t size)          { return malloc(size * sizeof(V)); }
      inline V     operator []   (const size_t i) const { return vals[i];                  }
      inline V&    operator []   (const size_t i)       { return vals[i];                  }
   };

#pragma region V2
   //------------------------------------------------------------------------------------------------------------------------//
   //                                          ROOT TEMPLATE V2 [L1, ABSTRACT]                                               //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 2D Vector Template for Floating Point (32/64) AND Integer (32/64). [L1]
   /// </summary>
   template <typename V, typename F>
   class V2 abstract : public VB<V, F>
   {
   protected:
      inline V*   thiss() const { return (V*)this; }

   public:
      union
      {
         struct { F x, y; };
         F vals[2];
      };
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2()                                                  { }
      inline V2(const F x, const F y) : x(x),         y(y)         { }
      inline V2(const F scalar)       : x(scalar),    y(scalar)    { }
      inline V2(const F values[2])    : x(values[0]), y(values[1]) { }
      inline V2(F* const values)      : x(values[0]), y(values[1]) { }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V ZERO()  { return V((F)0.0, (F)0.0); }
      static inline V UNITX() { return V((F)1.0, (F)0.0); }
      static inline V UNITY() { return V((F)0.0, (F)1.0); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  operator ==   (const V& v)     const { return (x == v.x && y == v.y);        }
      inline bool  operator !=   (const V& v)     const { return (x != v.x || y != v.y);        }
      inline bool  operator <    (const V& v)     const { return (x <  v.x && y <  v.y);        }
      inline bool  operator <=   (const V& v)     const { return (x <= v.x && y <= v.y);        }
      inline bool  operator >    (const V& v)     const { return (x >  v.x && y >  v.y);        }
      inline bool  operator >=   (const V& v)     const { return (x >= v.x && y >= v.y);        }
      inline V     operator +    (const V& v)     const { return V(x + v.x, y + v.y);           }
      inline V     operator -    (const V& v)     const { return V(x - v.x, y - v.y);           }
      inline V     operator *    (const V& v)     const { return V(x * v.x, y * v.y);           }
      inline V     operator /    (const V& v)     const { return V(x / v.x, y / v.y);           }
      inline V     operator *    (const F  s)     const { return V(x * s,   y * s);             }
      inline V     operator /    (const F  s)     const { return V(x / s,   y / s);             }
      inline V&    operator =    (const V& v)           { x =  v.x; y =  v.y; return *thiss();  }
      inline V&    operator +=   (const V& v)           { x += v.x; y += v.y; return *thiss();  }
      inline V&    operator -=   (const V& v)           { x -= v.x; y -= v.y; return *thiss();  }
      inline V&    operator *=   (const V& v)           { x *= v.x; y *= v.y; return *thiss();  }
      inline V&    operator /=   (const V& v)           { x /= v.x; y /= v.y; return *thiss();  }
      inline V&    operator =    (const F  s)           { x =  s;   y =  s;   return *thiss();  }
      inline V&    operator +=   (const F  s)           { x += s;   y += s;   return *thiss();  }
      inline V&    operator -=   (const F  s)           { x -= s;   y -= s;   return *thiss();  }
      inline V&    operator *=   (const F  s)           { x *= s;   y *= s;   return *thiss();  }
      inline V&    operator /=   (const F  s)           { x /= s;   y /= s;   return *thiss();  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline       V  operator - ()               const { return V(-x, -y);      }
      inline const V& operator + ()               const { return *this;          }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool isZero()                         const { return x == (F)0.0 && y == (F)0.0;              }
      inline bool isZero(const F e2)               const { return thiss()->length2() <= e2;                }
      inline bool equals(const V& v, const F e2)   const { return (*thiss() - v).length2() <= e2;          }
      inline void swap(V& v)                             { std::swap(x, v.x); std::swap(y, v.y);           }
      inline F    cross(const V& v)                const { return x * v.y - y * v.x;                       }
      inline F    dot(const V& v)                  const { return x * v.x + y * v.y;                       }
      inline F    length2()                        const { return thiss()->dot(*thiss());                  }
      inline F    length()                         const { return V::_sqrt(thiss()->length2());            }
      inline F    distance2(const V& v)            const { return (*thiss() - v).length2();                }
      inline F    distance(const V& v)             const { return V::_sqrt(thiss()->distance2(v));         }
      inline V    yx()                             const { return V(y, x);                                 }
      inline void yx()                                   { std::swap(x, y);                                }
      inline V    maxC(const V& v)                 const { return V(v.x > x ? v.x : x, v.y > y ? v.y : y); }
      inline V    minC(const V& v)                 const { return V(v.x < x ? v.x : x, v.y < y ? v.y : y); }
      inline V    boundC(const V& mi, const V& ma) const { V t(thiss()->minC(ma)); t.max(mi); return t;    }
      inline V    absC()                           const { return V(V::_abs(x), V::_abs(y));               }
      inline void max(const V& v)                        { if (v.x > x) x = v.x; if (v.y > y) y = v.y;     }
      inline void min(const V& v)                        { if (v.x < x) x = v.x; if (v.y < y) y = v.y;     }
      inline void bound(const V& mi, const V& ma)        { thiss()->min(ma); thiss()->max(mi);             }
      inline void abs()                                  { x = V::_abs(x); y = V::_abs(y);                 }
      inline V    perp1()                          const { return V(y, -x);                                }
      inline V    perp2()                          const { return V(-y, x);                                }
      //------------------------------------------------------------------------------------------------------------------------//
      inline F    side(const V& s, const V& e)                  const { return (e - s).cross(*thiss() - s);                    }
      inline bool inside(const V& min, const V& max)            const { return *thiss() >= min       && *thiss() <= max;       }
      inline bool inside(const V& min, const V& max, const F e) const { return *thiss() >= (min - e) && *thiss() <= (max + e); }
      inline bool inside(const V& m, const F r2)                const { return thiss()->distance2(m) <= r2;                    }
      inline bool inside(const V& m, const F r2, const F e)     const { return thiss()->distance2(m) <= (r2 + e);              }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V    random()                         { return V(std::rand(), std::rand());                    }
      static inline void random(V* v, const size_t size)  { for (size_t i = 0; i < size; i++) v[i] = V::random();  }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                 FLOATING POINT & INTEGER TEMPLATES [L2, ABSTRACT]                                      //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 2D Vector Template for Floating Point (32/64) [L2]
   /// </summary>
   template <typename V, typename F>
   class V2fdt abstract : public V2<V, F>
   {
   public:
      inline V2fdt() { }
      inline V2fdt(const F x, const F y) : V2(x, y)                 { }
      inline V2fdt(const F scalar)       : V2(scalar, scalar)       { }
      inline V2fdt(const F values[2])    : V2(values[0], values[1]) { }
      inline V2fdt(F* const values)      : V2(values[0], values[1]) { }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V& operator /= (const V& v)       { x /= v.x; y /= v.y; return (V&)*this;               }
      inline V  operator /  (const V& v) const { return V(x / v.x, y / v.y);                         }
      inline V& operator /= (const F  s)       { F t = (F)1.0 / s; x *= t; y *= t; return (V&)*this; }
      inline V  operator /  (const F  s) const { F t = (F)1.0 / s; return V(x * t, y * t);           }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  isNaN()                              const { return isnan<F>(x) || isnan<F>(y);     }
      inline void  normalise()                                { *thiss() /= thiss()->length();         }
      inline V     normaliseC()                         const { V t(*thiss()); t.normalise(); return t; }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V     roundC()                             const { return V(V::_round(x), V::_round(y)); }
      inline V     floorC()                             const { return V(V::_floor(x), V::_floor(y)); }
      inline V     ceilC()                              const { return V(V::_ceil(x),  V::_ceil(y));  }
      inline void  round()                                    { x = V::_round(x); y = V::_round(y);   }
      inline void  floor()                                    { x = V::_floor(x); y = V::_floor(y);   }
      inline void  ceil()                                     { x = V::_ceil(x);  y = V::_ceil(y);    }
      //------------------------------------------------------------------------------------------------------------------------//
      inline F angle()              const { return V::_acos(x / thiss()->length());                                       }
      inline F angleNoN()           const { return V::_acos(x);                                                           }
      inline F angleOri()           const { F t = thiss()->angle();    if (y < (F)0.0) t = (F)TWOPI - t; return t;        }
      inline F angleOriNoN()        const { F t = thiss()->angleNoN(); if (y < (F)0.0) t = (F)TWOPI - t; return t;        }
      inline F angle(const V& v)    const { F lp = thiss()->length() * v.length(); return V::_acos(thiss()->dot(v) / lp); }
      inline F angleOri(const V& v) const { F t = thiss()->angle(v); if (thiss()->cross(v) < (F)0.0) t = (F)TWOPI - t; return t;   }
      //------------------------------------------------------------------------------------------------------------------------//
      inline F    area(const V& p, const V& q)                  const { return (F)0.5 * (p - *thiss()).cross(q - *thiss()); }
      inline void rotate(F r)
      {
         F cs = V::_cos(r);
         F sn = V::_sin(r);
         F p = x;
         x = p * cs - y * sn;
         y = p * sn + y * cs;
      }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V    randomN()                        { V t(V::random()); t.normalise(); return t;             }
      static inline void randomN(V* v, const size_t size) { for (size_t i = 0; i < size; i++) v[i] = V::randomN(); }
      //------------------------------------------------------------------------------------------------------------------------//
   };

   /// <summary>
   /// Abstract 2D Vector for Integer (32/64) [L2]
   /// </summary>
   template <typename V, typename F>
   class V2ilt abstract : public V2<V, F>
   {
   public:
      static inline F _abs(const int s)  { return ::abs(s);         }
      static inline F _sqrt(const int s) { return (F)::sqrt<F>(s);  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2ilt()                                { }
      inline V2ilt(const F x, const F y) : V2(x, y) { }
      inline V2ilt(const F s)            : V2(s, s) { }
      inline V2ilt(const F v[2])         : V2(v)    { }
      inline V2ilt(F* const v)           : V2(v)    { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                       32-BIT & 64-BIT TEMPLATES [L3, ABSTRACT]                                         //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 2D Vector Template for Single Precision FP [L3]
   /// </summary>
   template <typename V>
   class V2ft abstract : public V2fdt<V, float>
   {
   public:
      static inline float _abs(const float s)   { return ::fabsf(s);  }
      static inline float _round(const float s) { return ::roundf(s); }
      static inline float _floor(const float s) { return ::floorf(s); }
      static inline float _ceil(const float s)  { return ::ceilf(s);  }
      static inline float _sqrt(const float s)  { return ::sqrtf(s);  }
      static inline float _cos(const float s)   { return ::cosf(s);   }
      static inline float _sin(const float s)   { return ::sinf(s);   }
      static inline float _acos(const float s)  { return ::acosf(s);  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2ft() { }
      inline V2ft(const float x, const float y) : V2fdt(x, y) { }
      inline V2ft(const float s)                : V2fdt(s, s) { }
      inline V2ft(const float v[2])             : V2fdt(v)    { }
      inline V2ft(float* const v)               : V2fdt(v)    { }
   };

   /// <summary>
   /// Abstract 2D Vector Template for Double Precision FP [L3]
   /// </summary>
   template <typename V>
   class V2dt abstract : public V2fdt<V, double>
   {
   public:
      static inline double _abs(const double s)   { return ::abs(s);   }
      static inline double _round(const double s) { return ::round(s); }
      static inline double _floor(const double s) { return ::floor(s); }
      static inline double _ceil(const double s)  { return ::ceil(s);  }
      static inline double _sqrt(const double s)  { return ::sqrt(s);  }
      static inline double _cos(const double s)   { return ::cos(s);   }
      static inline double _sin(const double s)   { return ::sin(s);   }
      static inline double _acos(const double s)  { return ::acos(s);  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2dt() { }
      inline V2dt(const double x, const double y) : V2fdt(x, y) { }
      inline V2dt(const double s)                 : V2fdt(s, s) { }
      inline V2dt(const double v[2])              : V2fdt(v)    { }
      inline V2dt(double* const v)                : V2fdt(v)    { }
   };

   /// <summary>
   /// Abstract 2D Vector Template for Integer (32) [L3]
   /// </summary>
   template <typename V>
   class V2it abstract : public V2ilt<V, int>
   {
   public:
      inline V2it()                                       { }
      inline V2it(const int x, const int y) : V2ilt(x, y) { }
      inline V2it(const int s)              : V2ilt(s, s) { }
      inline V2it(const int v[2])           : V2ilt(v)    { }
      inline V2it(int* const v)             : V2ilt(v)    { }
   };

   /// <summary>
   /// Abstract 2D Vector Template for Integer (64) [L3]
   /// </summary>
   template <typename V>
   class V2lt abstract : public V2ilt<V, long long>
   {
   public:
      inline V2lt() { }
      inline V2lt(const long long x, const long long y) : V2ilt(x, y) { }
      inline V2lt(const long long s)                    : V2ilt(s, s) { }
      inline V2lt(const long long v[2])                 : V2ilt(v)    { }
      inline V2lt(long long* const v)                   : V2ilt(v)    { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                     GENERIC/NON-SIMD CLASSES [L4]                                                      //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Single Precision 2D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V2fg : public V2ft<V2fg>
   {
   public:
      inline V2fg()                                                                { }
      inline V2fg(const float x, const float y)   : V2ft(x, y)                     { }
      inline V2fg(const float s)                  : V2ft(s, s)                     { }
      inline V2fg(const float v[2])               : V2ft(v)                        { }
      inline V2fg(float* const v)                 : V2ft(v)                        { }
      inline V2fg(const int v[2])                 : V2ft((float)v[0], (float)v[1]) { }
      inline V2fg(const double x, const double y) : V2ft((float)x,    (float)y)    { }
      inline V2fg(const int x, const int y)       : V2ft((float)x,    (float)y)    { }
   };

   /// <summary>
   /// Double Precision 2D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V2dg : public V2dt<V2dg>
   {
   public:
      inline V2dg()                                                                  { }
      inline V2dg(const double x, const double y) : V2dt(x, y)                       { }
      inline V2dg(const double s)                 : V2dt(s, s)                       { }
      inline V2dg(const double v[2])              : V2dt(v)                          { }
      inline V2dg(double* const v)                : V2dt(v)                          { }
      inline V2dg(const int v[2])                 : V2dt((double)v[0], (double)v[1]) { }
      inline V2dg(const float x, const float y)   : V2dt((double)x,    (double)y)    { }
      inline V2dg(const int x, const int y)       : V2dt((double)x,    (double)y)    { }
   };

   /// <summary>
   /// 32-Bit Integer 2D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V2ig : public V2it<V2ig>
   {
   public:
      inline V2ig() { }
      inline V2ig(const int x, const int y) : V2it(x, y) { }
      inline V2ig(const int s)              : V2it(s, s) { }
      inline V2ig(const int v[2])           : V2it(v)    { }
      inline V2ig(int* const v)             : V2it(v)    { }
   };

   /// <summary>
   /// 64-Bit Integer 2D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V2lg : public V2lt<V2lg>
   {
   public:
      inline V2lg() { }
      inline V2lg(const long long x, const long long y) : V2lt(x, y) { }
      inline V2lg(const long long s)                    : V2lt(s, s) { }
      inline V2lg(const long long v[2])                 : V2lt(v)    { }
      inline V2lg(long long* const v)                   : V2lt(v)    { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                             SIMD CLASSES [L4]                                                          //
   //------------------------------------------------------------------------------------------------------------------------//
#if defined(SIMD_V2_FP_32_SSE2)
   /// <summary>
   /// Single Precision 2D Vector (SSE/SIMD)
   /// </summary>
   ALIGN8 class V2fs : public V2ft<V2fs>
   {
   public:
      inline __m128 load()                const { return _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals)); }
      inline void   store(const __m128 v) const { _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(v));    }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2fs()                                                          { }
      inline V2fs(const float x, const float y)   : V2ft(x, y)               { }
      inline V2fs(const float s)                  : V2ft(s, s)               { }
      inline V2fs(const double x, const double y) : V2ft((float)x, (float)y) { }
      inline V2fs(const int x, const int y)       : V2ft((float)x, (float)y) { }
      inline V2fs(const float values[2])          { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values)); }
      inline V2fs(float* const values)            { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values)); }
      inline V2fs(const int values[2])            { store(_mm_cvtepi32_ps(_mm_loadl_epi64((__m128i*)values)));           }
      inline V2fs(const __m128 values)            { store(values);                                                       }
      //------------------------------------------------------------------------------------------------------------------------//
      inline       void* operator new  (size_t size)        { return _aligned_malloc(sizeof(V2fs), 8);                        }
      inline       void* operator new[](size_t size)        { return _aligned_malloc(size * sizeof(V2fs), 8);                 }
      inline       bool  operator == (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmpeq_ps(load(), v.load())) == 0x0F; }
      inline       bool  operator != (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmpeq_ps(load(), v.load())) != 0x00; }
      inline       bool  operator <  (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmplt_ps(load(), v.load())) == 0x0F; }
      inline       bool  operator <= (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmple_ps(load(), v.load())) == 0x0F; }
      inline       bool  operator >  (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmpgt_ps(load(), v.load())) == 0x0F; }
      inline       bool  operator >= (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmpge_ps(load(), v.load())) == 0x0F; }
      inline       V2fs  operator +  (const V2fs&  v) const { return V2fs(_mm_add_ps(load(), v.load()));                      }
      inline       V2fs  operator -  (const V2fs&  v) const { return V2fs(_mm_sub_ps(load(), v.load()));                      }
      inline       V2fs  operator *  (const V2fs&  v) const { return V2fs(_mm_mul_ps(load(), v.load()));                      }
      inline       V2fs  operator /  (const V2fs&  v) const { return V2fs(_mm_div_ps(load(), v.load()));                      }
      inline       V2fs  operator *  (const float  s) const { return V2fs(_mm_mul_ps(load(), _mm_set1_ps(s)));                }
      inline       V2fs  operator /  (const float  s) const { return V2fs(_mm_div_ps(load(), _mm_set1_ps(s)));                }
      inline       V2fs  operator -  ()               const { return V2fs(_mm_sub_ps(_mm_setzero_ps(), load()));              }
      inline const V2fs& operator +  ()               const { return *this;                                                   }
      inline       V2fs& operator =  (const V2fs&  v)       { store(v.load());                           return *this;        }
      inline       V2fs& operator += (const V2fs&  v)       { store(_mm_add_ps(load(), v.load()));       return *this;        }
      inline       V2fs& operator -= (const V2fs&  v)       { store(_mm_sub_ps(load(), v.load()));       return *this;        }
      inline       V2fs& operator *= (const V2fs&  v)       { store(_mm_mul_ps(load(), v.load()));       return *this;        }
      inline       V2fs& operator /= (const V2fs&  v)       { store(_mm_div_ps(load(), v.load()));       return *this;        }
      inline       V2fs& operator =  (const float  s)       { store(_mm_set1_ps(s));                     return *this;        }
      inline       V2fs& operator += (const float  s)       { store(_mm_add_ps(load(), _mm_set1_ps(s))); return *this;        }
      inline       V2fs& operator -= (const float  s)       { store(_mm_sub_ps(load(), _mm_set1_ps(s))); return *this;        }
      inline       V2fs& operator *= (const float  s)       { store(_mm_mul_ps(load(), _mm_set1_ps(s))); return *this;        }
      inline       V2fs& operator /= (const float  s)       { store(_mm_div_ps(load(), _mm_set1_ps(s))); return *this;        }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void swap(V2fs& v)                                { __m128 t(load()); store(v.load()); v.store(t);         }
      inline V2fs absC()                                 const { return V2fs(_mm_andnot_ps(_mm_set1_ps(-0.f), load())); }
      inline V2fs maxC(const V2fs& v)                    const { return V2fs(_mm_max_ps(load(), v.load()));             }
      inline V2fs minC(const V2fs& v)                    const { return V2fs(_mm_min_ps(load(), v.load()));             }
      inline V2fs boundC(const V2fs& mi, const V2fs& ma) const { V2fs t(minC(ma)); t.max(mi); return t;                 }
      inline void abs()                                        { store(_mm_andnot_ps(_mm_set1_ps(-0.), load()));        }
      inline void max(const V2fs& v)                           { store(_mm_max_ps(load(), v.load()));                   }
      inline void min(const V2fs& v)                           { store(_mm_min_ps(load(), v.load()));                   }
      inline void bound(const V2fs& mi, const V2fs& ma)        { min(ma); max(mi);                                      }
      //------------------------------------------------------------------------------------------------------------------------//
      inline float dot(const V2fs& v) const
      {
         __m128 a(_mm_mul_ps(load(), v.load()));
         __m128 b(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 c(_mm_add_ss(a, b));
         return c.m128_f32[0];
      }
      inline float length() const
      {
         __m128 t(load());
         __m128 a(_mm_mul_ps(t, t));
         __m128 b(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 c(_mm_add_ss(a, b));
         __m128 d(_mm_sqrt_ss(c));
         return d.m128_f32[0];
      }
      inline float side(const V2fs& s, const V2fs& e) const
      {
         __m128 t(s.load());
         __m128 a(_mm_sub_ps(e.load(), t));
         __m128 b(_mm_sub_ps(load(), t));
         __m128 c(_mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 d(_mm_mul_ps(a, c));
         __m128 f(_mm_shuffle_ps(d, d, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 g(_mm_sub_ss(d, f));
         return g.m128_f32[0];
      }
      inline bool inside(const V2fs& min, const V2fs& max) const
      {
         __m128 a(load());
         __m128 b(_mm_cmpge_ps(a, min.load()));
         __m128 c(_mm_cmple_ps(a, max.load()));
         __m128 d(_mm_and_ps(b, c));
         return _mm_movemask_ps(d) == 0x0F;
      }
      inline bool inside(const V2fs& min, const V2fs& max, const float e) const
      {
         __m128 eps(_mm_set1_ps(e));
         __m128 a(load());
         __m128 b(_mm_cmpge_ps(a, _mm_sub_ps(min.load(), eps)));
         __m128 c(_mm_cmple_ps(a, _mm_add_ps(max.load(), eps)));
         __m128 d(_mm_and_ps(b, c));
         return _mm_movemask_ps(d) == 0x0F;
      }
      inline float area(const V2fs& p, const V2fs& q) const
      {
         __m128 t(load());
         __m128 a(_mm_sub_ps(p.load(), t));
         __m128 b(_mm_sub_ps(q.load(), t));
         __m128 c(_mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 d(_mm_mul_ps(a, c));
         __m128 e(_mm_shuffle_ps(d, d, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 f(_mm_sub_ss(d, e));
         return 0.5f * f.m128_f32[0];
      }
#if defined(SIMD_V2_FP_32_SSE41)
      inline V2fs  roundC() const { return V2fs(_mm_round_ps(load(), _MM_FROUND_NINT)); }
      inline V2fs  floorC() const { return V2fs(_mm_round_ps(load(), _MM_FROUND_FLOOR)); }
      inline V2fs  ceilC()  const { return V2fs(_mm_round_ps(load(), _MM_FROUND_CEIL)); }
      inline void  round()        { store(_mm_round_ps(load(), _MM_FROUND_NINT)); }
      inline void  floor()        { store(_mm_round_ps(load(), _MM_FROUND_FLOOR)); }
      inline void  ceil()         { store(_mm_round_ps(load(), _MM_FROUND_CEIL)); }
#endif
      //------------------------------------------------------------------------------------------------------------------------//
   };
   typedef V2fs V2f;  // use SIMD as default
#else
   typedef V2fg V2f;  // use plain as default
#endif

   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//

#if defined(SIMD_V2_FP_64_SSE2)
   /// <summary>
   /// Double Precision 2D Vector
   /// </summary>
   ALIGN16 class V2ds : public V2dt<V2ds>
   {
   public:
      inline __m128d load()                 const { return _mm_load_pd(vals); }
      inline void    store(const __m128d v)       { _mm_store_pd(vals, v);    }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2ds() { }
      inline V2ds(const double fX, const double fY) { store(_mm_set_pd(fY, fX));                                 }
      inline V2ds(const float fX, const float fY)   { store(_mm_set_pd((double)fY, (double)fX));                 }
      inline V2ds(const int fX, const int fY)       { store(_mm_set_pd((double)fY, (double)fX));                 }
      inline V2ds(const double scalar)              { store(_mm_set1_pd(scalar));                                }
      inline V2ds(const double values[2])           { store(_mm_loadu_pd(values));                               }
      inline V2ds(double* const values)             { store(_mm_loadu_pd(values));                               }
      inline V2ds(const int values[2])              { store(_mm_cvtepi32_pd(_mm_loadl_epi64((__m128i*)values))); }
      inline V2ds(const __m128d values)             { store(values);                                             }
      //------------------------------------------------------------------------------------------------------------------------//
      inline       void* operator new  (size_t size)        { return _aligned_malloc(sizeof(V2ds), 16);                       }
      inline       void* operator new[](size_t size)        { return _aligned_malloc(size* sizeof(V2ds), 16);                 }
      inline       bool  operator == (const V2ds&  v) const { return _mm_movemask_pd(_mm_cmpeq_pd(load(), v.load())) == 0x03; }
      inline       bool  operator != (const V2ds&  v) const { return _mm_movemask_pd(_mm_cmpeq_pd(load(), v.load())) != 0x00; }
      inline       bool  operator <  (const V2ds&  v) const { return _mm_movemask_pd(_mm_cmplt_pd(load(), v.load())) == 0x03; }
      inline       bool  operator <= (const V2ds&  v) const { return _mm_movemask_pd(_mm_cmple_pd(load(), v.load())) == 0x03; }
      inline       bool  operator >  (const V2ds&  v) const { return _mm_movemask_pd(_mm_cmpgt_pd(load(), v.load())) == 0x03; }
      inline       bool  operator >= (const V2ds&  v) const { return _mm_movemask_pd(_mm_cmpge_pd(load(), v.load())) == 0x03; }
      inline       V2ds  operator +  (const V2ds&  v) const { return V2ds(_mm_add_pd(load(), v.load()));                      }
      inline       V2ds  operator -  (const V2ds&  v) const { return V2ds(_mm_sub_pd(load(), v.load()));                      }
      inline       V2ds  operator *  (const V2ds&  v) const { return V2ds(_mm_mul_pd(load(), v.load()));                      }
      inline       V2ds  operator /  (const V2ds&  v) const { return V2ds(_mm_div_pd(load(), v.load()));                      }
      inline       V2ds  operator *  (const double s) const { return V2ds(_mm_mul_pd(load(), _mm_set1_pd(s)));                }
      inline       V2ds  operator /  (const double s) const { return V2ds(_mm_div_pd(load(), _mm_set1_pd(s)));                }
      inline       V2ds  operator -  ()               const { return V2ds(_mm_sub_pd(_mm_setzero_pd(), load()));              }
      inline const V2ds& operator +  ()               const { return *this;                                                   }
      inline       V2ds& operator =  (const V2ds&  v)       { store(v.load());                           return *this;        }
      inline       V2ds& operator += (const V2ds&  v)       { store(_mm_add_pd(load(), v.load()));       return *this;        }
      inline       V2ds& operator -= (const V2ds&  v)       { store(_mm_sub_pd(load(), v.load()));       return *this;        }
      inline       V2ds& operator *= (const V2ds&  v)       { store(_mm_mul_pd(load(), v.load()));       return *this;        }
      inline       V2ds& operator /= (const V2ds&  v)       { store(_mm_div_pd(load(), v.load()));       return *this;        }
      inline       V2ds& operator =  (const double s)       { store(_mm_set1_pd(s));                     return *this;        }
      inline       V2ds& operator += (const double s)       { store(_mm_add_pd(load(), _mm_set1_pd(s))); return *this;        }
      inline       V2ds& operator -= (const double s)       { store(_mm_sub_pd(load(), _mm_set1_pd(s))); return *this;        }
      inline       V2ds& operator *= (const double s)       { store(_mm_mul_pd(load(), _mm_set1_pd(s))); return *this;        }
      inline       V2ds& operator /= (const double s)       { store(_mm_div_pd(load(), _mm_set1_pd(s))); return *this;        }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void swap(V2ds& v)                                { __m128d t(load()); store(v.load()); v.store(t);       }
      inline V2ds absC()                                 const { return V2ds(_mm_andnot_pd(_mm_set1_pd(-0.), load())); }
      inline V2ds maxC(const V2ds& v)                    const { return V2ds(_mm_max_pd(load(), v.load()));            }
      inline V2ds minC(const V2ds& v)                    const { return V2ds(_mm_min_pd(load(), v.load()));            }
      inline V2ds boundC(const V2ds& mi, const V2ds& ma) const { V2ds t(minC(ma)); t.max(mi); return t;                }
      inline void abs()                                        { store(_mm_andnot_pd(_mm_set1_pd(-0.), load()));       }
      inline void max(const V2ds& v)                           { store(_mm_max_pd(load(), v.load()));                  }
      inline void min(const V2ds& v)                           { store(_mm_min_pd(load(), v.load()));                  }
      inline void bound(const V2ds& mi, const V2ds& ma)        { min(ma); max(mi);                                     }
      inline void rotate(double r)
      {
         __m128d cs(_mm_set1_pd(_cos(r)));
         __m128d sn(_mm_set1_pd(_sin(r)));
         __m128d p(_mm_set_pd(x, -y));
         store(_mm_add_pd(_mm_mul_pd(load(), cs), _mm_mul_pd(p, sn)));
      }
      inline double cross(const V2ds& v) const
      {
         __m128d a(_mm_shuffle_pd(v.load(), v.load(), _MM_SHUFFLE2(0, 1)));
         __m128d b(_mm_mul_pd(load(), a));
         __m128d c(_mm_shuffle_pd(b, b, _MM_SHUFFLE2(0, 1)));
         __m128d d(_mm_sub_sd(b, c));
         return d.m128d_f64[0];
      }
      inline bool inside(const V2ds& min, const V2ds& max) const
      {
         __m128d a(_mm_cmpge_pd(load(), min.load()));
         __m128d b(_mm_cmple_pd(load(), max.load()));
         __m128d c(_mm_and_pd(a, b));
         return _mm_movemask_pd(c) == 0x03;
      }
      inline bool inside(const V2ds& min, const V2ds& max, const double e) const
      {
         __m128d eps(_mm_set1_pd(e));
         __m128d a(_mm_cmpge_pd(load(), _mm_sub_pd(min.load(), eps)));
         __m128d b(_mm_cmple_pd(load(), _mm_add_pd(max.load(), eps)));
         __m128d c(_mm_and_pd(a, b));
         return _mm_movemask_pd(c) == 0x03;
      }
#if defined(SIMD_V2_FP_64_SSE41)
      inline double dot(const V2ds& v) const { return _mm_dp_pd(load(), v.load(), 0x31).m128d_f64[0];                            }
      inline double length()           const { __m128d t(load()); return _mm_sqrt_pd(_mm_dp_pd(t, t, 0x31)).m128d_f64[0];        }
      inline V2ds   roundC()           const { return V2ds(_mm_round_pd(load(), _MM_FROUND_NINT));                               }
      inline V2ds   floorC()           const { return V2ds(_mm_round_pd(load(), _MM_FROUND_FLOOR));                              }
      inline V2ds   ceilC()            const { return V2ds(_mm_round_pd(load(), _MM_FROUND_CEIL));                               }
      inline void   round()                  { store(_mm_round_pd(load(), _MM_FROUND_NINT));                                     }
      inline void   floor()                  { store(_mm_round_pd(load(), _MM_FROUND_FLOOR));                                    }
      inline void   ceil()                   { store(_mm_round_pd(load(), _MM_FROUND_CEIL));                                     }
      inline void   normalise()              { __m128d t(load()); store(_mm_div_pd(load(), _mm_sqrt_pd(_mm_dp_pd(t, t, 0x33)))); }
#else
      inline double dot(const V2ds& v) const 
      {
         __m128d a(_mm_mul_pd(load(), v.load()));
         __m128d b(_mm_castps_pd(_mm_movehl_ps(_mm_undefined_ps(), _mm_castpd_ps(a))));
         __m128d c(_mm_add_sd(a, b));
         return c.m128d_f64[0];
      }
      inline double length() const 
      { 
         __m128d t(load());
         __m128d a(_mm_mul_pd(t, t));
         __m128d b(_mm_castps_pd(_mm_movehl_ps(_mm_undefined_ps(), _mm_castpd_ps(a))));
         __m128d c(_mm_add_sd(a, b));
         __m128d d(_mm_sqrt_sd(c, c));
         return d.m128d_f64[0];
      }
      inline void   normalise()
      { 
         __m128d t(load());
         __m128d a(_mm_mul_pd(t, t));
         __m128d b(_mm_shuffle_pd(a, a, _MM_SHUFFLE2(0,1)));
         __m128d c(_mm_add_pd(a, b));
         __m128d d(_mm_sqrt_pd(c));
         store(_mm_div_pd(t, d));
      }
#endif
      //------------------------------------------------------------------------------------------------------------------------//
   };
   typedef V2ds V2d;  // use SIMD as default
#else
   typedef V2dg V2d;  // use plain as default
#endif
   
   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//

#if defined(SIMD_V2_INT_32_SSE2)
   // TODO INT32 SSE
   typedef V2is V2i;  // use SIMD as default
#else
   typedef V2ig V2i;  // use plain as default
#endif

#if defined(SIMD_V2_INT_64_SSE2)
   // TODO INT64 SSE
   typedef V2ls V2l;  // use SIMD as default
#else
   typedef V2lg V2l;  // use plain as default
#endif

#pragma endregion

   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//

#pragma region V3
   //------------------------------------------------------------------------------------------------------------------------//
   //                                          ROOT TEMPLATE [L1, ABSTRACT]                                                  //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 3D Vector Template for Floating Point (32/64) AND Integer (32/64). [L1]
   /// </summary>
   template <typename V, typename F>
   class V3 abstract : public VB<V, F>
   {
   protected:
      inline V*   thiss() const { return ((V*)this); }

   public:
      union
      {
         struct { F x, y, z; };
         F vals[3];
      };
      //------------------------------------------------------------------------------------------------------------------------//
      inline V3()                                                            { }
      inline V3(const F x, const F y, const F z) : x(x),    y(y),    z(z)    { }
      inline V3(const F s)                       : x(s),    y(s),    z(s)    { }
      inline V3(const F v[3])                    : x(v[0]), y(v[1]), z(v[2]) { }
      inline V3(F* const v)                      : x(v[0]), y(v[1]), z(v[2]) { }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V ZERO()  { return V((F)0.0, (F)0.0, (F)0.0); }
      static inline V UNITX() { return V((F)1.0, (F)0.0, (F)0.0); }
      static inline V UNITY() { return V((F)0.0, (F)1.0, (F)0.0); }
      static inline V UNITZ() { return V((F)0.0, (F)0.0, (F)1.0); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  operator ==   (const V& v)     const { return (x == v.x && y == v.y && z == v.z);      }
      inline bool  operator !=   (const V& v)     const { return (x != v.x || y != v.y || z != v.z);      }
      inline bool  operator <    (const V& v)     const { return (x <  v.x && y <  v.y && z <  v.z);      }
      inline bool  operator <=   (const V& v)     const { return (x <= v.x && y <= v.y && z <= v.z);      }
      inline bool  operator >    (const V& v)     const { return (x >  v.x && y >  v.y && z >  v.z);      }
      inline bool  operator >=   (const V& v)     const { return (x >= v.x && y >= v.y && z >= v.z);      }
      inline V     operator +    (const V& v)     const { return V(x + v.x, y + v.y, z + v.z);            }
      inline V     operator -    (const V& v)     const { return V(x - v.x, y - v.y, z - v.z);            }
      inline V     operator *    (const V& v)     const { return V(x * v.x, y * v.y, z * v.z);            }
      inline V     operator /    (const V& v)     const { return V(x / v.x, y / v.y, z / v.z);            }
      inline V     operator *    (const F  s)     const { return V(x * s, y * s, z * s);                  }
      inline V     operator /    (const F  s)     const { return V(x / s, y / s, z / s);                  }
      inline V&    operator =    (const V& v)           { x = v.x;  y = v.y;  z =  v.z; return *thiss();  }
      inline V&    operator +=   (const V& v)           { x += v.x; y += v.y; z += v.z; return *thiss();  }
      inline V&    operator -=   (const V& v)           { x -= v.x; y -= v.y; z -= v.z; return *thiss();  }
      inline V&    operator *=   (const V& v)           { x *= v.x; y *= v.y; z *= v.z; return *thiss();  }
      inline V&    operator /=   (const V& v)           { x /= v.x; y /= v.y; z /= v.z; return *thiss();  }
      inline V&    operator =    (const F  s)           { x =  s;   y =  s; z =  s;     return *thiss();  }
      inline V&    operator +=   (const F  s)           { x += s;   y += s; z += s;     return *thiss();  }
      inline V&    operator -=   (const F  s)           { x -= s;   y -= s; z -= s;     return *thiss();  }
      inline V&    operator *=   (const F  s)           { x *= s;   y *= s; z *= s;     return *thiss();  }
      inline V&    operator /=   (const F  s)           { x /= s;   y /= s; z /= s;     return *thiss();  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline       V  operator - ()               const { return V(-x, -y, -z); }
      inline const V& operator + ()               const { return *this; }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool isZero()                         const { return x == (F)0.0 && y == (F)0.0 && z == (F)0.0;                  }
      inline bool isZero(const F e2)               const { return thiss()->length2() <= e2;                                   }
      inline bool equals(const V& v, const F e2)   const { return (thiss() - v).length2() <= e2;                              }
      inline void swap(V& v)                             { std::swap(x, v.x); std::swap(y, v.y); std::swap(z, v.z);           }
      inline V    cross(const V& v)                const { return V(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
      inline F    dot(const V& v)                  const { return x * v.x + y * v.y + z * v.z;                                }
      inline F    length2()                        const { return thiss()->dot(*((V*)this));                                       }
      inline F    length()                         const { return V::_sqrt(thiss()->length2());                               }
      inline F    distance2(const V& v)            const { return (thiss() - v).length2();                                    }
      inline F    distance(const V& v)             const { return V::_sqrt(thiss()->distance2(v));                            }
      inline V    maxC(const V& v)                 const { return V(v.x > x ? v.x : x, v.y > y ? v.y : y, v.z > z ? v.z : z); }
      inline V    minC(const V& v)                 const { return V(v.x < x ? v.x : x, v.y < y ? v.y : y, v.z < z ? v.z : z); }
      inline V    boundC(const V& mi, const V& ma) const { V t(thiss()->minC(ma)); t.max(mi); return t;                       }
      inline V    absC()                           const { return V(V::_abs(x), V::_abs(y), V::_abs(z));                      }
      inline void max(const V& v)                        { if (v.x > x) x = v.x; if (v.y > y) y = v.y; if (v.z > z) z = v.z;  }
      inline void min(const V& v)                        { if (v.x < x) x = v.x; if (v.y < y) y = v.y; if (v.z < z) z = v.z;  }
      inline void bound(const V& mi, const V& ma)        { thiss()->min(ma); thiss()->max(mi);                                }
      inline void abs()                                  { x = V::_abs(x); y = V::_abs(y); z = V::_abs(z);                    }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V    xzy()                            const { return V(x, z, y); }
      inline void xzy()                                  { std::swap(y, z);   }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V    random()                        { return V(std::rand(), std::rand(), std::rand());      }
      static inline void random(V* v, const size_t size) { for (size_t i = 0; i < size; i++) v[i] = V::random(); }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                 FLOATING POINT & INTEGER TEMPLATES V3 [L2, ABSTRACT]                                   //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 3D Vector Template for Floating Point (32/64) [L2]
   /// </summary>
   template <typename V, typename F>
   class V3fdt abstract : public V3<V, F>
   {
   public:
      inline V3fdt()                                              { }
      inline V3fdt(const F x, const F y, const F z) : V3(x, y, z) { }
      inline V3fdt(const F s)                       : V3(s, s, s) { }
      inline V3fdt(const F v[3])                    : V3(v)       { }
      inline V3fdt(F* const v)                      : V3(v)       { }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V& operator /= (const V& v)       { x /= v.x; y /= v.y; z /= v.z; return (V&)*this;              }
      inline V  operator /  (const V& v) const { return V(x / v.x, y / v.y, z / v.z);                         }
      inline V& operator /= (const F  s)       { F t = (F)1.0 / s; x *= t; y *= t; z *= t; return (V&)*this;  }
      inline V  operator /  (const F  s) const { F t = (F)1.0 / s; return V(x * t, y * t, z * t);             }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  isNaN()                              const { return isnan<F>(x) || isnan<F>(y) || isnan<F>(z);  }
      inline void  normalise()                                { *thiss() /= thiss()->length();                      }
      inline V     normaliseC()                         const { V t(*thiss()); t.normalise(); return t;             }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V     roundC()                             const { return V(V::_round(x), V::_round(y), V::_round(z));   }
      inline V     floorC()                             const { return V(V::_floor(x), V::_floor(y), V::_floor(z));   }
      inline V     ceilC()                              const { return V(V::_ceil(x),  V::_ceil(y),  V::_ceil(z));    }
      inline void  round()                                    { x = V::_round(x); y = V::_round(y); z = V::_round(z); }
      inline void  floor()                                    { x = V::_floor(x); y = V::_floor(y); z = V::_floor(z); }
      inline void  ceil()                                     { x = V::_ceil(x);  y = V::_ceil(y);  z = V::_ceil(z);  }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V    randomN()                            { V t(V::random()); t.normalise(); return t;             }
      static inline void randomN(V* v, const size_t size)     { for (size_t i = 0; i < size; i++) v[i] = V::randomN(); }
      //------------------------------------------------------------------------------------------------------------------------//
   };

   /// <summary>
   /// Abstract 3D Vector for Integer (32/64) [L2]
   /// </summary>
   template <typename V, typename F>
   class V3ilt abstract : public V3<V, F>
   {
   public:
      static inline F _abs(const int s)  { return ::abs(s);        }
      static inline F _sqrt(const int s) { return (F)::sqrt<F>(s); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V3ilt()                                              { }
      inline V3ilt(const F x, const F y, const F z) : V3(x, y, z) { }
      inline V3ilt(const F s)                       : V3(s, s, s) { }
      inline V3ilt(const F v[3])                    : V3(v)       { }
      inline V3ilt(F* const v)                      : V3(v)       { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                       32-BIT & 64-BIT TEMPLATES [L3, ABSTRACT]                                         //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 3D Vector Template for Single Precision FP [L3]
   /// </summary>
   template <typename V>
   class V3ft abstract : public V3fdt<V, float>
   {
   public:
      static inline float _abs(const float s)   { return ::fabsf(s);  }
      static inline float _round(const float s) { return ::roundf(s); }
      static inline float _floor(const float s) { return ::floorf(s); }
      static inline float _ceil(const float s)  { return ::ceilf(s);  }
      static inline float _sqrt(const float s)  { return ::sqrtf(s);  }
      static inline float _cos(const float s)   { return ::cosf(s);   }
      static inline float _sin(const float s)   { return ::sinf(s);   }
      static inline float _acos(const float s)  { return ::acosf(s);  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V3ft() { }
      inline V3ft(const float x, const float y, const float z) : V3fdt(x, y, z) { }
      inline V3ft(const float s)                               : V3fdt(s, s, s) { }
      inline V3ft(const float v[3])                            : V3fdt(v)       { }
      inline V3ft(float* const v)                              : V3fdt(v)       { }
   };

   /// <summary>
   /// Abstract 3D Vector Template for Double Precision FP [L3]
   /// </summary>
   template <typename V>
   class V3dt abstract : public V3fdt<V, double>
   {
   public:
      static inline double _abs(const double s)   { return ::abs(s); }
      static inline double _round(const double s) { return ::round(s); }
      static inline double _floor(const double s) { return ::floor(s); }
      static inline double _ceil(const double s)  { return ::ceil(s); }
      static inline double _sqrt(const double s)  { return ::sqrt(s); }
      static inline double _cos(const double s)   { return ::cos(s); }
      static inline double _sin(const double s)   { return ::sin(s); }
      static inline double _acos(const double s)  { return ::acos(s); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V3dt() { }
      inline V3dt(const double x, const double y, const double z) : V3fdt(x, y, z) { }
      inline V3dt(const double s)                                 : V3fdt(s, s, s) { }
      inline V3dt(const double v[3])                              : V3fdt(v)       { }
      inline V3dt(double* const v)                                : V3fdt(v)       { }
   };

   /// <summary>
   /// Abstract 3D Vector Template for Integer (32) [L3]
   /// </summary>
   template <typename V>
   class V3it abstract : public V3ilt<V, int>
   {
   public:
      inline V3it() { }
      inline V3it(const int x, const int y, const int z) : V3ilt(x, y, z) { }
      inline V3it(const int s)                           : V3ilt(s, s, s) { }
      inline V3it(const int v[3])                        : V3ilt(v)       { }
      inline V3it(int* const v)                          : V3ilt(v)       { }
   };

   /// <summary>
   /// Abstract 3D Vector Template for Integer (64) [L3]
   /// </summary>
   template <typename V>
   class V3lt abstract : public V3ilt<V, long long>
   {
   public:
      inline V3lt() { }
      inline V3lt(const long long x, const long long y, const long long z) : V3ilt(x, y, z) { }
      inline V3lt(const long long s)                                       : V3ilt(s, s, s) { }
      inline V3lt(const long long v[3])                                    : V3ilt(v)       { }
      inline V3lt(long long* const v)                                      : V3ilt(v)       { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                     GENERIC/NON-SIMD CLASSES [L4]                                                      //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Single Precision 3D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V3fg : public V3ft<V3fg>
   {
   public:
      inline V3fg() { }
      inline V3fg(const float x, const float y, const float z)    : V3ft(x, y, z)                                 { }
      inline V3fg(const float s)                                  : V3ft(s, s, s)                                 { }
      inline V3fg(const float v[3])                               : V3ft(v)                                       { }
      inline V3fg(float* const v)                                 : V3ft(v)                                       { }
      inline V3fg(const int v[3])                                 : V3ft((float)v[0], (float)v[1], (float)v[2])   { }
      inline V3fg(const double x, const double y, const double z) : V3ft((float)x,    (float)y,    (float)z)      { }
      inline V3fg(const int x, const int y, const int z)          : V3ft((float)x,    (float)y,    (float)z)      { }
   };

   /// <summary>
   /// Double Precision 3D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V3dg : public V3dt<V3dg>
   {
   public:
      inline V3dg()                                                                                                   { }
      inline V3dg(const double x, const double y, const double z) : V3dt(x, y, z)                                     { }
      inline V3dg(const double s)                                 : V3dt(s, s, s)                                     { }
      inline V3dg(const double v[3])                              : V3dt(v)                                           { }
      inline V3dg(double* const v)                                : V3dt(v)                                           { }
      inline V3dg(const int v[3])                                 : V3dt((double)v[0], (double)v[1], (double)v[2])    { }
      inline V3dg(const float x, const float y, const float z)    : V3dt((double)x,    (double)y,    (double)z)       { }
      inline V3dg(const int x, const int y, const int z)          : V3dt((double)x,    (double)y,    (double)z)       { }
   };

   /// <summary>
   /// 32-Bit Integer 3D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V3ig : public V3it<V3ig>
   {
   public:
      inline V3ig() { }
      inline V3ig(const int x, const int y, const int z) : V3it(x, y, z) { }
      inline V3ig(const int s)                           : V3it(s, s, s) { }
      inline V3ig(const int v[3])                        : V3it(v)       { }
      inline V3ig(int* const v)                          : V3it(v)       { }
   };

   /// <summary>
   /// 64-Bit Integer 3D Vector (Generic, no SIMD) [L4]
   /// </summary>//------------------------------------------------------------------------------------------------------------------------//
   class V3lg : public V3lt<V3lg>
   {
   public:
      inline V3lg() { }
      inline V3lg(const long long x, const long long y, const long long z) : V3lt(x, y, z) { }
      inline V3lg(const long long s)                                       : V3lt(s, s, s) { }
      inline V3lg(const long long v[3])                                    : V3lt(v)       { }
      inline V3lg(long long* const v)                                      : V3lt(v)       { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                             SIMD CLASSES [L4]                                                          //
   //------------------------------------------------------------------------------------------------------------------------//
#if defined(SIMD_V2_FP_32_SSE2)
   /// <summary>
   /// Single Precision 3D Vector (unaligned SSE/SIMD)
   /// </summary>
   ALIGN16 class V3fs : public V3ft<V3fs>
   {
   public:
      inline __m128 load()                  const { return _mm_load_ps(vals);             }
      inline void   store(const __m128 v)   const { _mm_store_ps((float*)vals, v);        }
      //------------------------------------------------------------------------------------------------------------------------//
      inline static __m128i unsetMaskHi()           { return _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF); }
      inline static __m128  unsetHi(const __m128 v) { return _mm_and_ps(v, _mm_castsi128_ps(unsetMaskHi()));                }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V3fs()                                               { }
      inline V3fs(const float x, const float y, const float z)    { store(_mm_set_ps(0.0f, z, y, x));                      }
      inline V3fs(const float s)                                  { store(_mm_set_ps1(s));                                 }
      inline V3fs(const double x, const double y, const double z) { store(_mm_set_ps(0.0f, (float)z, (float)y, (float)x)); }
      inline V3fs(const int x, const int y, const int z)          { store(_mm_set_ps(0.0f, (float)z, (float)y, (float)x)); }
      inline V3fs(const float v[3])                               { store(unsetHi(_mm_loadu_ps(v)));                       }
      inline V3fs(float* const v)                                 { store(unsetHi(_mm_loadu_ps(v)));                       }
      inline V3fs(const int v[3])                                 { store(unsetHi(_mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)v)))); }
      inline V3fs(const __m128 v)                                 { store(v);                                             }
      //------------------------------------------------------------------------------------------------------------------------//
      inline       void* operator new  (size_t size)       { return _aligned_malloc(sizeof(V3fs), 16);                       }
      inline       void* operator new[](size_t size)       { return _aligned_malloc(size * sizeof(V3fs), 16);                }
      inline       bool  operator == (const V3fs& v) const { return _mm_movemask_ps(_mm_cmpeq_ps(load(), v.load())) == 0x0F; }
      inline       bool  operator != (const V3fs& v) const { return _mm_movemask_ps(_mm_cmpeq_ps(load(), v.load())) != 0x00; }
      inline       bool  operator <  (const V3fs& v) const { return _mm_movemask_ps(_mm_cmplt_ps(load(), v.load())) == 0x0F; }
      inline       bool  operator <= (const V3fs& v) const { return _mm_movemask_ps(_mm_cmple_ps(load(), v.load())) == 0x0F; }
      inline       bool  operator >  (const V3fs& v) const { return _mm_movemask_ps(_mm_cmpgt_ps(load(), v.load())) == 0x0F; }
      inline       bool  operator >= (const V3fs& v) const { return _mm_movemask_ps(_mm_cmpge_ps(load(), v.load())) == 0x0F; }
      inline       V3fs  operator +  (const V3fs& v) const { return V3fs(_mm_add_ps(load(), v.load()));                  }
      inline       V3fs  operator -  (const V3fs& v) const { return V3fs(_mm_sub_ps(load(), v.load()));                  }
      inline       V3fs  operator *  (const V3fs& v) const { return V3fs(_mm_mul_ps(load(), v.load()));                  }
      inline       V3fs  operator /  (const V3fs& v) const { return V3fs(_mm_div_ps(load(), v.load()));                  }
      inline       V3fs  operator *  (const float s) const { return V3fs(_mm_mul_ps(load(), _mm_set1_ps(s)));            }
      inline       V3fs  operator /  (const float s) const { return V3fs(_mm_div_ps(load(), _mm_set1_ps(s)));            }
      inline       V3fs  operator -  ()              const { return V3fs(_mm_sub_ps(_mm_setzero_ps(), load()));          }
      inline const V3fs& operator +  ()              const { return *this;                                               }
      inline       V3fs& operator =  (const V3fs& v)       { store(v.load());                           return *this;    }
      inline       V3fs& operator += (const V3fs& v)       { store(_mm_add_ps(load(), v.load()));       return *this;    }
      inline       V3fs& operator -= (const V3fs& v)       { store(_mm_sub_ps(load(), v.load()));       return *this;    }
      inline       V3fs& operator *= (const V3fs& v)       { store(_mm_mul_ps(load(), v.load()));       return *this;    }
      inline       V3fs& operator /= (const V3fs& v)       { store(_mm_div_ps(load(), v.load()));       return *this;    }
      inline       V3fs& operator =  (const float s)       { store(_mm_set1_ps(s));                     return *this;    }
      inline       V3fs& operator += (const float s)       { store(_mm_add_ps(load(), _mm_set1_ps(s))); return *this;    }
      inline       V3fs& operator -= (const float s)       { store(_mm_sub_ps(load(), _mm_set1_ps(s))); return *this;    }
      inline       V3fs& operator *= (const float s)       { store(_mm_mul_ps(load(), _mm_set1_ps(s))); return *this;    }
      inline       V3fs& operator /= (const float s)       { store(_mm_div_ps(load(), _mm_set1_ps(s))); return *this;    }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void swap(V3fs& v)                                { __m128 t(load()); store(v.load()); v.store(t);         }
      inline V3fs absC()                                 const { return V3fs(_mm_andnot_ps(_mm_set1_ps(-0.f), load())); }
      inline V3fs maxC(const V3fs& v)                    const { return V3fs(_mm_max_ps(load(), v.load()));             }
      inline V3fs minC(const V3fs& v)                    const { return V3fs(_mm_min_ps(load(), v.load()));             }
      inline V3fs boundC(const V3fs& mi, const V3fs& ma) const { V3fs t(minC(ma)); t.max(mi); return t;                 }
      inline void abs()                                        { store(_mm_andnot_ps(_mm_set1_ps(-0.), load()));        }
      inline void max(const V3fs& v)                           { store(_mm_max_ps(load(), v.load()));                   }
      inline void min(const V3fs& v)                           { store(_mm_min_ps(load(), v.load()));                   }
      inline void bound(const V3fs& mi, const V3fs& ma)        { min(ma); max(mi);                                      }
      //------------------------------------------------------------------------------------------------------------------------//
      inline float dot(const V3fs& v) const
      {
         __m128 a(_mm_mul_ps(load(), v.load()));
         __m128 b(_mm_add_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1))));
         __m128 c(_mm_add_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3))));
         return c.m128_f32[0];
      }
      inline float length() const
      {
         __m128 t(load());
         __m128 a(_mm_mul_ps(t, t));
         __m128 b(_mm_add_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1))));
         __m128 c(_mm_add_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3))));
         __m128 e(_mm_sqrt_ss(c));
         return e.m128_f32[0];
      }
      inline V3fs cross(const V3fs& v) const 
      {
         __m128 a(load());
         __m128 b(v.load());
         __m128 c(_mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))));
         __m128 d(_mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1))));
         __m128 e(_mm_sub_ps(c, d));
         return V3fs(e);
      }
      /*inline bool inside(const V2fs& min, const V2fs& max) const
      {
         __m128 a(load());
         __m128 b(_mm_cmpge_ps(a, min.load()));
         __m128 c(_mm_cmple_ps(a, max.load()));
         __m128 d(_mm_and_ps(b, c));
         return _mm_movemask_ps(d) == 0x0F;
      }
      inline bool inside(const V2fs& min, const V2fs& max, const float e) const
      {
         __m128 eps(_mm_set1_ps(e));
         __m128 a(load());
         __m128 b(_mm_cmpge_ps(a, _mm_sub_ps(min.load(), eps)));
         __m128 c(_mm_cmple_ps(a, _mm_add_ps(max.load(), eps)));
         __m128 d(_mm_and_ps(b, c));
         return _mm_movemask_ps(d) == 0x0F;
      }*/
#if defined(SIMD_V2_FP_32_SSE41)
      inline V3fs roundC() const { return V3fs(_mm_round_ps(load(), _MM_FROUND_NINT));  }
      inline V3fs floorC() const { return V3fs(_mm_round_ps(load(), _MM_FROUND_FLOOR)); }
      inline V3fs ceilC()  const { return V3fs(_mm_round_ps(load(), _MM_FROUND_CEIL));  }
      inline void round()        { store(_mm_round_ps(load(), _MM_FROUND_NINT));        }
      inline void floor()        { store(_mm_round_ps(load(), _MM_FROUND_FLOOR));       }
      inline void ceil()         { store(_mm_round_ps(load(), _MM_FROUND_CEIL));        }
#endif
      //------------------------------------------------------------------------------------------------------------------------//
   };
   typedef V3fs V3f;  // use SIMD as default
#else
   typedef V3fg V3f;  // use plain as default
#endif

#pragma endregion
}

#pragma once
//------------------------------------------------------------------------------------------------------------------------//
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <random>
#include <intrin.h>
//------------------------------------------------------------------------------------------------------------------------//
#define TWOPI (2.0*M_PI)

#if defined(SIMD_V2_FP_32_SSE41) && !defined(SIMD_V2_FP_32_SSE2)
# define SIMD_V2_FP_32_SSE2
#endif
#if defined(SIMD_V2_FP_64_SSE41) && !defined(SIMD_V2_FP_64_SSE2)
# define SIMD_V2_FP_64_SSE2
#endif
#if defined(SIMD_V3_FP_32_SSE41) && !defined(SIMD_V3_FP_32_SSE2)
# define SIMD_V3_FP_32_SSE2
#endif

#if defined(SIMD_TYPES_SSE) || defined(SIMD_TYPES_AVX)
#  define ALIGN8                          __declspec(align(8))
#  define ALIGN16 __declspec(intrin_type) __declspec(align(16))
#  define ALIGN32 __declspec(intrin_type) __declspec(align(32))
#endif
//------------------------------------------------------------------------------------------------------------------------//
namespace simd
{
#pragma region V2
   //------------------------------------------------------------------------------------------------------------------------//
   //                                          ROOT TEMPLATE V2 [L1, ABSTRACT]                                               //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 2D Vector Template for Floating Point (32/64) AND Integer (32/64). [L1]
   /// </summary>
   template <typename V, typename F>
   class V2
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
      inline void* operator new  (size_t size)          { return malloc(sizeof(V));        }
      inline void* operator new[](size_t size)          { return malloc(size * sizeof(V)); }
      inline F     operator []   (const size_t i) const { return vals[i];                  }
      inline F     operator []   (const size_t i)       { return vals[i];                  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  operator ==   (const V& v)     const { return ((x == v.x) & (y == v.y));     }
      inline bool  operator !=   (const V& v)     const { return ((x != v.x) | (y != v.y));     }
      inline bool  operator <    (const V& v)     const { return ((x <  v.x) & (y <  v.y));     }
      inline bool  operator <=   (const V& v)     const { return ((x <= v.x) & (y <= v.y));     }
      inline bool  operator >    (const V& v)     const { return ((x >  v.x) & (y >  v.y));     }
      inline bool  operator >=   (const V& v)     const { return ((x >= v.x) & (y >= v.y));     }
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
      inline void madd(const V& m, const V& a)           { *thiss() = (*thiss() * m) + a;                  }
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
   class V2fdt : public V2<V, F>
   {
   public:
      inline V2fdt()                                                      { }
      inline V2fdt(const F x, const F y) : V2<V, F>(x, y)                 { }
      inline V2fdt(const F scalar)       : V2<V, F>(scalar, scalar)       { }
      inline V2fdt(const F values[2])    : V2<V, F>(values[0], values[1]) { }
      inline V2fdt(F* const values)      : V2<V, F>(values[0], values[1]) { }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V& operator /= (const V& v)       { x /= v.x; y /= v.y; return (V&)*this;               }
      inline V  operator /  (const V& v) const { return V(x / v.x, y / v.y);                         }
      inline V& operator /= (const F  s)       { F t = (F)1.0 / s; x *= t; y *= t; return (V&)*this; }
      inline V  operator /  (const F  s) const { F t = (F)1.0 / s; return V(x * t, y * t);           }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  isNaN()                              const { return isnan<F>(x) || isnan<F>(y);      }
      inline void  normalise()                                { *thiss() /= thiss()->length();          }
      inline V     normaliseC()                         const { V t(*thiss()); t.normalise(); return t; }
      inline void  scaleTo(const F l)                         { *thiss() *= (l / thiss()->length());    }
      // scaleToC
      //------------------------------------------------------------------------------------------------------------------------//
      inline V     roundC()                             const { return V(V::_round(x), V::_round(y));           }
      inline V     floorC()                             const { return V(V::_floor(x), V::_floor(y));           }
      inline V     ceilC()                              const { return V(V::_ceil(x),  V::_ceil(y));            }
      inline V     roundByC(const F n)                  const { V c(*thiss()); c.roundBy(n); return c;          }
      inline void  round()                                    { x = V::_round(x); y = V::_round(y);             }
      inline void  floor()                                    { x = V::_floor(x); y = V::_floor(y);             }
      inline void  ceil()                                     { x = V::_ceil(x);  y = V::_ceil(y);              }
      inline void  roundBy(const F n)                         { *thiss() /= n; thiss()->round(); *thiss() *= n; }
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
   class V2ilt : public V2<V, F>
   {
   public:
      static inline F _abs(const int s)  { return ::abs(s);         }
      static inline F _sqrt(const int s) { return (F)::sqrt<F>(s);  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2ilt()                                      { }
      inline V2ilt(const F x, const F y) : V2<V, F>(x, y) { }
      inline V2ilt(const F s)            : V2<V, F>(s, s) { }
      inline V2ilt(const F v[2])         : V2<V, F>(v)    { }
      inline V2ilt(F* const v)           : V2<V, F>(v)    { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                       32-BIT & 64-BIT TEMPLATES [L3, ABSTRACT]                                         //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 2D Vector Template for Single Precision FP [L3]
   /// </summary>
   template <typename V>
   class V2ft : public V2fdt<V, float>
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
      inline V2ft(const float x, const float y) : V2fdt<V, float>(x, y) { }
      inline V2ft(const float s)                : V2fdt<V, float>(s, s) { }
      inline V2ft(const float v[2])             : V2fdt<V, float>(v)    { }
      inline V2ft(float* const v)               : V2fdt<V, float>(v)    { }
   };

   /// <summary>
   /// Abstract 2D Vector Template for Double Precision FP [L3]
   /// </summary>
   template <typename V>
   class V2dt : public V2fdt<V, double>
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
      inline V2dt()                                                        { }
      inline V2dt(const double x, const double y) : V2fdt<V, double>(x, y) { }
      inline V2dt(const double s)                 : V2fdt<V, double>(s, s) { }
      inline V2dt(const double v[2])              : V2fdt<V, double>(v)    { }
      inline V2dt(double* const v)                : V2fdt<V, double>(v)    { }
   };

   /// <summary>
   /// Abstract 2D Vector Template for Integer (32) [L3]
   /// </summary>
   template <typename V>
   class V2it : public V2ilt<V, int>
   {
   public:
      inline V2it()                                               { }
      inline V2it(const int x, const int y) : V2ilt<V, int>(x, y) { }
      inline V2it(const int s)              : V2ilt<V, int>(s, s) { }
      inline V2it(const int v[2])           : V2ilt<V, int>(v)    { }
      inline V2it(int* const v)             : V2ilt<V, int>(v)    { }
   };

   /// <summary>
   /// Abstract 2D Vector Template for Integer (64) [L3]
   /// </summary>
   template <typename V>
   class V2lt : public V2ilt<V, long long>
   {
   public:
      inline V2lt()                                                                 { }
      inline V2lt(const long long x, const long long y) : V2ilt<V, long long>(x, y) { }
      inline V2lt(const long long s)                    : V2ilt<V, long long>(s, s) { }
      inline V2lt(const long long v[2])                 : V2ilt<V, long long>(v)    { }
      inline V2lt(long long* const v)                   : V2ilt<V, long long>(v)    { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                     GENERIC/NON-SIMD CLASSES [L4]                                                      //
   //------------------------------------------------------------------------------------------------------------------------//
   class V2fg;
   class V2dg;

   /// <summary>
   /// Single Precision 2D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V2fg : public V2ft<V2fg>
   {
   public:
      inline V2fg()                                                                      { }
      inline V2fg(const float x, const float y)   : V2ft<V2fg>(x, y)                     { }
      inline V2fg(const float s)                  : V2ft<V2fg>(s, s)                     { }
      inline V2fg(const float v[2])               : V2ft<V2fg>(v)                        { }
      inline V2fg(float* const v)                 : V2ft<V2fg>(v)                        { }
      inline V2fg(const int v[2])                 : V2ft<V2fg>((float)v[0], (float)v[1]) { }
      inline V2fg(const double x, const double y) : V2ft<V2fg>((float)x,    (float)y)    { }
      inline V2fg(const int x, const int y)       : V2ft<V2fg>((float)x,    (float)y)    { }
      inline V2fg(const V2dt<V2dg>& v)            : V2ft<V2fg>((float)v.x,  (float)v.y)  { }
   };

   /// <summary>
   /// Double Precision 2D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V2dg : public V2dt<V2dg>
   {
   public:
      inline V2dg()                                                                        { }
      inline V2dg(const double x, const double y) : V2dt<V2dg>(x, y)                       { }
      inline V2dg(const double s)                 : V2dt<V2dg>(s, s)                       { }
      inline V2dg(const double v[2])              : V2dt<V2dg>(v)                          { }
      inline V2dg(double* const v)                : V2dt<V2dg>(v)                          { }
      inline V2dg(const int v[2])                 : V2dt<V2dg>((double)v[0], (double)v[1]) { }
      inline V2dg(const float x, const float y)   : V2dt<V2dg>((double)x,    (double)y)    { }
      inline V2dg(const int x, const int y)       : V2dt<V2dg>((double)x,    (double)y)    { }
      inline V2dg(const V2ft<V2fg>& v)            : V2dt<V2dg>((double)v.x,  (double)v.y)  { }
   };

   /// <summary>
   /// 32-Bit Integer 2D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V2ig : public V2it<V2ig>
   {
   public:
      inline V2ig()                                            { }
      inline V2ig(const int x, const int y) : V2it<V2ig>(x, y) { }
      inline V2ig(const int s)              : V2it<V2ig>(s, s) { }
      inline V2ig(const int v[2])           : V2it<V2ig>(v)    { }
      inline V2ig(int* const v)             : V2it<V2ig>(v)    { }
   };

   /// <summary>
   /// 64-Bit Integer 2D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V2lg : public V2lt<V2lg>
   {
   public:
      inline V2lg()                                                        { }
      inline V2lg(const long long x, const long long y) : V2lt<V2lg>(x, y) { }
      inline V2lg(const long long s)                    : V2lt<V2lg>(s, s) { }
      inline V2lg(const long long v[2])                 : V2lt<V2lg>(v)    { }
      inline V2lg(long long* const v)                   : V2lt<V2lg>(v)    { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                             SIMD CLASSES [L4]                                                          //
   //------------------------------------------------------------------------------------------------------------------------//
#if defined(SIMD_V2_FP_32_SSE2)
   class V2fs;
   class V2ds;

   /// <summary>
   /// Single Precision 2D Vector (SSE/SIMD)
   /// </summary>
   ALIGN8 class V2fs : public V2ft<V2fs>
   {
   public:
      inline __m128 load()                  const { return _mm_castpd_ps(_mm_load_sd((double*)vals)); }
      inline __m128 load2()                 const { return _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals)); }
      inline void   store(const __m128& v)        { _mm_store_sd((double*)vals, _mm_castps_pd(v));    }
      inline void   store2(const __m128& v)       { _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(v)); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2fs()                                                                { }
      inline V2fs(const float x, const float y)   : V2ft<V2fs>(x, y)               { }
      inline V2fs(const float s)                  : V2ft<V2fs>(s, s)               { }
      inline V2fs(const double x, const double y) : V2ft<V2fs>((float)x, (float)y) { }
      inline V2fs(const int x, const int y)       : V2ft<V2fs>((float)x, (float)y) { }
      inline V2fs(const float values[2])          { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values)); }
      inline V2fs(float* const values)            { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values)); }
      inline V2fs(const int values[2])            { store(_mm_cvtepi32_ps(_mm_loadl_epi64((__m128i*)values)));           }
      inline V2fs(const __m128& values)           { store(values);                                                       }
      inline V2fs(const V2dt<V2ds>& v)            { store(_mm_cvtpd_ps(_mm_load_pd(v.vals)));                            }
      //------------------------------------------------------------------------------------------------------------------------//
      inline       void* operator new  (size_t size)        { return _aligned_malloc(sizeof(V2fs), 8);                          }
      inline       void* operator new[](size_t size)        { return _aligned_malloc(size * sizeof(V2fs), 8);                   }
      inline       bool  operator == (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmpeq_ps(load2(), v.load2())) == 0x0F; }
      inline       bool  operator != (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmpeq_ps(load2(), v.load2())) != 0x00; }
      inline       bool  operator <  (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmplt_ps(load2(), v.load2())) == 0x0F; }
      inline       bool  operator <= (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmple_ps(load2(), v.load2())) == 0x0F; }
      inline       bool  operator >  (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmpgt_ps(load2(), v.load2())) == 0x0F; }
      inline       bool  operator >= (const V2fs&  v) const { return _mm_movemask_ps(_mm_cmpge_ps(load2(), v.load2())) == 0x0F; }
      inline       V2fs  operator +  (const V2fs&  v) const { return V2fs(_mm_add_ps(load2(), v.load2()));                      }
      inline       V2fs  operator -  (const V2fs&  v) const { return V2fs(_mm_sub_ps(load2(), v.load2()));                      }
      inline       V2fs  operator *  (const V2fs&  v) const { return V2fs(_mm_mul_ps(load2(), v.load2()));                      }
      inline       V2fs  operator /  (const V2fs&  v) const { return V2fs(_mm_div_ps(load2(), v.load2()));                      }
      inline       V2fs  operator *  (const float  s) const { return V2fs(_mm_mul_ps(load2(), _mm_set1_ps(s)));                 }
      inline       V2fs  operator /  (const float  s) const { return V2fs(_mm_div_ps(load2(), _mm_set1_ps(s)));                 }
      inline       V2fs  operator -  ()               const { return V2fs(_mm_sub_ps(_mm_setzero_ps(), load2()));               }
      inline const V2fs& operator +  ()               const { return *this;                                                     }
      inline       V2fs& operator =  (const V2fs&  v)       { store(v.load());                            return *this;         }
      inline       V2fs& operator += (const V2fs&  v)       { store(_mm_add_ps(load2(), v.load2()));      return *this;         }
      inline       V2fs& operator -= (const V2fs&  v)       { store(_mm_sub_ps(load2(), v.load2()));      return *this;         }
      inline       V2fs& operator *= (const V2fs&  v)       { store(_mm_mul_ps(load2(), v.load2()));      return *this;         }
      inline       V2fs& operator /= (const V2fs&  v)       { store(_mm_div_ps(load2(), v.load2()));      return *this;         }
      inline       V2fs& operator =  (const float  s)       { store(_mm_set1_ps(s));                      return *this;         }
      inline       V2fs& operator += (const float  s)       { store(_mm_add_ps(load2(), _mm_set1_ps(s))); return *this;         }
      inline       V2fs& operator -= (const float  s)       { store(_mm_sub_ps(load2(), _mm_set1_ps(s))); return *this;         }
      inline       V2fs& operator *= (const float  s)       { store(_mm_mul_ps(load2(), _mm_set1_ps(s))); return *this;         }
      inline       V2fs& operator /= (const float  s)       { store(_mm_div_ps(load2(), _mm_set1_ps(s))); return *this;         }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void swap(V2fs& v)                                { __m128 t(load2()); store(v.load2()); v.store(t);                }
      inline V2fs absC()                                 const { return V2fs(_mm_andnot_ps(_mm_set1_ps(-0.f), load2()));         }
      inline V2fs maxC(const V2fs& v)                    const { return V2fs(_mm_max_ps(load2(), v.load2()));                    }
      inline V2fs minC(const V2fs& v)                    const { return V2fs(_mm_min_ps(load2(), v.load2()));                    }
      inline V2fs boundC(const V2fs& mi, const V2fs& ma) const { return V2fs(_mm_max_ps(_mm_min_ps(load2(), ma.load2()), mi.load2())); }
      inline void abs()                                        { store(_mm_andnot_ps(_mm_set1_ps(-0.), load2()));                }
      inline void max(const V2fs& v)                           { store(_mm_max_ps(load2(), v.load2()));                          }
      inline void min(const V2fs& v)                           { store(_mm_min_ps(load2(), v.load2()));                          }
      inline void bound(const V2fs& mi, const V2fs& ma)        { store(_mm_max_ps(_mm_min_ps(load2(), ma.load2()), mi.load2())); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline float dot(const V2fs& v) const
      {
         __m128 a(_mm_mul_ps(load2(), v.load2()));
         __m128 b(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 c(_mm_add_ss(a, b));
         return c.m128_f32[0];
      }
      inline float length() const
      {
         __m128 t(load2());
         __m128 a(_mm_mul_ps(t, t));
         __m128 b(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 c(_mm_add_ss(a, b));
         __m128 d(_mm_sqrt_ss(c));
         return d.m128_f32[0];
      }
      inline float side(const V2fs& s, const V2fs& e) const
      {
         __m128 t(s.load2());
         __m128 a(_mm_sub_ps(e.load2(), t));
         __m128 b(_mm_sub_ps(load2(), t));
         __m128 c(_mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 d(_mm_mul_ps(a, c));
         __m128 f(_mm_shuffle_ps(d, d, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 g(_mm_sub_ss(d, f));
         return g.m128_f32[0];
      }
      inline bool inside(const V2fs& min, const V2fs& max) const
      {
         return _mm_movemask_ps(_mm_and_ps(
            _mm_cmpge_ps(load2(), min.load2()), 
            _mm_cmple_ps(load2(), max.load2()))) == 0x0F;
      }
      inline bool inside(const V2fs& min, const V2fs& max, const float e) const
      {
         const __m128 eps(_mm_set1_ps(e));
         const __m128 a(_mm_and_ps(
            _mm_cmpge_ps(load2(), _mm_sub_ps(min.load2(), eps)), 
            _mm_cmple_ps(load2(), _mm_add_ps(max.load2(), eps))));
         return _mm_movemask_ps(a) == 0x0F;
      }
      inline bool inside(const V2fs& m, const float r2) const
      {
         __m128 t(_mm_sub_ps(load2(), m.load2()));
         __m128 a(_mm_mul_ps(t, t));
         __m128 b(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 c(_mm_add_ss(a, b));
         return c.m128_f32[0] <= r2;
      }
      inline bool inside(const V2fs& m, const float r2, const float e) const
      {
         __m128 t(_mm_sub_ps(load2(), m.load2()));
         __m128 a(_mm_mul_ps(t, t));
         __m128 b(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 c(_mm_add_ss(a, b));
         return c.m128_f32[0] <= (r2+e); 
      }
      inline float area(const V2fs& p, const V2fs& q) const
      {
         __m128 t(load2());
         __m128 a(_mm_sub_ps(p.load2(), t));
         __m128 b(_mm_sub_ps(q.load2(), t));
         __m128 c(_mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 d(_mm_mul_ps(a, c));
         __m128 e(_mm_shuffle_ps(d, d, _MM_SHUFFLE(2, 3, 0, 1)));
         __m128 f(_mm_sub_ss(d, e));
         return 0.5f * f.m128_f32[0];
      }
#if defined(SIMD_V2_FP_32_SSE41)
      inline V2fs  roundC()                const { return V2fs(_mm_round_ps(load2(), _MM_FROUND_NINT));  }
      inline V2fs  floorC()                const { return V2fs(_mm_round_ps(load2(), _MM_FROUND_FLOOR)); }
      inline V2fs  ceilC()                 const { return V2fs(_mm_round_ps(load2(), _MM_FROUND_CEIL));  }
      inline void  round()                       { store(_mm_round_ps(load2(), _MM_FROUND_NINT));        }
      inline void  floor()                       { store(_mm_round_ps(load2(), _MM_FROUND_FLOOR));       }
      inline void  ceil()                        { store(_mm_round_ps(load2(), _MM_FROUND_CEIL));        }
      inline void  roundBy(const float n)        { store(_mm_mul_ps(_mm_round_ps(_mm_mul_ps(load2(), _mm_set1_ps(1.0f / n)), _MM_FROUND_NINT), _mm_set1_ps(n))); }
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
      inline V2ds(const double fX, const double fY) { store(_mm_set_pd(fY, fX));                                                }
      inline V2ds(const float fX, const float fY)   { store(_mm_set_pd((double)fY, (double)fX));                                }
      inline V2ds(const int fX, const int fY)       { store(_mm_set_pd((double)fY, (double)fX));                                }
      inline V2ds(const double scalar)              { store(_mm_set1_pd(scalar));                                               }
      inline V2ds(const double values[2])           { store(_mm_loadu_pd(values));                                              }
      inline V2ds(double* const values)             { store(_mm_loadu_pd(values));                                              }
      inline V2ds(const int values[2])              { store(_mm_cvtepi32_pd(_mm_loadl_epi64((__m128i*)values)));                }
      inline V2ds(const __m128d values)             { store(values);                                                            }
      inline V2ds(const V2ft<V2fs>& v)              { store(_mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)v.vals)))); }
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
      inline void swap(V2ds& v)                                { __m128d t(load()); store(v.load()); v.store(t);              }
      inline V2ds absC()                                 const { return V2ds(_mm_andnot_pd(_mm_set1_pd(-0.), load()));        }
      inline V2ds maxC(const V2ds& v)                    const { return V2ds(_mm_max_pd(load(), v.load()));                   }
      inline V2ds minC(const V2ds& v)                    const { return V2ds(_mm_min_pd(load(), v.load()));                   }
      inline V2ds boundC(const V2ds& mi, const V2ds& ma) const { return V2ds(_mm_max_pd(_mm_min_pd(load(), ma.load()), mi.load())); }
      inline void abs()                                        { store(_mm_andnot_pd(_mm_set1_pd(-0.), load()));              }
      inline void max(const V2ds& v)                           { store(_mm_max_pd(load(), v.load()));                         }
      inline void min(const V2ds& v)                           { store(_mm_min_pd(load(), v.load()));                         }
      inline void bound(const V2ds& mi, const V2ds& ma)        { store(_mm_max_pd(_mm_min_pd(load(), ma.load()), mi.load())); }
      //------------------------------------------------------------------------------------------------------------------------//
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
      inline void   roundBy(const double n)  { store(_mm_mul_pd(_mm_round_pd(_mm_mul_pd(load(), _mm_set1_pd(1.0 / n)), _MM_FROUND_NINT), _mm_set1_pd(n))); }
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

#if defined(SIMD_V2_INT_32_SSE41)
   /// <summary>
   /// 32-Bit Integer 2D Vector (SSE/SIMD)
   /// </summary>
   ALIGN8 class V2is : public V2it<V2is>
   {
   public:
      inline __m128i load()                  const { return _mm_loadl_epi64((__m128i*)vals); }
      inline void    store(const __m128i& v)       { _mm_storel_epi64((__m128i*)vals, v);    }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V2is()                         {                                                                              }
      inline V2is(const int x, const int y) { store(_mm_set_epi32(0, 0, y, x));                                            }
      inline V2is(const int s)              { store(_mm_set1_epi32(s));                                                    }
      inline V2is(const int values[2])      { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values));         }
      inline V2is(int* const values)        { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)values));         }
      inline V2is(const float values[2])    { store(_mm_cvtps_epi32(_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)values)))); }
      inline V2is(const __m128i& values)    { store(values);                                                               }
      //------------------------------------------------------------------------------------------------------------------------//
      inline       void* operator new  (size_t size)        { return _aligned_malloc(sizeof(V2is), 8);                         }
      inline       void* operator new[](size_t size)        { return _aligned_malloc(size * sizeof(V2is), 8);                  }
      inline       bool  operator == (const V2is&  v) const { return _mm_movemask_epi8(_mm_cmpeq_epi32(load(),v.load()))==0xFFFF;}
      inline       bool  operator != (const V2is&  v) const { return _mm_movemask_epi8(_mm_cmpeq_epi32(load(),v.load()))!=0xFFFF;}
      inline       bool  operator <  (const V2is&  v) const { return _mm_movemask_epi8(_mm_cmplt_epi32(load(),v.load()))==0xFF;  }
      inline       bool  operator >  (const V2is&  v) const { return _mm_movemask_epi8(_mm_cmpgt_epi32(load(),v.load()))==0xFF;  }
      inline       V2is  operator +  (const V2is&  v) const { return V2is(_mm_add_epi32(load(), v.load()));                    }
      inline       V2is  operator -  (const V2is&  v) const { return V2is(_mm_sub_epi32(load(), v.load()));                    }
      inline       V2is  operator *  (const V2is&  v) const { return V2is(_mm_mullo_epi32(load(), v.load()));                  }
      //inline     V2is  operator /  (const V2is&  v) const { return V2is(_mm_div_epi32(load(), v.load()));                    }
      inline       V2is  operator *  (const int  s)   const { return V2is(_mm_mullo_epi32(load(), _mm_set1_epi32(s)));         }
      //inline     V2is  operator /  (const float  s) const { return V2is(_mm_div_epi32(load(), _mm_set1_epi32(s)));           }
      inline       V2is  operator -  ()               const { return V2is(_mm_sub_epi32(_mm_setzero_si128(), load()));         }
      inline const V2is& operator +  ()               const { return *this;                                                    }
      inline       V2is& operator =  (const V2is&  v)       { store(v.load());                                   return *this; }
      inline       V2is& operator += (const V2is&  v)       { store(_mm_add_epi32(load(), v.load()));            return *this; }
      inline       V2is& operator -= (const V2is&  v)       { store(_mm_sub_epi32(load(), v.load()));            return *this; }
      inline       V2is& operator *= (const V2is&  v)       { store(_mm_mullo_epi32(load(), v.load()));          return *this; }
      //inline     V2is& operator /= (const V2is&  v)       { store(_mm_div_epi32(load(), v.load()));            return *this; }
      inline       V2is& operator =  (const int  s)         { store(_mm_set1_epi32(s));                          return *this; }
      inline       V2is& operator += (const int  s)         { store(_mm_add_epi32(load(), _mm_set1_epi32(s)));   return *this; }
      inline       V2is& operator -= (const int  s)         { store(_mm_sub_epi32(load(), _mm_set1_epi32(s)));   return *this; }
      inline       V2is& operator *= (const int  s)         { store(_mm_mullo_epi32(load(), _mm_set1_epi32(s))); return *this; }
      //inline     V2is& operator /= (const float  s)       { store(_mm_div_epi32(load(), _mm_set1_epi32(s)));   return *this; }
      inline       bool  operator <= (const V2is&  v) const 
      {
         __m128i a(load());
         __m128i b(v.load());
         __m128i lt(_mm_cmplt_epi32(a, b));
         __m128i eq(_mm_cmpeq_epi32(a, b));
         __m128i or(_mm_or_si128(lt, eq));
         return _mm_movemask_epi8(or) == 0xFFFF;
      }
      inline       bool  operator >= (const V2is&  v) const
      {
         __m128i a(load());
         __m128i b(v.load());
         __m128i lt(_mm_cmpgt_epi32(a, b));
         __m128i eq(_mm_cmpeq_epi32(a, b));
         __m128i or (_mm_or_si128(lt, eq));
         return _mm_movemask_epi8(or ) == 0xFFFF;
      }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void swap(V2is& v)                                { __m128i t(load()); store(v.load()); v.store(t); }
      inline V2is absC()                                 const { return V2is(_mm_abs_epi32(load()));             }
      inline V2is maxC(const V2is& v)                    const { return V2is(_mm_max_epi32(load(), v.load()));   }
      inline V2is minC(const V2is& v)                    const { return V2is(_mm_min_epi32(load(), v.load()));   }
      inline V2is boundC(const V2is& mi, const V2is& ma) const { V2is t(minC(ma)); t.max(mi); return t;          }
      inline void abs()                                        { store(_mm_abs_epi32(load()));                   }
      inline void max(const V2is& v)                           { store(_mm_max_epi32(load(), v.load()));         }
      inline void min(const V2is& v)                           { store(_mm_min_epi32(load(), v.load()));         }
      inline void bound(const V2is& mi, const V2is& ma)        { min(ma); max(mi);                               }
      //------------------------------------------------------------------------------------------------------------------------//
   };
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
   class V3
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
      inline void* operator new  (size_t size)          { return malloc(sizeof(V));        }
      inline void* operator new[](size_t size)          { return malloc(size * sizeof(V)); }
      inline F     operator []   (const size_t i) const { return vals[i];                  }
      inline F     operator []   (const size_t i)       { return vals[i];                  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  operator ==   (const V& v)     const { return ((x == v.x) & (y == v.y) & (z == v.z));  }
      inline bool  operator !=   (const V& v)     const { return ((x != v.x) | (y != v.y) | (z != v.z));  }
      inline bool  operator <    (const V& v)     const { return ((x <  v.x) & (y <  v.y) & (z <  v.z));  }
      inline bool  operator <=   (const V& v)     const { return ((x <= v.x) & (y <= v.y) & (z <= v.z));  }
      inline bool  operator >    (const V& v)     const { return ((x >  v.x) & (y >  v.y) & (z >  v.z));  }
      inline bool  operator >=   (const V& v)     const { return ((x >= v.x) & (y >= v.y) & (z >= v.z));  }
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
      inline void madd(const V& m, const V& a)           { *thiss() = (*thiss() * m) + a;                                     }
      inline bool isZero()                         const { return x == (F)0.0 && y == (F)0.0 && z == (F)0.0;                  }
      inline bool isZero(const F e2)               const { return thiss()->length2() <= e2;                                   }
      inline bool equals(const V& v, const F e2)   const { return (thiss() - v).length2() <= e2;                              }
      inline void swap(V& v)                             { std::swap(x, v.x); std::swap(y, v.y); std::swap(z, v.z);           }
      inline V    cross(const V& v)                const { return V(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
      inline F    dot(const V& v)                  const { return x * v.x + y * v.y + z * v.z;                                }
      inline F    length2()                        const { return thiss()->dot(*((V*)this));                                  }
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
   class V3fdt : public V3<V, F>
   {
   public:
      inline V3fdt()                                                    { }
      inline V3fdt(const F x, const F y, const F z) : V3<V, F>(x, y, z) { }
      inline V3fdt(const F s)                       : V3<V, F>(s, s, s) { }
      inline V3fdt(const F v[3])                    : V3<V, F>(v)       { }
      inline V3fdt(F* const v)                      : V3<V, F>(v)       { }
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
      inline V     roundByC(const F n)                  const { V c(*thiss()); c.roundBy(n); return c;               }
      inline void  round()                                    { x = V::_round(x); y = V::_round(y); z = V::_round(z); }
      inline void  floor()                                    { x = V::_floor(x); y = V::_floor(y); z = V::_floor(z); }
      inline void  ceil()                                     { x = V::_ceil(x);  y = V::_ceil(y);  z = V::_ceil(z);  }
      inline void  roundBy(const F n)                         { *thiss() /= n; thiss()->round(); *thiss() *= n;       }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V    randomN()                            { V t(V::random()); t.normalise(); return t;             }
      static inline void randomN(V* v, const size_t size)     { for (size_t i = 0; i < size; i++) v[i] = V::randomN(); }
      //------------------------------------------------------------------------------------------------------------------------//
   };

   /// <summary>
   /// Abstract 3D Vector for Integer (32/64) [L2]
   /// </summary>
   template <typename V, typename F>
   class V3ilt : public V3<V, F>
   {
   public:
      static inline F _abs(const int s)  { return ::abs(s);        }
      static inline F _sqrt(const int s) { return (F)::sqrt<F>(s); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V3ilt()                                                    { }
      inline V3ilt(const F x, const F y, const F z) : V3<V, F>(x, y, z) { }
      inline V3ilt(const F s)                       : V3<V, F>(s, s, s) { }
      inline V3ilt(const F v[3])                    : V3<V, F>(v)       { }
      inline V3ilt(F* const v)                      : V3<V, F>(v)       { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                       32-BIT & 64-BIT TEMPLATES [L3, ABSTRACT]                                         //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 3D Vector Template for Single Precision FP [L3]
   /// </summary>
   template <typename V>
   class V3ft : public V3fdt<V, float>
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
      inline V3ft()                                                                       { }
      inline V3ft(const float x, const float y, const float z) : V3fdt<V, float>(x, y, z) { }
      inline V3ft(const float s)                               : V3fdt<V, float>(s, s, s) { }
      inline V3ft(const float v[3])                            : V3fdt<V, float>(v)       { }
      inline V3ft(float* const v)                              : V3fdt<V, float>(v)       { }
   };

   /// <summary>
   /// Abstract 3D Vector Template for Double Precision FP [L3]
   /// </summary>
   template <typename V>
   class V3dt : public V3fdt<V, double>
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
      inline V3dt()                                                                           { }
      inline V3dt(const double x, const double y, const double z) : V3fdt<V, double>(x, y, z) { }
      inline V3dt(const double s)                                 : V3fdt<V, double>(s, s, s) { }
      inline V3dt(const double v[3])                              : V3fdt<V, double>(v)       { }
      inline V3dt(double* const v)                                : V3fdt<V, double>(v)       { }
   };

   /// <summary>
   /// Abstract 3D Vector Template for Integer (32) [L3]
   /// </summary>
   template <typename V>
   class V3it : public V3ilt<V, int>
   {
   public:
      inline V3it()                                                               { }
      inline V3it(const int x, const int y, const int z) : V3ilt<V, int>(x, y, z) { }
      inline V3it(const int s)                           : V3ilt<V, int>(s, s, s) { }
      inline V3it(const int v[3])                        : V3ilt<V, int>(v)       { }
      inline V3it(int* const v)                          : V3ilt<V, int>(v)       { }
   };

   /// <summary>
   /// Abstract 3D Vector Template for Integer (64) [L3]
   /// </summary>
   template <typename V>
   class V3lt : public V3ilt<V, long long>
   {
   public:
      inline V3lt()                                                                                       { }
      inline V3lt(const long long x, const long long y, const long long z) : V3ilt<V, long long>(x, y, z) { }
      inline V3lt(const long long s)                                       : V3ilt<V, long long>(s, s, s) { }
      inline V3lt(const long long v[3])                                    : V3ilt<V, long long>(v)       { }
      inline V3lt(long long* const v)                                      : V3ilt<V, long long>(v)       { }
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
      inline V3fg()                                                                                                     { }
      inline V3fg(const float x, const float y, const float z)    : V3ft<V3fg>(x, y, z)                                 { }
      inline V3fg(const float s)                                  : V3ft<V3fg>(s, s, s)                                 { }
      inline V3fg(const float v[3])                               : V3ft<V3fg>(v)                                       { }
      inline V3fg(float* const v)                                 : V3ft<V3fg>(v)                                       { }
      inline V3fg(const int v[3])                                 : V3ft<V3fg>((float)v[0], (float)v[1], (float)v[2])   { }
      inline V3fg(const double x, const double y, const double z) : V3ft<V3fg>((float)x,    (float)y,    (float)z)      { }
      inline V3fg(const int x, const int y, const int z)          : V3ft<V3fg>((float)x,    (float)y,    (float)z)      { }
   };

   /// <summary>
   /// Double Precision 3D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V3dg : public V3dt<V3dg>
   {
   public:
      inline V3dg()                                                                                                      { }
      inline V3dg(const double x, const double y, const double z) : V3dt<V3dg>(x, y, z)                                  { }
      inline V3dg(const double s)                                 : V3dt<V3dg>(s, s, s)                                  { }
      inline V3dg(const double v[3])                              : V3dt<V3dg>(v)                                        { }
      inline V3dg(double* const v)                                : V3dt<V3dg>(v)                                        { }
      inline V3dg(const int v[3])                                 : V3dt<V3dg>((double)v[0], (double)v[1], (double)v[2]) { }
      inline V3dg(const float x, const float y, const float z)    : V3dt<V3dg>((double)x,    (double)y,    (double)z)    { }
      inline V3dg(const int x, const int y, const int z)          : V3dt<V3dg>((double)x,    (double)y,    (double)z)    { }
   };

   /// <summary>
   /// 32-Bit Integer 3D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V3ig : public V3it<V3ig>
   {
   public:
      inline V3ig()                                                            { }
      inline V3ig(const int x, const int y, const int z) : V3it<V3ig>(x, y, z) { }
      inline V3ig(const int s)                           : V3it<V3ig>(s, s, s) { }
      inline V3ig(const int v[3])                        : V3it<V3ig>(v)       { }
      inline V3ig(int* const v)                          : V3it<V3ig>(v)       { }
   };

   /// <summary>
   /// 64-Bit Integer 3D Vector (Generic, no SIMD) [L4]
   /// </summary>//------------------------------------------------------------------------------------------------------------------------//
   class V3lg : public V3lt<V3lg>
   {
   public:
      inline V3lg()                                                                              { }
      inline V3lg(const long long x, const long long y, const long long z) : V3lt<V3lg>(x, y, z) { }
      inline V3lg(const long long s)                                       : V3lt<V3lg>(s, s, s) { }
      inline V3lg(const long long v[3])                                    : V3lt<V3lg>(v)       { }
      inline V3lg(long long* const v)                                      : V3lt<V3lg>(v)       { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                             SIMD CLASSES [L4]                                                          //
   //------------------------------------------------------------------------------------------------------------------------//
#if defined(SIMD_V3_FP_32_SSE2)
   /// <summary>
   /// Single Precision 3D Vector (unaligned SSE/SIMD)
   /// </summary>
   ALIGN16 class V3fs : public V3ft<V3fs>
   {
   public:
      inline __m128 load()                 const { return _mm_load_ps(vals);             }
      inline void   store(const __m128& v)       { _mm_store_ps((float*)vals, v);        }
      //------------------------------------------------------------------------------------------------------------------------//
      inline static __m128i unsetMaskHi()            { return _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF); }
      inline static __m128  unsetHi(const __m128& v) { return _mm_and_ps(v, _mm_castsi128_ps(unsetMaskHi()));                }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V3fs()                                               { }
      inline V3fs(const float x, const float y, const float z)    { store(_mm_set_ps(0.0f, z, y, x));                      }
      inline V3fs(const float s)                                  { store(_mm_set_ps1(s));                                 }
      inline V3fs(const double x, const double y, const double z) { store(_mm_set_ps(0.0f, (float)z, (float)y, (float)x)); }
      inline V3fs(const int x, const int y, const int z)          { store(_mm_set_ps(0.0f, (float)z, (float)y, (float)x)); }
      inline V3fs(const float v[3])                               { store(unsetHi(_mm_loadu_ps(v)));                       }
      inline V3fs(float* const v)                                 { store(unsetHi(_mm_loadu_ps(v)));                       }
      inline V3fs(const int v[3])                                 { store(unsetHi(_mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)v)))); }
      inline V3fs(const __m128& v)                                { store(v);                                             }
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
      inline void swap(V3fs& v)                                { __m128 t(load()); store(v.load()); v.store(t);               }
      inline V3fs absC()                                 const { return V3fs(_mm_andnot_ps(_mm_set1_ps(-0.f), load()));       }
      inline V3fs maxC(const V3fs& v)                    const { return V3fs(_mm_max_ps(load(), v.load()));                   }
      inline V3fs minC(const V3fs& v)                    const { return V3fs(_mm_min_ps(load(), v.load()));                   }
      inline V3fs boundC(const V3fs& mi, const V3fs& ma) const { return V3fs(_mm_max_ps(_mm_min_ps(load(), ma.load()), mi.load())); }
      inline void abs()                                        { store(_mm_andnot_ps(_mm_set1_ps(-0.), load()));              }
      inline void max(const V3fs& v)                           { store(_mm_max_ps(load(), v.load()));                         }
      inline void min(const V3fs& v)                           { store(_mm_min_ps(load(), v.load()));                         }
      inline void bound(const V3fs& mi, const V3fs& ma)        { store(_mm_max_ps(_mm_min_ps(load(), ma.load()), mi.load())); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool isZero()                         const { return *this == ZERO(); }
      
      inline bool isZero(const float e2)               const { return thiss()->length2() <= e2; }

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
#if defined(SIMD_V3_FP_32_SSE41)
      inline V3fs roundC() const { return V3fs(_mm_round_ps(load(), _MM_FROUND_NINT));  }
      inline V3fs floorC() const { return V3fs(_mm_round_ps(load(), _MM_FROUND_FLOOR)); }
      inline V3fs ceilC()  const { return V3fs(_mm_round_ps(load(), _MM_FROUND_CEIL));  }
      inline void round()        { store(_mm_round_ps(load(), _MM_FROUND_NINT));        }
      inline void floor()        { store(_mm_round_ps(load(), _MM_FROUND_FLOOR));       }
      inline void ceil()         { store(_mm_round_ps(load(), _MM_FROUND_CEIL));        }
      inline void roundBy(const float n) { store(_mm_mul_ps(_mm_round_ps(_mm_mul_ps(load(), _mm_set1_ps(1.0f / n)), _MM_FROUND_NINT), _mm_set1_ps(n))); }
#endif
      //------------------------------------------------------------------------------------------------------------------------//
   };
   typedef V3fs V3f;  // use SIMD as default
#else
   typedef V3fg V3f;  // use plain as default
#endif

#pragma endregion


   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//
   //------------------------------------------------------------------------------------------------------------------------//

#pragma region V4
   //------------------------------------------------------------------------------------------------------------------------//
   //                                          ROOT TEMPLATE [L1, ABSTRACT]                                                  //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 4D Vector Template for Floating Point (32/64) AND Integer (32/64). [L1]
   /// </summary>
   template <typename V, typename F>
   class V4
   {
   protected:
      inline V*   thiss() const { return ((V*)this); }

   public:
      union
      {
         struct { F x, y, z, w; };
         F vals[4];
      };
      //------------------------------------------------------------------------------------------------------------------------//
      inline V4()                                                                                { }
      inline V4(const F x, const F y, const F z, const F w) : x(x), y(y), z(z), w(w)             { }
      inline V4(const F s)                                  : x(s), y(s), z(s), w(s)             { }
      inline V4(const F v[4])                               : x(v[0]), y(v[1]), z(v[2]), w(v[3]) { }
      inline V4(F* const v)                                 : x(v[0]), y(v[1]), z(v[2]), w(v[3]) { }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V ZERO()  { return V((F)0.0, (F)0.0, (F)0.0, (F)0.0); }
      static inline V UNITX() { return V((F)1.0, (F)0.0, (F)0.0, (F)0.0); }
      static inline V UNITY() { return V((F)0.0, (F)1.0, (F)0.0, (F)0.0); }
      static inline V UNITZ() { return V((F)0.0, (F)0.0, (F)1.0, (F)0.0); }
      static inline V UNITW() { return V((F)0.0, (F)0.0, (F)0.0, (F)1.0); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void* operator new  (size_t size)          { return malloc(sizeof(V));        }
      inline void* operator new[](size_t size)          { return malloc(size * sizeof(V)); }
      inline F     operator []   (const size_t i) const { return vals[i];                  }
      inline F     operator []   (const size_t i)       { return vals[i];                  }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  operator ==   (const V& v)     const { return ((x == v.x) & (y == v.y) & (z == v.z) & (w == v.w)); }
      inline bool  operator !=   (const V& v)     const { return ((x != v.x) | (y != v.y) | (z != v.z) | (w != v.w)); }
      inline bool  operator <    (const V& v)     const { return ((x <  v.x) & (y <  v.y) & (z <  v.z) & (w <  v.w)); }
      inline bool  operator <=   (const V& v)     const { return ((x <= v.x) & (y <= v.y) & (z <= v.z) & (w <= v.w)); }
      inline bool  operator >    (const V& v)     const { return ((x >  v.x) & (y >  v.y) & (z >  v.z) & (w >  v.w)); }
      inline bool  operator >=   (const V& v)     const { return ((x >= v.x) & (y >= v.y) & (z >= v.z) & (w >= v.w)); }
      inline V     operator +    (const V& v)     const { return V(x + v.x, y + v.y, z + v.z, w + v.w);               }
      inline V     operator -    (const V& v)     const { return V(x - v.x, y - v.y, z - v.z, w - v.w);               }
      inline V     operator *    (const V& v)     const { return V(x * v.x, y * v.y, z * v.z, w * v.w);               }
      inline V     operator /    (const V& v)     const { return V(x / v.x, y / v.y, z / v.z, w / v.w);               }
      inline V     operator *    (const F  s)     const { return V(x * s,   y * s,   z * s,   w * s);                 }
      inline V     operator /    (const F  s)     const { return V(x / s,   y / s,   z / s,   w / s);                 }
      inline V&    operator =    (const V& v)           { x  = v.x; y  = v.y; z  = v.z; w  = v.w; return *thiss();    }
      inline V&    operator +=   (const V& v)           { x += v.x; y += v.y; z += v.z; w += v.w; return *thiss();    }
      inline V&    operator -=   (const V& v)           { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *thiss();    }
      inline V&    operator *=   (const V& v)           { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *thiss();    }
      inline V&    operator /=   (const V& v)           { x /= v.x; y /= v.y; z /= v.z; w /= v.w; return *thiss();    }
      inline V&    operator =    (const F  s)           { x  = s;   y  = s;   z  = s;   w  = s;   return *thiss();    }
      inline V&    operator +=   (const F  s)           { x += s;   y += s;   z += s;   w += s;   return *thiss();    }
      inline V&    operator -=   (const F  s)           { x -= s;   y -= s;   z -= s;   w -= s;   return *thiss();    }
      inline V&    operator *=   (const F  s)           { x *= s;   y *= s;   z *= s;   w *= s;   return *thiss();    }
      inline V&    operator /=   (const F  s)           { x /= s;   y /= s;   z /= s;   w /= s;   return *thiss();    }
      //------------------------------------------------------------------------------------------------------------------------//
      inline       V  operator - ()               const { return V(-x, -y, -z, -w); }
      inline const V& operator + ()               const { return *this;             }
      //------------------------------------------------------------------------------------------------------------------------//
      inline void madd(const V& m, const V& a)           { *thiss() = (*thiss() * m) + a;                                    }
      inline bool isZero()                         const { return x == (F)0.0 && y == (F)0.0 && z == (F)0.0 && w == (F)0.0;  }
      inline bool isZero(const F e2)               const { return thiss()->length2() <= e2;                                  }
      inline bool equals(const V& v, const F e2)   const { return (thiss() - v).length2() <= e2;                             }
      inline void swap(V& v)                             { std::swap(x, v.x); std::swap(y, v.y); std::swap(z, v.z); std::swap(w, v.w); }
      inline F    dot(const V& v)                  const { return x * v.x + y * v.y + z * v.z + w * v.w;                     }
      inline F    length2()                        const { return thiss()->dot(*((V*)this));                                 }
      inline F    length()                         const { return V::_sqrt(thiss()->length2());                              }
      inline F    distance2(const V& v)            const { return (thiss() - v).length2();                                   }
      inline F    distance(const V& v)             const { return V::_sqrt(thiss()->distance2(v));                           }
      inline V    maxC(const V& v)                 const { return V(v.x > x ? v.x : x, v.y > y ? v.y : y, v.z > z ? v.z : z, v.w > w ? v.w : w); }
      inline V    minC(const V& v)                 const { return V(v.x < x ? v.x : x, v.y < y ? v.y : y, v.z < z ? v.z : z, v.w < w ? v.w : w); }
      inline V    boundC(const V& mi, const V& ma) const { V t(thiss()->minC(ma)); t.max(mi); return t;                      }
      inline V    absC()                           const { return V(V::_abs(x), V::_abs(y), V::_abs(z), V::_abs(w));         }
      inline void max(const V& v)                        { if (v.x > x) x = v.x; if (v.y > y) y = v.y; if (v.z > z) z = v.z; if (v.w > w) w = v.w; }
      inline void min(const V& v)                        { if (v.x < x) x = v.x; if (v.y < y) y = v.y; if (v.z < z) z = v.z; if (v.w < w) w = v.w; }
      inline void bound(const V& mi, const V& ma)        { thiss()->min(ma); thiss()->max(mi);                               }
      inline void abs()                                  { x = V::_abs(x); y = V::_abs(y); z = V::_abs(z); w = V::_abs(w);   }
      //------------------------------------------------------------------------------------------------------------------------//
      //inline V    xzy()                            const { return V(x, z, y); }
      //inline void xzy() { std::swap(y, z); }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V    random()                        { return V(std::rand(), std::rand(), std::rand(), std::rand()); }
      static inline void random(V* v, const size_t size) { for (size_t i = 0; i < size; i++) v[i] = V::random();         }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                 FLOATING POINT & INTEGER TEMPLATES V3 [L2, ABSTRACT]                                   //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 4D Vector Template for Floating Point (32/64) [L2]
   /// </summary>
   template <typename V, typename F>
   class V4fdt : public V4<V, F>
   {
   public:
      inline V4fdt()                                                                  { }
      inline V4fdt(const F x, const F y, const F z, const F w) : V4<V, F>(x, y, z, w) { }
      inline V4fdt(const F s)                                  : V4<V, F>(s, s, s, s) { }
      inline V4fdt(const F v[4])                               : V4<V, F>(v)          { }
      inline V4fdt(F* const v)                                 : V4<V, F>(v)          { }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V& operator /= (const V& v)       { x /= v.x; y /= v.y; z /= v.z; w /= v.w; return (V&)*this;           }
      inline V  operator /  (const V& v) const { return V(x / v.x, y / v.y, z / v.z, w / v.w);                       }
      inline V& operator /= (const F  s)       { F t = (F)1.0 / s; x *= t; y *= t; z *= t; w *= t; return (V&)*this; }
      inline V  operator /  (const F  s) const { F t = (F)1.0 / s; return V(x * t, y * t, z * t, w * t);             }
      //------------------------------------------------------------------------------------------------------------------------//
      inline bool  isNaN()                              const { return isnan<F>(x) || isnan<F>(y) || isnan<F>(z) || isnan<F>(w); }
      inline void  normalise()                                { *thiss() /= thiss()->length();                                   }
      inline V     normaliseC()                         const { V t(*thiss()); t.normalise(); return t;                          }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V     roundC()                             const { return V(V::_round(x), V::_round(y), V::_round(z), V::_round(w)); }
      inline V     floorC()                             const { return V(V::_floor(x), V::_floor(y), V::_floor(z), V::_floor(w)); }
      inline V     ceilC()                              const { return V(V::_ceil(x),  V::_ceil(y),  V::_ceil(z),  V::_ceil(w));  }
      inline V     roundByC(const F n)                  const { V c(*thiss()); c.roundBy(n); return c;                            }
      inline void  round()                                    { x = V::_round(x); y = V::_round(y); z = V::_round(z); w = V::_round(w); }
      inline void  floor()                                    { x = V::_floor(x); y = V::_floor(y); z = V::_floor(z); w = V::_floor(w); }
      inline void  ceil()                                     { x = V::_ceil(x);  y = V::_ceil(y);  z = V::_ceil(z);  w = V::_ceil(w);  }
      inline void  roundBy(const F n)                         { *thiss() /= n; thiss()->round(); *thiss() *= n;                   }
      //------------------------------------------------------------------------------------------------------------------------//
      static inline V    randomN() { V t(V::random()); t.normalise(); return t; }
      static inline void randomN(V* v, const size_t size) { for (size_t i = 0; i < size; i++) v[i] = V::randomN(); }
      //------------------------------------------------------------------------------------------------------------------------//
   };

   /// <summary>
   /// Abstract 4D Vector for Integer (32/64) [L2]
   /// </summary>
   template <typename V, typename F>
   class V4ilt : public V4<V, F>
   {
   public:
      static inline F _abs(const int s) { return ::abs(s); }
      static inline F _sqrt(const int s) { return (F)::sqrt<F>(s); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V4ilt()                                                                  { }
      inline V4ilt(const F x, const F y, const F z, const F w) : V4<V, F>(x, y, z, w) { }
      inline V4ilt(const F s)                                  : V4<V, F>(s, s, s, s) { }
      inline V4ilt(const F v[4])                               : V4<V, F>(v)          { }
      inline V4ilt(F* const v)                                 : V4<V, F>(v)          { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                       32-BIT & 64-BIT TEMPLATES [L3, ABSTRACT]                                         //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Abstract 4D Vector Template for Single Precision FP [L3]
   /// </summary>
   template <typename V>
   class V4ft : public V4fdt<V, float>
   {
   public:
      static inline float _abs(const float s) { return ::fabsf(s); }
      static inline float _round(const float s) { return ::roundf(s); }
      static inline float _floor(const float s) { return ::floorf(s); }
      static inline float _ceil(const float s) { return ::ceilf(s); }
      static inline float _sqrt(const float s) { return ::sqrtf(s); }
      static inline float _cos(const float s) { return ::cosf(s); }
      static inline float _sin(const float s) { return ::sinf(s); }
      static inline float _acos(const float s) { return ::acosf(s); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V4ft()                                                                                         { }
      inline V4ft(const float x, const float y, const float z, const float w) : V4fdt<V, float>(x, y, z, w) { }
      inline V4ft(const float s)                                              : V4fdt<V, float>(s, s, s, s) { }
      inline V4ft(const float v[4])                                           : V4fdt<V, float>(v) { }
      inline V4ft(float* const v)                                             : V4fdt<V, float>(v) { }
   };

   /// <summary>
   /// Abstract 4D Vector Template for Double Precision FP [L3]
   /// </summary>
   template <typename V>
   class V4dt : public V4fdt<V, double>
   {
   public:
      static inline double _abs(const double s) { return ::abs(s); }
      static inline double _round(const double s) { return ::round(s); }
      static inline double _floor(const double s) { return ::floor(s); }
      static inline double _ceil(const double s) { return ::ceil(s); }
      static inline double _sqrt(const double s) { return ::sqrt(s); }
      static inline double _cos(const double s) { return ::cos(s); }
      static inline double _sin(const double s) { return ::sin(s); }
      static inline double _acos(const double s) { return ::acos(s); }
      //------------------------------------------------------------------------------------------------------------------------//
      inline V4dt() { }
      inline V4dt(const double x, const double y, const double z, const double w) : V4fdt<V, double>(x, y, z, w) { }
      inline V4dt(const double s)                                                 : V4fdt<V, double>(s, s, s, s) { }
      inline V4dt(const double v[4])                                              : V4fdt<V, double>(v)          { }
      inline V4dt(double* const v)                                                : V4fdt<V, double>(v)          { }
   };

   /// <summary>
   /// Abstract 4D Vector Template for Integer (32) [L3]
   /// </summary>
   template <typename V>
   class V4it : public V4ilt<V, int>
   {
   public:
      inline V4it() { }
      inline V4it(const int x, const int y, const int z, const int w) : V4ilt<V, int>(x, y, z, w) { }
      inline V4it(const int s)                                        : V4ilt<V, int>(s, s, s, s) { }
      inline V4it(const int v[4])                                     : V4ilt<V, int>(v)          { }
      inline V4it(int* const v)                                       : V4ilt<V, int>(v)          { }
   };

   /// <summary>
   /// Abstract 4D Vector Template for Integer (64) [L3]
   /// </summary>
   template <typename V>
   class V4lt : public V4ilt<V, long long>
   {
   public:
      inline V4lt() { }
      inline V4lt(const long long x, const long long y, const long long z, const long long w) : V4ilt<V, long long>(x, y, z, w) { }
      inline V4lt(const long long s)                                                          : V4ilt<V, long long>(s, s, s, s) { }
      inline V4lt(const long long v[4])                                                       : V4ilt<V, long long>(v)          { }
      inline V4lt(long long* const v)                                                         : V4ilt<V, long long>(v)          { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                     GENERIC/NON-SIMD CLASSES [L4]                                                      //
   //------------------------------------------------------------------------------------------------------------------------//
   /// <summary>
   /// Single Precision 4D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V4fg : public V4ft<V4fg>
   {
   public:
      inline V4fg()                                                                                        { }
      inline V4fg(const float x, const float y, const float z, const float w)     : V4ft<V4fg>(x, y, z, w) { }
      inline V4fg(const float s)                                                  : V4ft<V4fg>(s, s, s, s) { }
      inline V4fg(const float v[4])                                               : V4ft<V4fg>(v)          { }
      inline V4fg(float* const v)                                                 : V4ft<V4fg>(v)          { }
      inline V4fg(const int v[4])                                                 : V4ft<V4fg>((float)v[0], (float)v[1], (float)v[2], (float)v[3]) { }
      inline V4fg(const double x, const double y, const double z, const double w) : V4ft<V4fg>((float)x, (float)y, (float)z, (float)w)             { }
      inline V4fg(const int x, const int y, const int z, const int w)             : V4ft<V4fg>((float)x, (float)y, (float)z, (float)w)             { }
   };

   /// <summary>
   /// Double Precision 4D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V4dg : public V4dt<V4dg>
   {
   public:
      inline V4dg()                                                                                        { }
      inline V4dg(const double x, const double y, const double z, const double w) : V4dt<V4dg>(x, y, z, w) { }
      inline V4dg(const double s)                                                 : V4dt<V4dg>(s, s, s, s) { }
      inline V4dg(const double v[4])                                              : V4dt<V4dg>(v)          { }
      inline V4dg(double* const v)                                                : V4dt<V4dg>(v)          { }
      inline V4dg(const int v[4])                                                 : V4dt<V4dg>((double)v[0], (double)v[1], (double)v[2], (double)v[3]) { }
      inline V4dg(const float x, const float y, const float z, const float w)     : V4dt<V4dg>((double)x, (double)y, (double)z, (double)w)             { }
      inline V4dg(const int x, const int y, const int z, const int w)             : V4dt<V4dg>((double)x, (double)y, (double)z, (double)w)             { }
   };

   /// <summary>
   /// 32-Bit Integer 4D Vector (Generic, no SIMD) [L4]
   /// </summary>
   class V4ig : public V4it<V4ig>
   {
   public:
      inline V4ig()                                                                            { }
      inline V4ig(const int x, const int y, const int z, const int w) : V4it<V4ig>(x, y, z, w) { }
      inline V4ig(const int s)                                        : V4it<V4ig>(s, s, s, s) { }
      inline V4ig(const int v[4])                                     : V4it<V4ig>(v)          { }
      inline V4ig(int* const v)                                       : V4it<V4ig>(v)          { }
   };

   /// <summary>
   /// 64-Bit Integer 4D Vector (Generic, no SIMD) [L4]
   /// </summary>//------------------------------------------------------------------------------------------------------------------------//
   class V4lg : public V4lt<V4lg>
   {
   public:
      inline V4lg()                                                                                                    { }
      inline V4lg(const long long x, const long long y, const long long z, const long long w) : V4lt<V4lg>(x, y, z, w) { }
      inline V4lg(const long long s)                                                          : V4lt<V4lg>(s, s, s, s) { }
      inline V4lg(const long long v[4])                                                       : V4lt<V4lg>(v)          { }
      inline V4lg(long long* const v)                                                         : V4lt<V4lg>(v)          { }
   };

   //------------------------------------------------------------------------------------------------------------------------//
   //                                             SIMD CLASSES [L4]                                                          //
   //------------------------------------------------------------------------------------------------------------------------//

   typedef V4fg V4f;  // use plain as default
   typedef V4dg V4d;  // use plain as default
   typedef V4ig V4i;  // use plain as default
   typedef V4lg V4l;  // use plain as default

#pragma endregion

}

#include "geometry.h"
#include <chrono>
//------------------------------------------------------------------------------------------------------------------------//
using namespace simd;
using namespace std::chrono;
using namespace std;
//------------------------------------------------------------------------------------------------------------------------//
#define LOOPS 50000
#define LOOP(n, x) __pragma(loop(no_vector)) for (size_t i = 0; i < n; i++) { x }
#define PRINTHEADER                                               \
  printf("-----------------------------------------------------------------------------------------------------------\n"); \
  printf("CLS  |  OPERATION   |    TIME    |        VAL      || CLS  |  OPERATION   |    TIME    |        VAL      ||\n"); \
  printf("-----------------------------------------------------------------------------------------------------------\n");
//------------------------------------------------------------------------------------------------------------------------//
high_resolution_clock clk;
//------------------------------------------------------------------------------------------------------------------------//
template<typename T>
void bench(const char* name, const char* op, void task(T* ARR))
{
   T ARR[LOOPS];
   T::randomN(ARR, LOOPS);
   auto t1 = clk.now();
   task(ARR);
   auto t2 = clk.now();
   auto sp = t2 - t1;
   T s = T::ZERO(); for (size_t i = 0; i < LOOPS; i++) s += ARR[i];
   printf(name); printf(" | "); printf(op); printf("%010I64i | ", sp.count()); printf("%+015.7f || ", s.x + s.y);
}
//------------------------------------------------------------------------------------------------------------------------//
template<typename T, typename F> void benchEqV(T* ARR)       { LOOP(LOOPS - 1, ARR[i].x = (F)(ARR[i] == ARR[i + 1]);)  }
template<typename T, typename F> void benchNeqV(T* ARR)      { LOOP(LOOPS - 1, ARR[i].x = (F)(ARR[i] != ARR[i + 1]);)  }
template<typename T, typename F> void benchLtV(T* ARR)       { LOOP(LOOPS - 1, ARR[i].x = (F)(ARR[i] < ARR[i + 1]);)   }
template<typename T, typename F> void benchLeV(T* ARR)       { LOOP(LOOPS - 1, ARR[i].x = (F)(ARR[i] <= ARR[i + 1]);)  }
template<typename T, typename F> void benchGtV(T* ARR)       { LOOP(LOOPS - 1, ARR[i].x = (F)(ARR[i] > ARR[i + 1]);)   }
template<typename T, typename F> void benchGeV(T* ARR)       { LOOP(LOOPS - 1, ARR[i].x = (F)(ARR[i] >= ARR[i + 1]);)  }
template<typename T, typename F> void benchAddV(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] + ARR[i + 1];)        }
template<typename T, typename F> void benchSubV(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] - ARR[i + 1];)        }
template<typename T, typename F> void benchMulV(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] * ARR[i + 1];)        }
template<typename T, typename F> void benchDivV(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] / ARR[i + 1];)        }
template<typename T, typename F> void benchAddS(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] + ARR[i + 1].x;)      }
template<typename T, typename F> void benchSubS(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] - ARR[i + 1].x;)      }
template<typename T, typename F> void benchMulS(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] * ARR[i + 1].x;)      }
template<typename T, typename F> void benchDivS(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] / ARR[i + 1].x;)      }
template<typename T, typename F> void benchIsZero(T* ARR)    { LOOP(LOOPS, ARR[i].x = (F)ARR[i].isZero();)           }
template<typename T, typename F> void benchIsZeroE(T* ARR)   { LOOP(LOOPS, ARR[i].x = (F)ARR[i].isZero((F)0.01);)    }
template<typename T, typename F> void benchRotate(T* ARR)    { LOOP(LOOPS, ARR[i].rotate((F)M_PI_2);)                }
template<typename T, typename F> void benchLength(T* ARR)    { LOOP(LOOPS, ARR[i].x = ARR[i].length();)              }
template<typename T, typename F> void benchLength2(T* ARR)   { LOOP(LOOPS, ARR[i].x = ARR[i].length2();)             }
template<typename T, typename F> void benchNormalise(T* ARR) { LOOP(LOOPS, ARR[i].normalise();)                      }
template<typename T, typename F> void benchRound(T* ARR)     { LOOP(LOOPS, ARR[i].round();)                          }
template<typename T, typename F> void benchFloor(T* ARR)     { LOOP(LOOPS, ARR[i].floor();)                          }
template<typename T, typename F> void benchCeil(T* ARR)      { LOOP(LOOPS, ARR[i].ceil();)                           }
template<typename T, typename F> void benchAbs(T* ARR)       { LOOP(LOOPS, ARR[i].abs();)                            }
template<typename T, typename F> void benchSwap(T* ARR)      { LOOP(LOOPS - 1, ARR[i].swap(ARR[i + 1]);)             }
template<typename T, typename F> void benchDot(T* ARR)       { LOOP(LOOPS - 1, ARR[i].x = ARR[i].dot(ARR[i + 1]);)   }
template<typename T, typename F> void benchCross(T* ARR)     { LOOP(LOOPS - 1, ARR[i].x = ARR[i].cross(ARR[i + 1]);) }
template<typename T, typename F> void benchMax(T* ARR)       { LOOP(LOOPS - 1, ARR[i].max(ARR[i + 1]);)              }
template<typename T, typename F> void benchMin(T* ARR)       { LOOP(LOOPS - 1, ARR[i].min(ARR[i + 1]);)              }
template<typename T, typename F> void benchBound(T* ARR)     { LOOP(LOOPS - 2, ARR[i].bound(ARR[i + 1], ARR[i + 2]);) }
template<typename T, typename F> void benchSide(T* ARR)      { LOOP(LOOPS - 2, ARR[i].x = ARR[i].side(ARR[i + 1], ARR[i + 2]);)                    }
template<typename T, typename F> void benchInsideR(T* ARR)   { LOOP(LOOPS - 2, ARR[i].x = (F)ARR[i].inside(ARR[i + 1], ARR[i + 2]);)          }
template<typename T, typename F> void benchInsideRE(T* ARR)  { LOOP(LOOPS - 2, ARR[i].x = (F)ARR[i].inside(ARR[i + 1], ARR[i + 2], (F)0.001);)   }
template<typename T, typename F> void benchInsideC(T* ARR)   { LOOP(LOOPS - 2, ARR[i].x = (F)ARR[i].inside(ARR[i + 1], ARR[i + 2].x);)        }
template<typename T, typename F> void benchInsideCE(T* ARR)  { LOOP(LOOPS - 2, ARR[i].x = (F)ARR[i].inside(ARR[i + 1], ARR[i + 2].x, (F)0.001);) }
template<typename T, typename F> void benchAreaTri(T* ARR)   { LOOP(LOOPS - 2, ARR[i].x = ARR[i].area(ARR[i + 1], ARR[i + 2]);)                    }
template<typename T, typename F> void benchAngle(T* ARR)     { LOOP(LOOPS, ARR[i].x = ARR[i].angle();)                                             }
template<typename T, typename F> void benchAngleV(T* ARR)    { LOOP(LOOPS - 1, ARR[i].x = ARR[i].angle(ARR[i + 1]);)                               }
template<typename T, typename T2, typename F> void benchRun(const char* name, const char* nameT2)
{
   PRINTHEADER;
   bench<T>(name, "operator ==v | ", benchLtV<T, F>);       bench<T2>(nameT2, "operator ==v | ", benchLtV<T2, F>);       printf("\n");
   bench<T>(name, "operator !=v | ", benchLtV<T, F>);       bench<T2>(nameT2, "operator !=v | ", benchLtV<T2, F>);       printf("\n");
   bench<T>(name, "operator < v | ", benchLtV<T, F>);       bench<T2>(nameT2, "operator < v | ", benchLtV<T2, F>);       printf("\n");
   bench<T>(name, "operator <=v | ", benchLeV<T, F>);       bench<T2>(nameT2, "operator <=v | ", benchLeV<T2, F>);       printf("\n");
   bench<T>(name, "operator > v | ", benchGtV<T, F>);       bench<T2>(nameT2, "operator > v | ", benchGtV<T2, F>);       printf("\n");
   bench<T>(name, "operator >=v | ", benchGeV<T, F>);       bench<T2>(nameT2, "operator >=v | ", benchGeV<T2, F>);       printf("\n");
   bench<T>(name, "operator + v | ", benchAddV<T, F>);      bench<T2>(nameT2, "operator + v | ", benchAddV<T2, F>);      printf("\n");
   bench<T>(name, "operator - v | ", benchSubV<T, F>);      bench<T2>(nameT2, "operator - v | ", benchSubV<T2, F>);      printf("\n");
   bench<T>(name, "operator * v | ", benchMulV<T, F>);      bench<T2>(nameT2, "operator * v | ", benchMulV<T2, F>);      printf("\n");
   bench<T>(name, "operator / v | ", benchDivV<T, F>);      bench<T2>(nameT2, "operator / v | ", benchDivV<T2, F>);      printf("\n");
   bench<T>(name, "operator + s | ", benchAddS<T, F>);      bench<T2>(nameT2, "operator + s | ", benchAddS<T2, F>);      printf("\n");
   bench<T>(name, "operator - s | ", benchSubS<T, F>);      bench<T2>(nameT2, "operator - s | ", benchSubS<T2, F>);      printf("\n");
   bench<T>(name, "operator * s | ", benchMulS<T, F>);      bench<T2>(nameT2, "operator * s | ", benchMulS<T2, F>);      printf("\n");
   bench<T>(name, "operator / s | ", benchDivS<T, F>);      bench<T2>(nameT2, "operator / s | ", benchDivS<T2, F>);      printf("\n");
   bench<T>(name, "isZero()     | ", benchIsZero<T, F>);    bench<T2>(nameT2, "isZero()     | ", benchIsZero<T2, F>);    printf("\n");
   bench<T>(name, "isZero(e)    | ", benchIsZeroE<T, F>);   bench<T2>(nameT2, "isZero(e)    | ", benchIsZeroE<T2, F>);   printf("\n");
   bench<T>(name, "rotate()     | ", benchRotate<T, F>);    bench<T2>(nameT2, "rotate()     | ", benchRotate<T2, F>);    printf("\n");
   bench<T>(name, "length()     | ", benchLength<T, F>);    bench<T2>(nameT2, "length()     | ", benchLength<T2, F>);    printf("\n");
   bench<T>(name, "length2()    | ", benchLength2<T, F>);   bench<T2>(nameT2, "length2()    | ", benchLength2<T2, F>);   printf("\n");
   bench<T>(name, "normalise()  | ", benchNormalise<T, F>); bench<T2>(nameT2, "normalise()  | ", benchNormalise<T2, F>); printf("\n");
   bench<T>(name, "round()      | ", benchRound<T, F>);     bench<T2>(nameT2, "round()      | ", benchRound<T2, F>);     printf("\n");
   bench<T>(name, "floor()      | ", benchFloor<T, F>);     bench<T2>(nameT2, "floor()      | ", benchFloor<T2, F>);     printf("\n");
   bench<T>(name, "ceil()       | ", benchCeil<T, F>);      bench<T2>(nameT2, "ceil()       | ", benchCeil<T2, F>);      printf("\n");
   bench<T>(name, "abs()        | ", benchAbs<T, F>);       bench<T2>(nameT2, "abs()        | ", benchAbs<T2, F>);       printf("\n");
   bench<T>(name, "swap()       | ", benchSwap<T, F>);      bench<T2>(nameT2, "swap()       | ", benchSwap<T2, F>);      printf("\n");
   bench<T>(name, "dot()        | ", benchDot<T, F>);       bench<T2>(nameT2, "dot()        | ", benchDot<T2, F>);       printf("\n");
   bench<T>(name, "cross()      | ", benchCross<T, F>);     bench<T2>(nameT2, "cross()      | ", benchCross<T2, F>);     printf("\n");
   bench<T>(name, "max()        | ", benchMax<T, F>);       bench<T2>(nameT2, "max()        | ", benchMax<T2, F>);       printf("\n");
   bench<T>(name, "min()        | ", benchMin<T, F>);       bench<T2>(nameT2, "min()        | ", benchMin<T2, F>);       printf("\n");
   bench<T>(name, "bound()      | ", benchBound<T, F>);     bench<T2>(nameT2, "bound()      | ", benchBound<T2, F>);     printf("\n");
   bench<T>(name, "side(v,v)    | ", benchSide<T, F>);      bench<T2>(nameT2, "side(v,v)    | ", benchSide<T2, F>);      printf("\n");
   bench<T>(name, "inside(v,v)  | ", benchInsideR<T, F>);   bench<T2>(nameT2, "inside(v,v)  | ", benchInsideR<T2, F>);   printf("\n");
   bench<T>(name, "inside(v,v,e)| ", benchInsideRE<T, F>);  bench<T2>(nameT2, "inside(v,v,e)| ", benchInsideRE<T2, F>);  printf("\n");
   bench<T>(name, "inside(m,r)  | ", benchInsideC<T, F>);   bench<T2>(nameT2, "inside(m,r)  | ", benchInsideC<T2, F>);   printf("\n");
   bench<T>(name, "inside(m,r,e)| ", benchInsideCE<T, F>);  bench<T2>(nameT2, "inside(m,r,e)| ", benchInsideCE<T2, F>);  printf("\n");
   bench<T>(name, "area(v,v)    | ", benchAreaTri<T, F>);   bench<T2>(nameT2, "area(v,v)    | ", benchAreaTri<T2, F>);   printf("\n");
   bench<T>(name, "angle()      | ", benchAngle<T, F>);     bench<T2>(nameT2, "angle()      | ", benchAngle<T2, F>);     printf("\n");
   bench<T>(name, "angle(v)     | ", benchAngleV<T, F>);    bench<T2>(nameT2, "angle(v)     | ", benchAngleV<T2, F>);    printf("\n");
}
//------------------------------------------------------------------------------------------------------------------------//

void printSizes()
{
   V2f ff(1.0f, 1.0f);
   V2d fd(1.0, 1.0);

   size_t siz = sizeof(ff);
   printf("%i", siz);

   siz = sizeof(fd);
   printf("%i", siz);

}

int main()
{
   while (true)
   {
#if defined(SIMD_V2_32_SSE2)
      benchRun<V2dg, V2ds, double>("V2dg", "V2ds");
#else
      benchRun<V2dg, V2dg, double>("V2dg", "V2dg");
#endif

#if defined(SIMD_V2_64_SSE2)
      benchRun<V2fg, V2fs, float>("V2fg", "V2fs");
#else
      benchRun<V2fg, V2fg, float>("V2fg", "V2fg");
#endif
      getchar();
   }

   return 0;
}
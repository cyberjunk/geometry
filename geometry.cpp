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
  printf("------------------------------------------------- \n"); \
  printf("CLS |  OPERATION   |    TIME    |    VAL          \n"); \
  printf("------------------------------------------------- \n");
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
   printf(name); printf(" | "); printf(op); printf("%010I64i | ", sp.count()); printf("%+014.7f \n", s.x + s.y);
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
template<typename T, typename F> void benchRun(const char* name)
{
   PRINTHEADER;
   bench<T>(name, "operator ==v | ", benchLtV<T, F>);
   bench<T>(name, "operator !=v | ", benchLtV<T, F>);
   bench<T>(name, "operator < v | ", benchLtV<T, F>);
   bench<T>(name, "operator <=v | ", benchLeV<T, F>);
   bench<T>(name, "operator > v | ", benchGtV<T, F>);
   bench<T>(name, "operator >=v | ", benchGeV<T, F>);
   bench<T>(name, "operator + v | ", benchAddV<T, F>);
   bench<T>(name, "operator - v | ", benchSubV<T, F>);
   bench<T>(name, "operator * v | ", benchMulV<T, F>);
   bench<T>(name, "operator / v | ", benchDivV<T, F>);
   bench<T>(name, "operator + s | ", benchAddS<T, F>);
   bench<T>(name, "operator - s | ", benchSubS<T, F>);
   bench<T>(name, "operator * s | ", benchMulS<T, F>);
   bench<T>(name, "operator / s | ", benchDivS<T, F>);
   bench<T>(name, "isZero()     | ", benchIsZero<T, F>);
   bench<T>(name, "isZero(e)    | ", benchIsZeroE<T, F>);
   bench<T>(name, "rotate()     | ", benchRotate<T, F>);
   bench<T>(name, "length()     | ", benchLength<T, F>);
   bench<T>(name, "length2()    | ", benchLength2<T, F>);
   bench<T>(name, "normalise()  | ", benchNormalise<T, F>);
   bench<T>(name, "round()      | ", benchRound<T, F>);
   bench<T>(name, "floor()      | ", benchFloor<T, F>);
   bench<T>(name, "ceil()       | ", benchCeil<T, F>);
   bench<T>(name, "abs()        | ", benchAbs<T, F>);
   bench<T>(name, "swap()       | ", benchSwap<T, F>);
   bench<T>(name, "dot()        | ", benchDot<T, F>);
   bench<T>(name, "cross()      | ", benchCross<T, F>);
   bench<T>(name, "max()        | ", benchMax<T, F>);
   bench<T>(name, "min()        | ", benchMin<T, F>);
   bench<T>(name, "bound()      | ", benchBound<T, F>);
   bench<T>(name, "side(v,v)    | ", benchSide<T, F>);
   bench<T>(name, "inside(v,v)  | ", benchInsideR<T, F>);
   bench<T>(name, "inside(v,v,e)| ", benchInsideRE<T, F>);
   bench<T>(name, "inside(m,r)  | ", benchInsideC<T, F>);
   bench<T>(name, "inside(m,r,e)| ", benchInsideCE<T, F>);
   bench<T>(name, "area(v,v)    | ", benchAreaTri<T, F>);
   bench<T>(name, "angle()      | ", benchAngle<T, F>);
   bench<T>(name, "angle(v)     | ", benchAngleV<T, F>);
}
//------------------------------------------------------------------------------------------------------------------------//

int main()
{
   while (true)
   {
      
      benchRun<V2d, double>("V2d");
      benchRun<V2f, float>("V2f");
      getchar();
   }

   return 0;
}
#include "geometry.h"
#include <chrono>
//------------------------------------------------------------------------------------------------------------------------//
using namespace simd;
using namespace std::chrono;
using namespace std;
//------------------------------------------------------------------------------------------------------------------------//
#define LOOPS 50000
#define LOOP(n, x) __pragma(loop(no_vector)) for (size_t i = 0; i < n; i++) { x }
#define PRINTHEADER                            \
  printf("CLS |  OPERATION   |    TIME    |    VAL          \n"); \
  printf("------------------------------------------------- \n");
//------------------------------------------------------------------------------------------------------------------------//
high_resolution_clock clk;
//------------------------------------------------------------------------------------------------------------------------//
template<typename T>
void bench(const char* name, void task(T* ARR))
{
   T ARR[LOOPS];
   T::randomN(ARR, LOOPS);
   auto t1 = clk.now();
   task(ARR);
   auto t2 = clk.now();
   auto sp = t2 - t1;
   T s = T::ZERO(); for (size_t i = 0; i < LOOPS; i++) s += ARR[i];
   printf(name); printf("%010I64i | ", sp.count()); printf("%05.3f \n", s.x + s.y);
}
//------------------------------------------------------------------------------------------------------------------------//

template<typename T> void benchAddV(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] + ARR[i + 1];)        }
template<typename T> void benchSubV(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] - ARR[i + 1];)        }
template<typename T> void benchMulV(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] * ARR[i + 1];)        }
template<typename T> void benchDivV(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] / ARR[i + 1];)        }
template<typename T> void benchAddS(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] + ARR[i + 1].x;)      }
template<typename T> void benchSubS(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] - ARR[i + 1].x;)      }
template<typename T> void benchMulS(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] * ARR[i + 1].x;)      }
template<typename T> void benchDivS(T* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] / ARR[i + 1].x;)      }
template<typename T> void benchIsZero(T* ARR)    { LOOP(LOOPS, ARR[i].x = (double)ARR[i].isZero();)      }
template<typename T> void benchIsZeroE(T* ARR)   { LOOP(LOOPS, ARR[i].x = (double)ARR[i].isZero(0.01);)  }
template<typename T> void benchRotate(T* ARR)    { LOOP(LOOPS, ARR[i].rotate(M_PI_2);)                   }
template<typename T> void benchLength(T* ARR)    { LOOP(LOOPS, ARR[i].x = ARR[i].length();)              }
template<typename T> void benchLength2(T* ARR)   { LOOP(LOOPS, ARR[i].x = ARR[i].length2();)             }
template<typename T> void benchNormalise(T* ARR) { LOOP(LOOPS, ARR[i].normalise();)                      }
template<typename T> void benchRound(T* ARR)     { LOOP(LOOPS, ARR[i].round();)                          }
template<typename T> void benchFloor(T* ARR)     { LOOP(LOOPS, ARR[i].floor();)                          }
template<typename T> void benchCeil(T* ARR)      { LOOP(LOOPS, ARR[i].ceil();)                           }
template<typename T> void benchAbs(T* ARR)       { LOOP(LOOPS, ARR[i].abs();)                            }
template<typename T> void benchSwap(T* ARR)      { LOOP(LOOPS - 1, ARR[i].swap(ARR[i + 1]);)             }
template<typename T> void benchDot(T* ARR)       { LOOP(LOOPS - 1, ARR[i].x = ARR[i].dot(ARR[i + 1]);)   }
template<typename T> void benchCross(T* ARR)     { LOOP(LOOPS - 1, ARR[i].x = ARR[i].cross(ARR[i + 1]);) }
template<typename T> void benchMax(T* ARR)       { LOOP(LOOPS - 1, ARR[i].max(ARR[i + 1]);)              }
template<typename T> void benchMin(T* ARR)       { LOOP(LOOPS - 1, ARR[i].min(ARR[i + 1]);)              }
template<typename T> void benchBound(T* ARR)     { LOOP(LOOPS - 2, ARR[i].bound(ARR[i + 1], ARR[i + 2]);) }
template<typename T> void benchSide(T* ARR)      { LOOP(LOOPS - 2, ARR[i].x = ARR[i].side(ARR[i + 1], ARR[i + 2]);)                    }
template<typename T> void benchInsideR(T* ARR)   { LOOP(LOOPS - 2, ARR[i].x = (double)ARR[i].inside(ARR[i + 1], ARR[i + 2]);)          }
template<typename T> void benchInsideRE(T* ARR)  { LOOP(LOOPS - 2, ARR[i].x = (double)ARR[i].inside(ARR[i + 1], ARR[i + 2], 0.001);)   }
template<typename T> void benchInsideC(T* ARR)   { LOOP(LOOPS - 2, ARR[i].x = (double)ARR[i].inside(ARR[i + 1], ARR[i + 2].x);)        }
template<typename T> void benchInsideCE(T* ARR)  { LOOP(LOOPS - 2, ARR[i].x = (double)ARR[i].inside(ARR[i + 1], ARR[i + 2].x, 0.001);) }
template<typename T> void benchAreaTri(T* ARR)   { LOOP(LOOPS - 2, ARR[i].x = ARR[i].area(ARR[i + 1], ARR[i + 2]);)                    }
template<typename T> void benchAngle(T* ARR)     { LOOP(LOOPS, ARR[i].x = ARR[i].angle();)                                             }
template<typename T> void benchAngleV(T* ARR)    { LOOP(LOOPS - 1, ARR[i].x = ARR[i].angle(ARR[i + 1]);)                               }
void benchV2d()
{
   PRINTHEADER;
   bench<V2d>("V2d | operator + v | ", benchAddV<V2d>);
   bench<V2d>("V2d | operator - v | ", benchSubV<V2d>);
   bench<V2d>("V2d | operator * v | ", benchMulV<V2d>);
   bench<V2d>("V2d | operator / v | ", benchDivV<V2d>);
   bench<V2d>("V2d | operator + s | ", benchAddS<V2d>);
   bench<V2d>("V2d | operator - s | ", benchSubS<V2d>);
   bench<V2d>("V2d | operator * s | ", benchMulS<V2d>);
   bench<V2d>("V2d | operator / s | ", benchDivS<V2d>);
   bench<V2d>("V2d | isZero()     | ", benchIsZero<V2d>);
   bench<V2d>("V2d | isZero(e)    | ", benchIsZeroE<V2d>);
   bench<V2d>("V2d | rotate()     | ", benchRotate<V2d>);
   bench<V2d>("V2d | length()     | ", benchLength<V2d>);
   bench<V2d>("V2d | length2()    | ", benchLength2<V2d>);
   bench<V2d>("V2d | normalise()  | ", benchNormalise<V2d>);
   bench<V2d>("V2d | round()      | ", benchRound<V2d>);
   bench<V2d>("V2d | floor()      | ", benchFloor<V2d>);
   bench<V2d>("V2d | ceil()       | ", benchCeil<V2d>);
   bench<V2d>("V2d | abs()        | ", benchAbs<V2d>);
   bench<V2d>("V2d | swap()       | ", benchSwap<V2d>);
   bench<V2d>("V2d | dot()        | ", benchDot<V2d>);
   bench<V2d>("V2d | cross()      | ", benchCross<V2d>);
   bench<V2d>("V2d | max()        | ", benchMax<V2d>);
   bench<V2d>("V2d | min()        | ", benchMin<V2d>);
   bench<V2d>("V2d | bound()      | ", benchBound<V2d>);
   bench<V2d>("V2d | side(v,v)    | ", benchSide<V2d>);
   bench<V2d>("V2d | inside(v,v)  | ", benchInsideR<V2d>);
   bench<V2d>("V2d | inside(v,v,e)| ", benchInsideRE<V2d>);
   bench<V2d>("V2d | inside(m,r)  | ", benchInsideC<V2d>);
   bench<V2d>("V2d | inside(m,r,e)| ", benchInsideCE<V2d>);
   bench<V2d>("V2d | area(v,v)    | ", benchAreaTri<V2d>);
   bench<V2d>("V2d | angle()      | ", benchAngle<V2d>);
   bench<V2d>("V2d | angle(v)     | ", benchAngleV<V2d>);
}

int main()
{
   while (true)
   {
      
      benchV2d();
      getchar();
   }

   return 0;
}
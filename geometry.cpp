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
void benchV2d(const char* name, void task(V2d* ARR))
{
   V2d ARR[LOOPS];
   V2d::random(ARR, LOOPS);
   auto t1 = clk.now();
   task(ARR);
   auto t2 = clk.now();
   auto sp = t2 - t1;
   V2d s = V2d::ZERO(); for (size_t i = 0; i < LOOPS; i++) s += ARR[i];
   printf(name); printf("%010I64i | ", sp.count()); printf("%05.3f \n", s.x + s.y);
}

void benchV2dAddV(V2d* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] + ARR[i + 1];)        }
void benchV2dSubV(V2d* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] - ARR[i + 1];)        }
void benchV2dMulV(V2d* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] * ARR[i + 1];)        }
void benchV2dDivV(V2d* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] / ARR[i + 1];)        }
void benchV2dAddS(V2d* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] + ARR[i + 1].x;)      }
void benchV2dSubS(V2d* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] - ARR[i + 1].x;)      }
void benchV2dMulS(V2d* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] * ARR[i + 1].x;)      }
void benchV2dDivS(V2d* ARR)      { LOOP(LOOPS - 1, ARR[i] = ARR[i] / ARR[i + 1].x;)      }
void benchV2dIsZero(V2d* ARR)    { LOOP(LOOPS, ARR[i].x = (double)ARR[i].isZero();)      }
void benchV2dIsZeroE(V2d* ARR)   { LOOP(LOOPS, ARR[i].x = (double)ARR[i].isZero(0.01);)  }
void benchV2dRotate(V2d* ARR)    { LOOP(LOOPS, ARR[i].rotate(M_PI_2);)                   }
void benchV2dLength(V2d* ARR)    { LOOP(LOOPS, ARR[i].x = ARR[i].length();)              }
void benchV2dLength2(V2d* ARR)   { LOOP(LOOPS, ARR[i].x = ARR[i].length2();)             }
void benchV2dNormalise(V2d* ARR) { LOOP(LOOPS, ARR[i].normalise();)                      }
void benchV2dRound(V2d* ARR)     { LOOP(LOOPS, ARR[i].round();)                          }
void benchV2dFloor(V2d* ARR)     { LOOP(LOOPS, ARR[i].floor();)                          }
void benchV2dCeil(V2d* ARR)      { LOOP(LOOPS, ARR[i].ceil();)                           }
void benchV2dSwap(V2d* ARR)      { LOOP(LOOPS - 1, ARR[i].swap(ARR[i + 1]);)             }
void benchV2dDot(V2d* ARR)       { LOOP(LOOPS - 1, ARR[i].x = ARR[i].dot(ARR[i + 1]);)   }
void benchV2dCross(V2d* ARR)     { LOOP(LOOPS - 1, ARR[i].x = ARR[i].cross(ARR[i + 1]);) }

void bench()
{
   PRINTHEADER // V2d
   benchV2d("V2d | operator + v | ", &benchV2dAddV);
   benchV2d("V2d | operator - v | ", &benchV2dSubV);
   benchV2d("V2d | operator * v | ", &benchV2dMulV);
   benchV2d("V2d | operator / v | ", &benchV2dDivV);
   benchV2d("V2d | operator + s | ", &benchV2dAddS);
   benchV2d("V2d | operator - s | ", &benchV2dSubS);
   benchV2d("V2d | operator * s | ", &benchV2dMulS);
   benchV2d("V2d | operator / s | ", &benchV2dDivS);
   benchV2d("V2d | isZero()     | ", &benchV2dIsZero);
   benchV2d("V2d | isZero(e)    | ", &benchV2dIsZeroE);
   benchV2d("V2d | rotate()     | ", &benchV2dRotate);
   benchV2d("V2d | length()     | ", &benchV2dLength);
   benchV2d("V2d | length2()    | ", &benchV2dLength2);
   benchV2d("V2d | normalise()  | ", &benchV2dNormalise);
   benchV2d("V2d | round()      | ", &benchV2dRound);
   benchV2d("V2d | floor()      | ", &benchV2dFloor);
   benchV2d("V2d | ceil()       | ", &benchV2dCeil);
   benchV2d("V2d | swap()       | ", &benchV2dSwap);
   benchV2d("V2d | dot()        | ", &benchV2dDot);
   benchV2d("V2d | cross()      | ", &benchV2dCross);
}

int main()
{
   while (true)
   {
      
      bench();
      getchar();
   }

   return 0;
}
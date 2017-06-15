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

void benchV2dAdd(V2d* ARR)       { LOOP(LOOPS - 1, ARR[i] = ARR[i] + ARR[i + 1];)        }
void benchV2dSub(V2d* ARR)       { LOOP(LOOPS - 1, ARR[i] = ARR[i] - ARR[i + 1];)        }
void benchV2dMul(V2d* ARR)       { LOOP(LOOPS - 1, ARR[i] = ARR[i] * ARR[i + 1];)        }
void benchV2dDiv(V2d* ARR)       { LOOP(LOOPS - 1, ARR[i] = ARR[i] / ARR[i + 1];)        }
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
   benchV2d("V2d | operator + v | ", &benchV2dAdd);
   benchV2d("V2d | operator - v | ", &benchV2dSub);
   benchV2d("V2d | operator * v | ", &benchV2dMul);
   benchV2d("V2d | operator / v | ", &benchV2dDiv);
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
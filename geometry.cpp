#include "geometry.h"
#include <Windows.h>
#include <random>
#include <chrono>

using namespace geometry;
using namespace std::chrono;
using namespace std;

#define LOOPS 50000

high_resolution_clock clk;

void benchV2dRotate()
{
   V2d ARR[LOOPS];
   for (size_t i = 0; i < LOOPS; i++)
      ARR[i] = V2d((double)std::rand(), (double)std::rand());

   auto t1 = clk.now();
   for (size_t i = 0; i < LOOPS; i++)
      ARR[i].rotate(M_PI_2);
   auto t2 = clk.now();
   auto sp = t2 - t1;

   V2d s;
   for (size_t i = 0; i < LOOPS; i++)
      s += ARR[i];
   printf("Dumm: %f \n", s.x + s.y);
   printf("Time: %I64i \n", sp.count());
}

int main()
{
   double vals[2] = { 0.1, 2.1 };

   V2d a(1.0, 0.0);
   a *= vals;

   V2d a2(1.0, 0.0);
   V2d b(2.0f, 2.0f);
   V2d c(5.0f, 0.12f);
   float f1 = 2.53f;

   a.rotate(-M_PI*0.5);

   V2d s1 = ((a + b) * f1) / c;

   float side1 = c.side(a, b);

   a.swap(b);
   V2d e = a.perp2();

   s1.round();
  int kg = 1;

   while (true)
   {
      benchV2dRotate();
      getchar();
   }

   return 0;
}
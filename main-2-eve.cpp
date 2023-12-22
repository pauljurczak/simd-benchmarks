#include "common.hpp"
#include "vectorclass.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xsort.hpp>
#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <eve/wide.hpp>

constexpr size_t H = 816, W = 256;


int nested_loop(const array<uint8_t, H*W> &img) {
  int sum = 0;

  for (int y=0; y<H; y++) {
    int vMax = 0;
    int iMax = 0;

    for (int i=y*W; i<(y+1)*W; i++)
      if (img[i] > vMax) {
        vMax = img[i];
        iMax = i;
      }

    sum += iMax-y*W;
  }

  return sum;
}


int loop_std_max_element(const array<uint8_t, H*W> &img) {
  int sum = 0;

  for (auto i=img.begin(); i<img.end(); i+=W) 
    sum += max_element(i, i+W) - i;

  return sum;
}


int loop_eve_max_element(const array<uint8_t, H*W> &img) {
  int sum = 0;

  for (int i=0; i<H*W; i+=W) {
    std::span s{&img[i], W} ;
    sum += eve::algo::max_element(s)-s.begin();
  }

  return sum;
}


int main() {
  printVersion();

  vector<float> timings;
  int           nTests = 1000, result = 0;
  array<uint8_t, H*W> img;
  xt::xtensor_fixed<uint8_t, xt::xshape<H, W>> imgx;
  
  for (int i=0; i<img.size(); i++)
    img[i] = i%255;

  for (int y=0; y<H; y++) 
    for (int x=0; x<W; x++)
      imgx(y, x) = (x+y*W) % 255;

  for (int i = 0; i < nTests; i++) {
    auto t0 = high_resolution_clock::now();
    result += nested_loop(img);
    timings.push_back(duration<float, milli>{high_resolution_clock::now()-t0}.count());
  }

  sort(timings);
  print("nested_loop(): {:.3f}  {:.3f} [ms]  {}\n", timings[0], timings[timings.size()/2], result/nTests);
  timings.clear();
  result = 0;

  for (int i = 0; i < nTests; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_std_max_element(img);
    timings.push_back(duration<float, milli>{high_resolution_clock::now()-t0}.count());
  }

  sort(timings);
  print("loop_std_max_element(): {:.3f}  {:.3f} [ms]  {}\n", timings[0], timings[timings.size()/2], result/nTests);
  timings.clear();
  result = 0;

  for (int i = 0; i < nTests; i++) {
    auto t0 = high_resolution_clock::now();
    result += xt::sum(xt::argmax(imgx, 1))();
    timings.push_back(duration<float, milli>{high_resolution_clock::now()-t0}.count());
  }

  sort(timings);
  print("xt::sum(xt::argmax(imgx, 1)): {:.3f}  {:.3f} [ms]  {}\n", timings[0], timings[timings.size()/2], result/nTests);
  timings.clear();
  result = 0;

  for (int i = 0; i < nTests; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_eve_max_element(img);
    timings.push_back(duration<float, milli>{high_resolution_clock::now()-t0}.count());
  }

  sort(timings);
  print("loop_eve_max_element(): {:.3f}  {:.3f} [ms]  {}\n", timings[0], timings[timings.size()/2], result/nTests);
}

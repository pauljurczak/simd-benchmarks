#include "common.hpp"
#include "vectorclass.h"
#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <eve/wide.hpp>

using V32u1 = eve::wide<uint8_t, eve::fixed<32>>;
constexpr size_t H = 816, W = 256;


int loop_eve_max_element(const array<uint8_t, H*W> &img) {
  int sum = 0;

  for (auto i=img.begin(); i<img.end(); i+=W) 
    sum += eve::algo::min_element(eve::algo::as_range(i, i + W), eve::is_greater) - i;

  return sum;
}


int loop_eve_nested(const array<uint8_t, H*W> &img, const array<V32u1, 8> &idx) {
  int sum = 0;
  V32u1 vMax, iMax, vCurr, iCurr;

  for (int i=0; i<H*W; i+=W) {
    iMax = idx[0];
    vMax = V32u1(&img[i]);

    for (int j=1; j<8; j++) {
      iCurr = idx[j];
      vCurr = V32u1(&img[i+j*32]);
      iMax = eve::if_else(vCurr > vMax, iCurr, iMax);
      vMax = eve::max(vMax, vCurr);
    }

    V32u1 vMaxAll{eve::maximum(vMax)};
    sum += iMax.get(*eve::first_true(vMax == vMaxAll));
  }

  return sum;
}


int loop_vc_nested(const array<uint8_t, H*W> &img, const array<Vec32uc, 8> &idx) {
  int sum = 0;
  Vec32uc vMax, iMax, vCurr, iCurr;

  for (int i=0; i<H*W; i+=W) {
    iMax.load(&idx[0]);
    vMax.load(&img[i]);

    for (int j=1; j<8; j++) {
      iCurr.load(&idx[j]);
      vCurr.load(&img[i+j*32]);
      iMax = select(vCurr > vMax, iCurr, iMax);
      vMax = max(vMax, vCurr);
    }

    Vec32uc vMaxAll{horizontal_max(vMax)};
    sum += iMax[horizontal_find_first(vMax == vMaxAll)];
  }

  return sum;
}


int loop_vc_unrolled(const array<uint8_t, H*W> &img, const array<Vec32uc, 8> &idx) {
  int sum = 0;
  Vec32uc vMax0, iMax0, vMax1, iMax1, vMax2, iMax2, vMax3, iMax3, vMax4, iMax4, vMax5, iMax5, vMax6, iMax6, vMax7, iMax7;

  for (int i=0; i<H*W; i+=W) {
    iMax0.load(&idx[0]);
    vMax0.load(&img[i]);
    iMax1.load(&idx[1]);
    vMax1.load(&img[i+32]);
    iMax1 = select(vMax1 > vMax0, iMax1, iMax0);
    vMax1 = max(vMax1, vMax0);

    iMax2.load(&idx[2]);
    vMax2.load(&img[i+64]);
    iMax3.load(&idx[3]);
    vMax3.load(&img[i+96]);
    iMax3 = select(vMax3 > vMax2, iMax3, iMax2);
    vMax3 = max(vMax3, vMax2);

    iMax3 = select(vMax3 > vMax1, iMax3, iMax1);
    vMax3 = max(vMax3, vMax1);

    iMax4.load(&idx[4]);
    vMax4.load(&img[i+128]);
    iMax5.load(&idx[5]);
    vMax5.load(&img[i+160]);
    iMax5 = select(vMax5 > vMax4, iMax5, iMax4);
    vMax5 = max(vMax5, vMax4);

    iMax6.load(&idx[6]);
    vMax6.load(&img[i+192]);
    iMax7.load(&idx[7]);
    vMax7.load(&img[i+224]);
    iMax7 = select(vMax7 > vMax6, iMax7, iMax6);
    vMax7 = max(vMax7, vMax6);

    iMax7 = select(vMax7 > vMax5, iMax7, iMax5);
    vMax7 = max(vMax7, vMax5);
    iMax7 = select(vMax7 > vMax3, iMax7, iMax3);
    vMax7 = max(vMax7, vMax3);

    Vec32uc vMaxAll{horizontal_max(vMax7)};
    sum += iMax7[horizontal_find_first(vMax7 == vMaxAll)];
  }

  return sum;
}


int main() {
  printVersion();

  vector<float> timings;
  int           nTests = 1'000'000;
  array<uint8_t, H*W> img;
  array<uint8_t, W> i0_255;
  array<V32u1, 8> idxEve;
  array<Vec32uc, 8> idxVCL;
  
  for (int i=0; i<img.size(); i++)
    img[i] = i%255;

  iota(i0_255.begin(), i0_255.end(), 0);

  for (int i=0; i<8; i++) {
    idxEve[i] = V32u1(&i0_255[i*32]);
    idxVCL[i].load(&i0_255[i*32]);
  }

  uint64_t result = 0;

  for (int i = 0; i < nTests; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_eve_max_element(img);
    timings.push_back(duration<float, micro>{high_resolution_clock::now()-t0}.count());
  }

  sort(timings);
  print("loop_eve_max_element(): {:.3f}  {:.3f} [us]  {}\n", timings[0], timings[timings.size()/2], result/nTests);
  timings.clear();
  result = 0;

  for (int i = 0; i < nTests; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_eve_nested(img, idxEve);
    timings.push_back(duration<float, micro>{high_resolution_clock::now()-t0}.count());
  }

  sort(timings);
  print("loop_eve_nested(): {:.3f}  {:.3f} [us]  {}\n", timings[0], timings[timings.size()/2], result/nTests);
  timings.clear();
  result = 0;

  for (int i = 0; i < nTests; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_vc_nested(img, idxVCL);
    timings.push_back(duration<float, micro>{high_resolution_clock::now()-t0}.count());
  }

  sort(timings);
  print("loop_vc_nested(): {:.3f}  {:.3f} [us]  {}\n", timings[0], timings[timings.size()/2], result/nTests);
  result = 0;

  for (int i = 0; i < nTests; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_vc_unrolled(img, idxVCL);
    timings.push_back(duration<float, micro>{high_resolution_clock::now()-t0}.count());
  }

  sort(timings);
  print("loop_vc_unrolled(): {:.3f}  {:.3f} [us]  {}\n", timings[0], timings[timings.size()/2], result/nTests);
}

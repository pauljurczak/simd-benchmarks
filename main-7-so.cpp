#include "common.hpp"
#include "vectorclass.h"

constexpr size_t H = 816, W = 256;


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


int loop_vc_nested_tzcnt_u32(const array<uint8_t, H*W> &img, const array<Vec32uc, 8> &idx) {
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
    size_t maxidx = _tzcnt_u32(to_bits(vMax == vMaxAll));
    sum += iMax[maxidx];
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


int loop_vc_nested_noselect_2chains(const std::array<uint8_t, H*W> &img, const std::array<Vec32uc, 8> &idx) {
  int sum = 0;

  for (int i=0; i<H*W; i+=W) {
    __m256i tmpidx = _mm256_loadu_si256((__m256i*)&idx[0]);
    __m256i tmp = _mm256_loadu_si256((__m256i*)&img[i]);
    Vec16us vMaxlo = _mm256_unpacklo_epi8(tmpidx, tmp);
    Vec16us vMaxhi = _mm256_unpackhi_epi8(tmpidx, tmp);

    for (int j=1; j<8; j++) {
      Vec32uc vCurr, iCurr;
      iCurr.load(&idx[j]);  // these get hoisted out of the outer loop and reused across img iters
      vCurr.load(&img[i+j*32]);
      Vec16us lo = _mm256_unpacklo_epi8(iCurr, vCurr);
      Vec16us hi = _mm256_unpackhi_epi8(iCurr, vCurr);
      vMaxlo = max(vMaxlo, lo);
      vMaxhi = max(vMaxhi, hi);
          // vMax = max(vMax, max(lo,hi));  // GCC was optimizing to two dep chains anyway, and that's better on big-cores that can do more than 1 load+shuffle+max per clock
    }
    Vec16us vMax = max(vMaxlo, vMaxhi);

    // silly GCC uses vpextrw even though we're already truncating narrower
    auto maxidx = (uint8_t)horizontal_max(vMax); // retrieve the payload from the bottom of the max
    // TODO: use phminposuw like the last part of maxpos_u8_noscan_unpack
    // with indices loaded and inverted once, outside the outer loop.  (Manually unrolled if compilers don't do that for you)
    sum += maxidx;
  }

  return sum;
}


int main(int argc, char* argv[]) {
  printVersion();

  cxxopts::Options options("MyProgram", "One line description of MyProgram");
  options.add_options()
    ("n, nIter", "Number of iterations", cxxopts::value<int>()->default_value("1000000"))
    ("t, nSamples", "Number of time samples", cxxopts::value<int>()->default_value("5"));
  int nIter = options.parse(argc, argv)["n"].as<int>();
  int nSamples = options.parse(argc, argv)["t"].as<int>();
  array<uint8_t, H*W> img;
  array<uint8_t, W> i0_255;
  array<Vec32uc, 8> idxVCL;
  
  nIter -= (nIter-1)%(nSamples-1);
  vector<float> ts(nIter, 0);
  int tStride = (nIter-1)/(nSamples-1);
  uint64_t result = 0;

  for (int i=0; i<img.size(); i++)
    img[i] = i%255;

  iota(i0_255.begin(), i0_255.end(), 0);

  for (int i=0; i<8; i++) {
    idxVCL[i].load(&i0_255[i*32]);
  }

  for (int i = 0; i < nIter; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_vc_nested(img, idxVCL);
    ts[i] = duration<float, micro>{high_resolution_clock::now()-t0}.count();
  }

  sort(ts);
  print("loop_vc_nested(): {::.2f} [us]  {}\n", ts | stride(tStride), result/nIter);
  result = 0;

  for (int i = 0; i < nIter; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_vc_unrolled(img, idxVCL);
    ts[i] = duration<float, micro>{high_resolution_clock::now()-t0}.count();
  }

  sort(ts);
  print("loop_vc_unrolled(): {::.2f} [us]  {}\n", ts | stride(tStride), result/nIter);
  result = 0;

  for (int i = 0; i < nIter; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_vc_nested_noselect_2chains(img, idxVCL);
    ts[i] = duration<float, micro>{high_resolution_clock::now()-t0}.count();
  }

  sort(ts);
  print("loop_vc_nested_noselect_2chains(): {::.2f} [us]  {}\n", ts | stride(tStride), result/nIter);
  result = 0;

  for (int i = 0; i < nIter; i++) {
    auto t0 = high_resolution_clock::now();
    result += loop_vc_nested_tzcnt_u32(img, idxVCL);
    ts[i] = duration<float, micro>{high_resolution_clock::now()-t0}.count();
  }

  sort(ts);
  print("loop_vc_nested_tzcnt_u32(): {::.2f} [us]  {}\n", ts | stride(tStride), result/nIter);
}

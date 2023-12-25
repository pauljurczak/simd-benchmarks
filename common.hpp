#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <map>
#include <experimental/random>
#include <concepts>
#include <span>
#include <thread>
#include <valarray>
#include <array>
#include <filesystem>
#include <ranges>
#include <limits>
#include <numbers>
#include <float.h>
#include <fmt/format.h>
#include <fmt/color.h>
#include <fmt/ostream.h>
// #include <fmt/ranges.h>
#include "hwinfo/hwinfo.h"

using namespace std;
using namespace std::chrono;
using fmt::print;
using fmt::format;
using fmt::color;
using std::experimental::randint;
namespace fs = std::filesystem;


template<floating_point T>
int iround(T x) {return static_cast<int>(round(x));}


template<typename T1, typename T2>
inline bool contains(const T1 &container, const T2 &val) 
{
  return find(container.begin(), container.end(), val) != container.end();
}


template<typename T1, typename T2>
inline void remove(T1& container, const T2& val) 
{
  container.erase(remove(container.begin(), container.end(), val), container.end());
}


template<typename T>
inline void sort(T& container) 
{
  sort(begin(container), end(container));
}


template<typename Cont, typename Acc>   // Stroustrup's example
Acc accumulate(const Cont& c, Acc init = typename Cont::value_type())
{
  return std::accumulate(begin(c), end(c), init);
}


template<typename Cont>
inline auto min_element(const Cont& c) {
  return *min_element(begin(c), end(c));
}


template<typename Cont>
auto span(const Cont& c) {
  auto s = minmax_element(begin(c), end(c));

  return *s.second - *s.first;
}


template<typename Cont>
auto maxIndex(const Cont& c) {
  return max_element(begin(c), end(c)) - begin(c);
}


inline void printVersion() {
  auto cpu = hwinfo::getAllCPUs()[0];
  hwinfo::OS os;

  print("{} on {}\n", os.name(), cpu.modelName());

#ifdef __clang__
  print("clang v{} C++{}\n", __clang_version__, map<int, string>{{201703, "17"}, {201709, "20"}, {202002, "20"}}[__cplusplus]);
#else  
  // print("gcc    v{}.{} C++{}\n", __GNUC__, __GNUC_MINOR__, map<int, string>{{201703, "17"}, {201709, "20"}, {202002, "20"}}[__cplusplus]);
  print("gcc    v{}.{}   __cplusplus={}\n", __GNUC__, __GNUC_MINOR__, __cplusplus);
#endif 

#ifdef EIGEN_WORLD_VERSION
  print("Eigen v{}.{}.{}\n", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif

#ifdef CV_VERSION
  print("OpenCV v{},  {} threads,  {} supported\n", CV_VERSION, getNumThreads(), checkHardwareSupport(CPU_AVX2) ? "AVX2" : "_");
#endif

#ifdef _OPENMP
  print("OpenMP v{}\n", map<int, string>{{201307, "4.0"}, {201511, "4.5"}, {201811, "5.0"}, {202011, "5.1"}}[_OPENMP]);
#endif

#ifdef BLAZE_MAJOR_VERSION
  print("Blaze v{}.{}.{}\n", BLAZE_MAJOR_VERSION, BLAZE_MINOR_VERSION, BLAZE_PATCH_VERSION);
#endif

#ifdef FASTOR_MAJOR
  print("Fastor v{}.{}.{}\n", FASTOR_MAJOR, FASTOR_MINOR, FASTOR_PATCHLEVEL);
#endif

#ifdef FMT_VERSION
  print("{{fmt}}  v{}.{}.{}\n", FMT_VERSION/10000, FMT_VERSION%10000/100, FMT_VERSION%100);
#endif
}


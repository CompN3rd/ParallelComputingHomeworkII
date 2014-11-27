#ifndef om_cuda_tools_
#define om_cuda_tools_

#include <stdio.h>
#include <stdlib.h>

#ifndef __USE_POSIX199309
#define __USE_POSIX199309
#include <time.h>
#undef __USE_POSIX199309
#else
#include <time.h>
#endif

typedef struct timespec Timer;

static inline void initTimer(Timer* timer) {
  clock_gettime(CLOCK_MONOTONIC,timer);
}

static inline double getTimer(Timer* timer) {
  struct timespec endTimespec;
  clock_gettime(CLOCK_MONOTONIC,&endTimespec);
  return (endTimespec.tv_sec-timer->tv_sec)+
    (endTimespec.tv_nsec-timer->tv_nsec)*1e-9;
}

static inline double getAndResetTimer(Timer* timer) {
  struct timespec endTimespec;
  clock_gettime(CLOCK_MONOTONIC,&endTimespec);
  double result=(endTimespec.tv_sec-timer->tv_sec)+
    (endTimespec.tv_nsec-timer->tv_nsec)*1e-9;
  *timer=endTimespec;
  return result;
}

static inline double getTimerDifference(Timer* timerStart,Timer* timerEnd) {
  return (timerEnd->tv_sec-timerStart->tv_sec)+
    (timerEnd->tv_nsec-timerStart->tv_nsec)*1e-9;
}

#ifndef NDEBUG
#define cu_verify(x) do {                                                \
    cudaError_t result = x;                                              \
    if (result!=cudaSuccess) {                                           \
      fprintf(stderr,"%s:%i: error: cuda function call failed:\n"        \
              "  %s;\nmessage: %s\n",                                    \
              __FILE__,__LINE__,#x,cudaGetErrorString(result));          \
      exit(1);                                                           \
    }                                                                    \
  } while(0)
#define cu_verify_void(x) do {                                           \
    x;                                                                   \
    cudaError_t result = cudaGetLastError();                             \
    if (result!=cudaSuccess) {                                           \
      fprintf(stderr,"%s:%i: error: cuda function call failed:\n"        \
              "  %s;\nmessage: %s\n",                                    \
              __FILE__,__LINE__,#x,cudaGetErrorString(result));          \
      exit(1);                                                           \
    }                                                                    \
  } while(0)
#else
#define cu_verify(x) do {                                                \
    x;                                                                   \
    } while(0)
#define cu_verify_void(x) do {                                           \
    x;                                                                   \
    } while(0)
#endif

#endif

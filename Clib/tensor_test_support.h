/*
 * tensor_test_support.h - a clock and a memory reading for simple_tensor's
 * tests. Tests only; nothing in src/ includes it.
 *
 * Copyright (c) 2026 Larry Rix - MIT License
 */

#ifndef TENSOR_TEST_SUPPORT_H
#define TENSOR_TEST_SUPPORT_H

#if defined(_WIN32) || defined(EIF_WINDOWS)

#include <windows.h>
#include <psapi.h>

static long long tts_clock_ms(void)
{
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (c.QuadPart * 1000) / f.QuadPart;
}

/* Private (commit) memory of this process, MB. */
static long long tts_private_mb(void)
{
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return (long long) (pmc.PagefileUsage / (1024 * 1024));
}

#else

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

static long long tts_clock_ms(void)
{
    struct timespec l_now;
    clock_gettime(CLOCK_MONOTONIC, &l_now);
    return (long long) l_now.tv_sec * 1000 + l_now.tv_nsec / 1000000;
}

/* Memory charged to this process, MB: the physical footprint on macOS, the
   resident set on Linux. The tests look only at how it grows. */
static long long tts_private_mb(void)
{
#ifdef __APPLE__
    task_vm_info_data_t l_info;
    mach_msg_type_number_t l_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &l_info, &l_count) != KERN_SUCCESS) return 0;
    return (long long) (l_info.phys_footprint / (1024 * 1024));
#else
    long l_size = 0, l_resident = 0;
    FILE* l_file = fopen("/proc/self/statm", "r");
    if (!l_file) return 0;
    if (fscanf(l_file, "%ld %ld", &l_size, &l_resident) != 2) l_resident = 0;
    fclose(l_file);
    return (long long) l_resident * sysconf(_SC_PAGESIZE) / (1024 * 1024);
#endif
}

#endif

#endif /* TENSOR_TEST_SUPPORT_H */

#include <benchmark/benchmark.h>

#include <JuceHeader.h>

#include "dsp/diode_clipper/DiodeClipperWDF.h"

constexpr double sampleRate = 96000.0;
constexpr int blockSize = 2048;
constexpr int nSamples = int (sampleRate * 0.1);

DiodeClipperWDF wdf;
AudioBuffer<float> buffer (1, blockSize);
static void DiodeBench (benchmark::State& state)
{
    wdf.prepare (sampleRate);
    buffer.clear();
    buffer.setSample (0, 0, 1.0f);

    wdf.setParameters (1000.0f, (int) state.range (0));

    for (auto _ : state)
    {
        int count = 0;
        while (count < nSamples)
        {
            wdf.process (buffer);
            count += blockSize;
        }
    }
}
BENCHMARK (DiodeBench)->MinTime (5)->Unit (benchmark::kMillisecond)->Arg (0)->Arg (1)->Arg (2);

BENCHMARK_MAIN();

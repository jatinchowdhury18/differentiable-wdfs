#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <wdf_t.h>

namespace wdft = chowdsp::WDFT;

int main()
{
    std::cout << "Generating input signal..." << std::endl;

    constexpr double fs = 48000.0;
    constexpr int N = 5000;
    constexpr double freq = 100.0;
    auto x = std::vector<double>(N, 0.0);
    for(int n = 0; n < N; ++n)
        x[n] = 5.0 * std::sin(2.0 * M_PI * freq * (double) n / fs);

    std::cout << "Creating WDF..." << std::endl;
    wdft::ResistiveVoltageSourceT<double> Vs;
    wdft::ResistorT<double> R { 10.0e3 };
    wdft::WDFParallelT<double, decltype(Vs), decltype(R)> P1 { Vs, R };
    wdft::DiodePairT<double, decltype(P1)> dp { P1, 1.0e-9 };

    std::cout << "Processing data..." << std::endl;
    auto y = std::vector<double>(N, 0.0);
    for(int n = 0; n < N; ++n)
    {
        Vs.setVoltage (x[n]);
        dp.incident(P1.reflected());
        P1.incident(dp.reflected());

        y[n] = wdft::voltage<double>(R);
    }

    std::cout << "Saving x data to file..." << std::endl;
    std::ofstream x_file { "test_data/clipper_x.csv" };
    std::copy(x.begin(), x.end(), std::ostream_iterator<double>(x_file, "\n"));

    std::cout << "Saving y data to file..." << std::endl;
    std::ofstream y_file { "test_data/clipper_y.csv" };
    std::copy(y.begin(), y.end(), std::ostream_iterator<double>(y_file, "\n"));

    return 0;
}

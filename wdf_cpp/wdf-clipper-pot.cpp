#include <algorithm>
#include <array>
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
    constexpr int N = 2500;
    constexpr double freq = 100.0;
    constexpr std::array<double, 5> R_vals { 1000.0, 2000.0, 5000.0, 10000.0, 50000.0 };
    auto x = std::vector<double>(N * R_vals.size(), 0.0);
    auto r_data = std::vector<double>(N * R_vals.size(), 0.0);
    for(size_t n = 0; n < x.size(); ++n)
        x[n] = 5.0 * std::sin(2.0 * M_PI * freq * (double) n / fs);

    std::cout << "Creating WDF..." << std::endl;
    wdft::ResistiveVoltageSourceT<double> resVs { 4700.0 };
    wdft::CapacitorT<double> C { 47.0e-9f, fs };
    wdft::WDFParallelT<double, decltype(resVs), decltype(C)> P1 { resVs, C };
    wdft::DiodePairT<double, decltype(P1)> dp { P1, 1.0e-9 };

    std::cout << "Processing data..." << std::endl;
    auto y = std::vector<double>(x.size(), 0.0);
    for (size_t r_idx = 0; r_idx < R_vals.size(); ++r_idx)
    {
        resVs.setResistanceValue (R_vals[r_idx]);
        for(int n = 0; n < N; ++n)
        {
            auto idx = n + N * r_idx;
            r_data[idx] = resVs.R;

            resVs.setVoltage (x[idx]);
            dp.incident(P1.reflected());
            P1.incident(dp.reflected());

            y[idx] = wdft::voltage<double>(dp);
        }
    }

    std::cout << "Saving x data to file..." << std::endl;
    std::ofstream x_file { "test_data/clipper_pot_x.csv" };
    std::copy(x.begin(), x.end(), std::ostream_iterator<double>(x_file, "\n"));

    std::cout << "Saving R data to file..." << std::endl;
    std::ofstream r_file { "test_data/clipper_pot_r.csv" };
    std::copy(r_data.begin(), r_data.end(), std::ostream_iterator<double>(r_file, "\n"));

    std::cout << "Saving y data to file..." << std::endl;
    std::ofstream y_file { "test_data/clipper_pot_y.csv" };
    std::copy(y.begin(), y.end(), std::ostream_iterator<double>(y_file, "\n"));

    return 0;
}

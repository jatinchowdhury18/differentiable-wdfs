#pragma once

#include "../CircuitModel.h"
#include "../diode_clipper/DiodePairNeuralModel.h"

class TubeScreamer : public CircuitModel
{
public:
    TubeScreamer (const String& prefix, AudioProcessorValueTreeState& vts);

    static void addParameters (chowdsp::Parameters& params, const String& prefix);

    void prepare (double sampleRate, int samplesPerBlock) override;

    void process (AudioBuffer<float>& buffer) override;

private:
    std::atomic<float>* gainDBParam = nullptr;
    std::atomic<float>* driveParam = nullptr;
    std::atomic<float>* modelChoiceParam = nullptr;

    dsp::Gain<float> inputGain;

    // Port B
    wdft::ResistiveVoltageSourceT<float> Vin;
    wdft::CapacitorT<float> C2 { 1.0e-6f };
    wdft::WDFSeriesT<float, decltype (Vin), decltype (C2)> S1 { Vin, C2 };

    wdft::ResistorT<float> R5 { 10.0e3f };
    wdft::WDFParallelT<float, decltype (S1), decltype (R5)> P1 { S1, R5 };

    // Port C
    wdft::ResistorT<float> R4 { 4.7e3f };
    wdft::CapacitorT<float> C3 { 0.047e-6f };
    wdft::WDFSeriesT<float, decltype (R4), decltype (C3)> S2 { R4, C3 };

    // Port D
    wdft::ResistorT<float> RL { 1.0e6f };

    struct ImpedanceCalc
    {
        template <typename RType>
        static float calcImpedance (RType& R)
        {
            constexpr float Ag = 100.0f; // op-amp gain
            constexpr float Ri = 1.0e9f; // op-amp input impedance
            constexpr float Ro = 1.0e-1f; // op-amp output impedance

            const auto [Rb, Rc, Rd] = R.getPortImpedances();

            // This scattering matrix was derived using the R-Solver python script (https://github.com/jatinchowdhury18/R-Solver),
            // invoked with command: r_solver.py --adapt 0 --out scratch/tube_screamer_scatt.txt scratch/tube_screamer.txt
            R.setSMatrixData ({ { 0, (Ag * Rd * Ri - Rc * Rd + Rc * Ro) / ((Rb + Rc) * Rd + Rd * Ri - (Rb + Rc + Ri) * Ro), -((Ag + 1) * Rd * Ri + Rb * Rd - (Rb + Ri) * Ro) / ((Rb + Rc) * Rd + Rd * Ri - (Rb + Rc + Ri) * Ro), -Ro / (Rd - Ro) },
                                { -(Rb * Rc * Rd - Rb * Rc * Ro) / ((Ag + 1) * Rc * Rd * Ri + Rb * Rc * Rd - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro), ((Ag + 1) * Rc * Rc * Rd * Ri + (Ag + 1) * Rc * Rd * Ri * Ri - Rb * Rb * Rc * Rd + (Rb * Rb * Rc - (Rc + Rd) * Ri * Ri + (Rb * Rb - Rc * Rc) * Rd - (Rc * Rc + 2 * Rc * Rd) * Ri) * Ro) / ((Ag + 1) * Rc * Rd * Ri * Ri + ((Ag + 2) * Rb * Rc + (Ag + 1) * Rc * Rc) * Rd * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd - (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro), ((Ag + 1) * Rb * Rc * Rd * Ri + Rb * Rb * Rc * Rd - (Rb * Rb * Rc + 2 * (Rb * Rb + Rb * Rc) * Rd + (Rb * Rc + 2 * Rb * Rd) * Ri) * Ro) / ((Ag + 1) * Rc * Rd * Ri * Ri + ((Ag + 2) * Rb * Rc + (Ag + 1) * Rc * Rc) * Rd * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd - (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro), -Rb * Rc * Ro / ((Ag + 1) * Rc * Rd * Ri + Rb * Rc * Rd - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro) },
                                { -(Rb * Rc * Rd + Rc * Rd * Ri - (Rb * Rc + Rc * Ri) * Ro) / ((Ag + 1) * Rc * Rd * Ri + Rb * Rc * Rd - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro), (Ag * Rc * Rd * Ri * Ri + Rb * Rc * Rc * Rd + (Ag * Rb * Rc + (2 * Ag + 1) * Rc * Rc) * Rd * Ri - (Rb * Rc * Rc + 2 * (Rb * Rc + Rc * Rc) * Rd + (Rc * Rc + 2 * Rc * Rd) * Ri) * Ro) / ((Ag + 1) * Rc * Rd * Ri * Ri + ((Ag + 2) * Rb * Rc + (Ag + 1) * Rc * Rc) * Rd * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd - (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro), -((Ag + 1) * Rc * Rc * Rd * Ri + Rb * Rc * Rc * Rd - (Rb * Rc * Rc - Rd * Ri * Ri - (Rb * Rb - Rc * Rc) * Rd + (Rc * Rc - 2 * Rb * Rd) * Ri) * Ro) / ((Ag + 1) * Rc * Rd * Ri * Ri + ((Ag + 2) * Rb * Rc + (Ag + 1) * Rc * Rc) * Rd * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd - (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro), -(Rb * Rc + Rc * Ri) * Ro / ((Ag + 1) * Rc * Rd * Ri + Rb * Rc * Rd - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro) },
                                { (Ag * Rc * Rd * Ri - ((Rb + Rc) * Rd + Rd * Ri) * Ro) / ((Ag + 1) * Rc * Rd * Ri + Rb * Rc * Rd - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro), ((Ag * Ag + 2 * Ag) * Rc * Rd * Rd * Ri * Ri + (2 * Ag * Rb * Rc + Ag * Rc * Rc) * Rd * Rd * Ri + (Rc * Rd * Ri + (Rb * Rc + Rc * Rc) * Rd) * Ro * Ro - ((Rb * Rc + Rc * Rc) * Rd * Rd + (2 * Ag * Rc * Rd + Ag * Rd * Rd) * Ri * Ri + ((Ag * Rb + (Ag + 1) * Rc) * Rd * Rd + (2 * Ag * Rb * Rc + Ag * Rc * Rc) * Rd) * Ri) * Ro) / ((Ag + 1) * Rc * Rd * Rd * Ri * Ri + ((Ag + 2) * Rb * Rc + (Ag + 1) * Rc * Rc) * Rd * Rd * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + ((Ag + 2) * Rc * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Rb + Rc) * Rd * Rd + ((Ag + 4) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), -(Ag * Rb * Rc * Rd * Rd * Ri + (Ag * Ag + Ag) * Rc * Rd * Rd * Ri * Ri - ((2 * Rb + Rc) * Rd * Ri + Rd * Ri * Ri + (Rb * Rb + Rb * Rc) * Rd) * Ro * Ro + ((Rb * Rb + Rb * Rc) * Rd * Rd - (Ag * Rc * Rd + (Ag - 1) * Rd * Rd) * Ri * Ri - (Ag * Rb * Rc * Rd + ((Ag - 2) * Rb + (Ag - 1) * Rc) * Rd * Rd) * Ri) * Ro) / ((Ag + 1) * Rc * Rd * Rd * Ri * Ri + ((Ag + 2) * Rb * Rc + (Ag + 1) * Rc * Rc) * Rd * Rd * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + ((Ag + 2) * Rc * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Rb + Rc) * Rd * Rd + ((Ag + 4) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), -((Ag + 1) * Rc * Rd * Rd * Ri + Rb * Rc * Rd * Rd - (Rb * Rc + Rc * Ri) * Ro * Ro - ((Rb + Rc) * Rd * Rd + Rd * Rd * Ri) * Ro) / ((Ag + 1) * Rc * Rd * Rd * Ri + Rb * Rc * Rd * Rd + (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro * Ro - (2 * Rb * Rc * Rd + (Rb + Rc) * Rd * Rd + ((Ag + 2) * Rc * Rd + Rd * Rd) * Ri) * Ro) } });

            const auto Ra = ((Ag + 1) * Rc * Rd * Ri + Rb * Rc * Rd - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro) / ((Rb + Rc) * Rd + Rd * Ri - (Rb + Rc + Ri) * Ro);
            return Ra;
        }
    };

    wdft::RtypeAdaptor<float, 0, ImpedanceCalc, decltype (P1), decltype (S2), decltype (RL)> R { std::tie (P1, S2, RL) };

    // Port A
    static constexpr auto R6 = 51.0e3f;
    static constexpr auto Pot1 = 500.0e3f;
    wdft::ResistorT<float> R6_P1 { R6 };
    wdft::CapacitorT<float> C4 { 51.0e-12f };
    wdft::WDFParallelT<float, decltype (R6_P1), decltype (C4)> P2 { R6_P1, C4 };
    wdft::WDFParallelT<float, decltype (P2), decltype (R)> P3 { P2, R };

    wdft::DiodePairT<float, decltype (P3)> dpApprox { P3, 4.352e-9f, 25.85e-3f, 1.906f }; // 1N4148
    DiodePairNeuralModel<decltype (P3), 2, 16> dp2x8Model { P3, "_1N4148_1U1D_2x16_training_2000_json" };

    int modelChoice = 0;
    int prevModelChoice = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TubeScreamer)
};

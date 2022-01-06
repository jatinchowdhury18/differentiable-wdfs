#include "DiodeClipper.h"

namespace
{
const String gainTag = "gain";
const String cutoffTag = "cutoff";
} // namespace

DiodeClipper::DiodeClipper (const String& prefix, AudioProcessorValueTreeState& vts)
{
    for (auto& paramTag : { gainTag, cutoffTag })
        paramTags.add (prefix + paramTag);

    gainDBParam = vts.getRawParameterValue (prefix + gainTag);
    cutoffHzParam = vts.getRawParameterValue (prefix + cutoffTag);
}

void DiodeClipper::addParameters (chowdsp::Parameters& params, const String& prefix)
{
    using namespace chowdsp::ParamUtils;
    emplace_param<VTSParam> (params, prefix + gainTag, "Gain", String(), NormalisableRange { -18.0f, 18.0f }, 0.0f, &gainValToString, &stringToGainVal);

    NormalisableRange cutoffRange { 200.0f, 20000.0f };
    cutoffRange.setSkewForCentre (2000.0f);
    emplace_param<VTSParam> (params, prefix + cutoffTag, "Cutoff", String(), cutoffRange, 4000.0f, &freqValToString, &stringToFreqVal);
}

void DiodeClipper::prepare (double sampleRate, int samplesPerBlock)
{
    inputGain.prepare ({ sampleRate, (uint32) samplesPerBlock, 1 });
    inputGain.setRampDurationSeconds (0.02);

    C.prepare ((float) sampleRate);
}

void DiodeClipper::process (AudioBuffer<float>& buffer)
{
    // process input gain
    inputGain.setGainDecibels (gainDBParam->load());
    dsp::AudioBlock<float> block { buffer };
    inputGain.process (dsp::ProcessContextReplacing<float> { block });

    // set cutoff frequency
    const auto resVal = 1.0f / (MathConstants<float>::twoPi * *cutoffHzParam * capVal);
    Vs.setResistanceValue (resVal);

    auto* x = buffer.getWritePointer (0); // mono only!
    for (int n = 0; n < buffer.getNumSamples(); ++n)
    {
        Vs.setVoltage (x[n]);

        dp.incident (P1.reflected());
        x[n] = wdft::voltage<float> (C);
        P1.incident (dp.reflected());
    }
}

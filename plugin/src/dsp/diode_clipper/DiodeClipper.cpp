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
    emplace_param<VTSParam> (params, prefix + gainTag, "Gain", String(), NormalisableRange { -12.0f, 12.0f }, 0.0f, &gainValToString, &stringToGainVal);

    NormalisableRange cutoffRange { 200.0f, 20000.0f };
    cutoffRange.setSkewForCentre (2000.0f);
    emplace_param<VTSParam> (params, prefix + cutoffTag, "Cutoff", String(), cutoffRange, 4000.0f, &freqValToString, &stringToFreqVal);
}

void DiodeClipper::prepare (double sampleRate, int samplesPerBlock)
{
    inputGain.prepare ({ sampleRate, (uint32) samplesPerBlock, 1 });
    inputGain.setRampDurationSeconds (0.02);
}

void DiodeClipper::process (AudioBuffer<float>& buffer)
{
    inputGain.setGainDecibels (gainDBParam->load());

    dsp::AudioBlock<float> block { buffer };
    inputGain.process (dsp::ProcessContextReplacing<float> { block });
}

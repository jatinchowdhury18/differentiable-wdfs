#include "DifferentiableWDFPlugin.h"
#include "gui/PluginEditor.h"

using namespace DiffWDFParams;

DifferentiableWDFPlugin::DifferentiableWDFPlugin() : diodeClipper (diodeClipperPrefix, vts),
                                                     multiDiodeClipper (multiDiodeClipperPrefix, vts),
                                                     tubeScreamer (tubeScreamerPrefix, vts)
{
    modelChoiceParam = vts.getRawParameterValue (circuitChoiceTag);
}

void DifferentiableWDFPlugin::addParameters (Parameters& params)
{
    DiodeClipper::addParameters (params, diodeClipperPrefix);
    MultiDiodeClipper::addParameters (params, multiDiodeClipperPrefix);
    TubeScreamer::addParameters (params, tubeScreamerPrefix);

    chowdsp::ParamUtils::emplace_param<AudioParameterChoice> (params, circuitChoiceTag, "Circuit", circuitChoices, 0);
}

void DifferentiableWDFPlugin::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    diodeClipper.prepare (sampleRate, samplesPerBlock);
    multiDiodeClipper.prepare (sampleRate, samplesPerBlock);
    tubeScreamer.prepare (sampleRate, samplesPerBlock);

    monoBuffer.setSize (1, samplesPerBlock);

    dcBlocker.reset();
    dcBlocker.calcCoefs (25.0f, (float) sampleRate);

    loadMeasurer.reset (sampleRate, samplesPerBlock);
}

void DifferentiableWDFPlugin::releaseResources()
{
}

void DifferentiableWDFPlugin::processAudioBlock (AudioBuffer<float>& buffer)
{
    const auto numChannels = buffer.getNumChannels();
    const auto numSamples = buffer.getNumSamples();

    AudioProcessLoadMeasurer::ScopedTimer loadTimer { loadMeasurer, numSamples };

    // for testing with just channel 1 input
    //    buffer.copyFrom (0, 0, buffer, 1, 0, numSamples);

    // sum input to mono
    if (numChannels == 1)
    {
        monoBuffer.makeCopyOf (buffer);
    }
    else
    {
        monoBuffer.setSize (1, numSamples, false, false, true);
        monoBuffer.copyFrom (0, 0, buffer, 0, 0, numSamples);

        for (int ch = 1; ch < numChannels; ++ch)
            monoBuffer.addFrom (0, 0, buffer, ch, 0, numSamples);

        monoBuffer.applyGain (1.0f / (float) numChannels);
    }

    // circuit model process
    const auto circuitChoice = (int) *modelChoiceParam;
    if (circuitChoice == 0)
        diodeClipper.process (monoBuffer);
    else if (circuitChoice == 1)
        multiDiodeClipper.process (monoBuffer);
    else if (circuitChoice == 2)
        tubeScreamer.process (monoBuffer);
    else
        jassertfalse; // unknown circuit!

    dcBlocker.processBlock (monoBuffer.getWritePointer (0), numSamples);

    // split back to multi-channel
    for (int ch = 0; ch < numChannels; ++ch)
        buffer.copyFrom (ch, 0, monoBuffer, 0, 0, numSamples);
}

AudioProcessorEditor* DifferentiableWDFPlugin::createEditor()
{
    return new PluginEditor (*this);
}

// This creates new instances of the plugin
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DifferentiableWDFPlugin();
}

#include "DifferentiableWDFPlugin.h"
#include "gui/PluginEditor.h"

using namespace DiffWDFParams;

DifferentiableWDFPlugin::DifferentiableWDFPlugin() : diodeClipper (diodeClipperPrefix, vts)
{
}

void DifferentiableWDFPlugin::addParameters (Parameters& params)
{
    DiodeClipper::addParameters (params, diodeClipperPrefix);
}

void DifferentiableWDFPlugin::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    diodeClipper.prepare (sampleRate, samplesPerBlock);

    monoBuffer.setSize (1, samplesPerBlock);

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
    diodeClipper.process (monoBuffer);

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

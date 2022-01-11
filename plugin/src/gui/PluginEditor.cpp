#include "PluginEditor.h"

namespace
{
constexpr float topBarHeight = 0.15f;
}

PluginEditor::PluginEditor (DifferentiableWDFPlugin& p) : AudioProcessorEditor (p),
                                                          plugin (p),
                                                          modelGui (p.getDiodeClipper(), p.getVTS()),
                                                          cpuMeter (p)
{
    setSize (600, 400);

    addAndMakeVisible (cpuMeter);
    addAndMakeVisible (circuitModelSelector);
    addAndMakeVisible (modelGui);
}

void PluginEditor::paint (Graphics& g)
{
    g.fillAll (Colours::black);

    auto bounds = getLocalBounds();
    auto topBarBounds = bounds.removeFromTop (proportionOfHeight (topBarHeight));

    g.setColour (Colours::white);
    g.setFont (float (topBarBounds.getHeight()) * 0.45f);
    g.drawLine (Line { topBarBounds.getBottomLeft(), topBarBounds.getBottomRight() }.toFloat(), 2.0f);

    auto nameBounds = topBarBounds.removeFromLeft (proportionOfWidth (0.5f)).reduced (10);
    g.drawFittedText ("Differentiable WDFs", nameBounds, Justification::centred, 1);
}

void PluginEditor::resized()
{
    auto bounds = getLocalBounds();
    auto topBarBounds = bounds.removeFromTop (proportionOfHeight (topBarHeight));

    topBarBounds.removeFromLeft (proportionOfWidth (0.5f));
    circuitModelSelector.setBounds (topBarBounds.removeFromLeft (proportionOfWidth (0.32f)).reduced (5, 10));
    cpuMeter.setBounds (topBarBounds.reduced (5, 10));

    modelGui.setBounds (bounds);
}

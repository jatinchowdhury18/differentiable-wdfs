#pragma once

#include <pch.h>

class CircuitModel
{
public:
    CircuitModel() = default;

    virtual void prepare (double sampleRate, int samplesPerBlock) = 0;

    virtual void process (AudioBuffer<float>& buffer) = 0;

    const StringArray& getParamTags() const { return paramTags; }

protected:
    StringArray paramTags;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CircuitModel)
};

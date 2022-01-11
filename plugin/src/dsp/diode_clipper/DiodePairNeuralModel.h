#pragma once

#include <pch.h>

template <typename Next>
class DiodePairNeuralModel : public wdft::RootWDF
{
public:
    explicit DiodePairNeuralModel (Next& n, const String& modelName) : next (n)
    {
        DBG ("Initalizing diode model: " + modelName);
#if JUCE_DEBUG
        constexpr bool debug = true;
#else
        constexpr bool debug = false;
#endif

        int modelDataSize;
        auto modelData = ModelData::getNamedResource (modelName.getCharPointer(), modelDataSize);
        jassert (modelData != nullptr);

        MemoryInputStream modelInputStream { modelData, (size_t) modelDataSize, false };
        auto jsonInput = nlohmann::json::parse (modelInputStream.readEntireStreamAsString().toStdString());
        model.parseJson (jsonInput, debug);
    }

    void calcImpedance() override { logR = std::log (next.wdf.R); }

    inline void incident (float x) { wdf.a = x; }

    inline float reflected()
    {
        const float inData alignas(16)[] = { wdf.a, logR };
        wdf.b = -model.template forward (inData);
        return wdf.b;
    }

    wdft::WDFMembers<float> wdf;

private:
    const Next& next;
    float logR = 1.0f;

    RTNeural::ModelT<float,
                     2,
                     1,
                     RTNeural::DenseT<float, 2, 8>,
                     RTNeural::TanhActivationT<float, 8>,
                     RTNeural::DenseT<float, 8, 8>,
                     RTNeural::TanhActivationT<float, 8>,
                     RTNeural::DenseT<float, 8, 8>,
                     RTNeural::TanhActivationT<float, 8>,
                     RTNeural::DenseT<float, 8, 8>,
                     RTNeural::TanhActivationT<float, 8>,
                     RTNeural::DenseT<float, 8, 1>> model;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DiodePairNeuralModel)
};

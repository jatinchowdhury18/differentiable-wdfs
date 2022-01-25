#pragma once

#include <pch.h>
#include <toms917.hpp>

/**
 * Wave domain diode pair model basedon TOMS917 algorithm
 * for evaluating the Wright Omega function.
 *
 * Reference: https://people.sc.fsu.edu/~jburkardt/cpp_src/toms917/toms917.html
 */
template <typename T, typename Next, wdft::DiodeQuality Quality = wdft::DiodeQuality::Best>
class Toms917DiodePairT final : public wdft::RootWDF
{
public:
    /** Creates a new WDF diode pair, with the given diode specifications.
         * @param Is: reverse saturation current
         * @param Vt: thermal voltage
         * @param next: the next element in the WDF connection tree
         */
    Toms917DiodePairT (Next& n, T Is, T Vt = typename wdft::WDFMembers<T>::NumericType (25.85e-3), T nDiodes = 1) : next (n)
    {
        next.connectToParent (this);
        setDiodeParameters (Is, Vt, nDiodes);
    }

    /** Sets diode specific parameters */
    void setDiodeParameters (T newIs, T newVt, T nDiodes)
    {
        Is = newIs;
        Vt = nDiodes * newVt;
        twoVt = (T) 2 * Vt;
        oneOverVt = (T) 1 / Vt;
        calcImpedance();
    }

    inline void calcImpedance() override
    {
        R_Is = next.wdf.R * Is;
        R_Is_overVt = R_Is * oneOverVt;
        logR_Is_overVt = std::log (R_Is_overVt);
    }

    /** Accepts an incident wave into a WDF diode pair. */
    inline void incident (T x) noexcept
    {
        wdf.a = x;
    }

    /** Propogates a reflected wave from a WDF diode pair. */
    inline T reflected() noexcept
    {
        // See eqn (39) from reference paper
        T lambda = (T) chowdsp::signum (wdf.a);
        T lambda_a_over_vt = lambda * wdf.a * oneOverVt;
        wdf.b = wdf.a - twoVt * lambda * (tomsOmega (logR_Is_overVt + lambda_a_over_vt) - tomsOmega (logR_Is_overVt - lambda_a_over_vt));

        return wdf.b;
    }

    wdft::WDFMembers<T> wdf;

private:
    inline T tomsOmega (T x)
    {
        return (T) std::real (wrightomega (std::complex<double> ((double) x)));
    }

    T Is; // reverse saturation current
    T Vt; // thermal voltage

    // pre-computed vars
    T twoVt;
    T oneOverVt;
    T R_Is;
    T R_Is_overVt;
    T logR_Is_overVt;

    Next& next;
};

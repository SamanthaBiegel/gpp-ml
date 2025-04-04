import numpy as np
import pandas as pd

# Define the constants from par_splash
par_splash = {
    'kTkelvin': 273.15,  # freezing point in K (= 0 deg C)
    'kTo': 298.15,       # base temperature, K (from P-model)
    'kR': 8.31446262,    # universal gas constant, J/mol/K (Allen, 1973)
    'kMv': 18.02,        # molecular weight of water vapor, g/mol (Tsilingiris, 2008)
    'kMa': 28.963,       # molecular weight of dry air, g/mol (Tsilingiris, 2008)
    'kfFEC': 2.04,       # from flux to energy conversion, umol/J (Meek et al., 1984)
    'kPo': 101325,       # standard atmosphere, Pa (Allen, 1973)
    'kL': 0.0065,        # temperature lapse rate, K/m (Cavcar, 2000)
    'kG': 9.80665,       # gravitational acceleration, m/s^2 (Allen, 1973)
    'k_karman': 0.41,    # Von Karman constant; from bigleaf R package
    'eps': 9.999e-6,     # numerical imprecision allowed in mass conservation tests
    'cp': 1004.834,      # specific heat of air at constant pressure (J K-1 kg-1)
    'Rd': 287.0586,      # gas constant of dry air (J kg-1 K-1)
    'alpha': 1.26        # Priestly-Taylor coefficient
}

def calc_sat_slope(tc):
    """Calculates the slope of the saturation vapor pressure curve (Pa/K)."""
    sat_slope = (17.269 * 237.3 * 610.78 *
                 (np.exp(tc * 17.269 / (tc + 237.3)) / ((tc + 237.3) ** 2)))
    return sat_slope

def calc_enthalpy_vap(tc):
    """
    Calculates the enthalpy of vaporization, J/kg.

    Reference:
        Eq. 8, Henderson-Sellers (1984)

    Args:
        tc: Air temperature in degrees Celsius.

    Returns:
        Enthalpy of vaporization in J/kg.
    """
    enthalpy_vap = 1.91846e6 * ((tc + 273.15) / (tc + 273.15 - 33.91)) ** 2
    return enthalpy_vap

def calc_density_h2o(tc, patm):
    """
    Calculates density of water at a given temperature and pressure.

    Reference:
        Chen et al. (1977)

    Args:
        tc: Air temperature in degrees Celsius.
        patm: Atmospheric pressure in Pascals.

    Returns:
        Density of water in kg/m³.
    """

    # Calculate density at 1 atm (po)
    po = (
        0.99983952
        + 6.788260e-5 * tc
        - 9.08659e-6 * tc ** 2
        + 1.022130e-7 * tc ** 3
        - 1.35439e-9 * tc ** 4
        + 1.471150e-11 * tc ** 5
        - 1.11663e-13 * tc ** 6
        + 5.044070e-16 * tc ** 7
        - 1.00659e-18 * tc ** 8
    )

    # Calculate bulk modulus at 1 atm (ko)
    ko = (
        19652.17
        + 148.1830 * tc
        - 2.29995 * tc ** 2
        + 0.01281 * tc ** 3
        - 4.91564e-5 * tc ** 4
        + 1.035530e-7 * tc ** 5
    )

    # Calculate temperature-dependent coefficients (ca and cb)
    ca = (
        3.26138
        + 5.223e-4 * tc
        + 1.324e-4 * tc ** 2
        - 7.655e-7 * tc ** 3
        + 8.584e-10 * tc ** 4
    )

    cb = (
        7.2061e-5
        - 5.8948e-6 * tc
        + 8.69900e-8 * tc ** 2
        - 1.0100e-9 * tc ** 3
        + 4.3220e-12 * tc ** 4
    )

    # Convert atmospheric pressure to bar (1 bar = 100,000 Pa)
    pbar = (1.0e-5) * patm

    # Calculate density of water (kg/m³)
    numerator = ko + ca * pbar + cb * pbar ** 2
    denominator = ko + ca * pbar + cb * pbar ** 2 - pbar
    density_h2o = 1000.0 * po * (numerator / denominator)

    return density_h2o

def calc_psychro(tc, press, par_splash):
    """Calculates the psychrometric constant (Pa/K)."""
    my_tc = np.clip(tc, 0, 100)  # Adjust temperature to avoid numerical issues

    # Specific heat capacity of water (J/kg/K)
    cp = 1.0e3 * (1.0045714270
                  + 2.050632750e-3 * my_tc
                  - 1.631537093e-4 * my_tc ** 2
                  + 6.212300300e-6 * my_tc ** 3
                  - 8.830478888e-8 * my_tc ** 4
                  + 5.071307038e-10 * my_tc ** 5)

    lv = calc_enthalpy_vap(tc)

    # Psychrometric constant calculation
    psychro = cp * par_splash['kMa'] * press / (par_splash['kMv'] * lv)
    return psychro

def pet(netrad, tc, patm):
    """Calculates the potential evapotranspiration (PET).

    Args:
        netrad: Net radiation (W/m²).
        tc: Air temperature in degrees Celsius.
        patm: Atmospheric pressure (Pa).
        return_df: If True, returns a dictionary; otherwise, returns PET value.

    Returns:
        Potential evapotranspiration (mm/s) or a dictionary containing it.
    """
    sat_slope = calc_sat_slope(tc)
    lv = calc_enthalpy_vap(tc)
    pw = calc_density_h2o(tc, patm)
    gamma = calc_psychro(tc, patm, par_splash)
    econ = sat_slope / (lv * pw * (sat_slope + gamma))  # units: m³ J⁻¹

    # Equilibrium evapotranspiration in mm/s
    eet = netrad * econ * 1000

    # Priestley-Taylor potential evapotranspiration
    pet_val = par_splash['alpha'] * eet

    return pet_val

def cwd(df, thresh_terminate=0.0, thresh_drop=0.9, doy_reset=999):
    df = df.copy()

    df['doy'] = pd.to_datetime(df['TIMESTAMP']).dt.dayofyear

    df['iinst'] = pd.NA
    df['dday'] = pd.NA
    df['cwd'] = 0.0

    idx = 0
    iinst = 1
    n = len(df)

    while idx < n:
        # If the water balance is negative, start accumulating deficit
        if df['pwbal'].iloc[idx] < 0:
            dday = 0
            deficit = 0.0
            max_deficit = 0.0
            iidx = idx
            done_finding_dropday = False

            while (
                iidx < n and
                (deficit - df['pwbal'].iloc[iidx] > thresh_terminate * max_deficit)
            ):
                dday += 1
                deficit -= df['pwbal'].iloc[iidx]

                # Record the maximum deficit attained in this event
                if deficit > max_deficit:
                    max_deficit = deficit
                    done_finding_dropday = False

                # Record the day when deficit falls below (thresh_drop) times the maximum deficit
                if deficit < (max_deficit * thresh_drop) and not done_finding_dropday:
                    done_finding_dropday = True

                # Stop accumulating on reset day
                if df['doy'].iloc[iidx] == doy_reset:
                    max_deficit = deficit
                    break

                # Once deficit has fallen below threshold, subsequent dates are dropped
                if done_finding_dropday:
                    df.at[iidx, 'iinst'] = pd.NA
                    df.at[iidx, 'dday'] = pd.NA
                else:
                    df.at[iidx, 'iinst'] = iinst
                    df.at[iidx, 'dday'] = dday

                df.at[iidx, 'cwd'] = deficit

                iidx += 1

            # Update instance
            iinst += 1
            idx = iidx
        idx += 1

    df.drop(columns=['iinst', 'dday', 'doy'], inplace=True)

    return df

def apply_cwd_per_site(group):
    doy_reset = group['doy_reset'].iloc[0]
    group = group.copy()
    original_index = group.index
    group.reset_index(drop=True, inplace=True)
    result = cwd(
        group,
        thresh_terminate=0.0,
        thresh_drop=0.9,
        doy_reset=doy_reset
    )
    result.index = original_index
    return result
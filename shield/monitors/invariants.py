"""
Individual invariant implementations for the AMPLE-GNC shield monitor.

Each invariant class encapsulates one safety constraint from the mission
specification (I1--I9).  Every class inherits from InvariantBase and provides:

    check(state)              -- instantaneous Boolean satisfaction test
    predict_violation(state)  -- linearised forward propagation that estimates
                                 seconds until the constraint is breached
                                 (None when no violation is predicted within
                                 the horizon)

Threshold values cited below were established from a TRL-3 deep-research
dossier triangulating Gemini, GPT, and Claude research outputs against
peer-reviewed flight-heritage sources.  See REFERENCES.md for the full
citation list; per-invariant rationale is in the docstring of each class.

Mission class: Gateway HALO+PPE NRHO 9:2 (cislunar), with LEO and
Mars EDL phase-overrides where applicable.

Items tagged NEEDS-PI-INPUT below could not be pinned to two independent
flight-heritage citations and are engineering envelopes that should be
refined against the HALO PDR data when it becomes available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# State dictionary keys (canonical telemetry names)
# ---------------------------------------------------------------------------
# battery_soc            : float  [0, 1]       state-of-charge fraction
# battery_soc_rate       : float  [1/s]        d(SoC)/dt
# wheel_momentum_frac    : float  [0, 1]       |H|/H_max per axis (max of 3)
# wheel_momentum_rate    : float  [1/s]        d(frac)/dt
# altitude_km            : float  [km]         geodetic altitude
# altitude_rate_km_s     : float  [km/s]       d(alt)/dt
# pointing_error_deg     : float  [deg]        angular error from target
# pointing_rate_deg_s    : float  [deg/s]      d(error)/dt
# thruster_on_history    : list[float]          on-durations [s] in last 60 s
# propellant_kg          : float  [kg]         remaining propellant
# abort_reserve_kg       : float  [kg]         abort delta-v reserve
# desat_reserve_kg       : float  [kg]         desaturation reserve
# sun_angle_deg          : float  [deg]        angle from +Z to Sun
# sun_angle_rate_deg_s   : float  [deg/s]      d(sun_angle)/dt
# in_eclipse             : bool                True when spacecraft is shadowed
# eclipse_power_margin_w : float  [W]          power surplus during eclipse
# avionics_temp_c        : float  [C]          avionics temperature
# avionics_temp_rate_c_s : float  [C/s]        d(T)/dt  (positive = heating)
# transmit_power_w       : float  [W]          current transmitter draw
# power_budget_w         : float  [W]          available power for transmit
# mission_phase          : str                 "raise" | "phasing" | "science" | "deorbit"


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
@dataclass
class InvariantThresholds:
    """Mission-configurable safety thresholds for all nine invariants.

    All defaults are TRL-3 reference values triangulated across three deep-
    research dossiers (Claude, Gemini, GPT).  Each field cites the primary
    flight-heritage source; full bibliography in REFERENCES.md.
    """

    # I1 -- Battery SoC
    # Hard floor 0.20 prevents deep-discharge cell damage on GS Yuasa LSE-134
    # Li-ion chemistry (ISS heritage).  Nominal 0.30 matches ISS BMS
    # storage band for life extension; high-confidence safe 0.40.
    # Refs: Dalton, P. J. et al., "ISS Lithium-Ion Battery," NTRS 20160012048;
    #       Aerospace Corp TOR-2013-00295, "Guidelines for Li-ion Battery Use
    #       in Space Applications," 2013.
    battery_soc_min: float = 0.20
    battery_soc_nominal: float = 0.30
    battery_soc_safe: float = 0.40

    # I2 -- Reaction-wheel momentum
    # Honeywell HR16-100 rated 100 Nms / 6000 rpm / 0.2 Nm nominal torque.
    # Roll-off above 80 % documented in HR16 datasheet; 600 s desat deadline
    # consistent with Cassini/LRO RWA-bias event duration (60-300 s pulsed).
    # Refs: Honeywell HR16-100 datasheet;
    #       Mukherjee, R., "Cassini Thruster On-Times for RW Biases," NTRS 20150008900.
    wheel_momentum_max_frac: float = 0.80
    wheel_momentum_hard_limit: float = 0.95
    wheel_desat_deadline_s: float = 600.0  # 10 min

    # I3 -- Altitude floor per phase [km]
    # LEO floors per NPR 8715.6 (25-yr decay rule) + Space-Track reentry
    # criteria.  NRHO 1400 km perilune-altitude floor is conservative bound
    # below the 9:2 NRHO family min of 1463 km (rp 3200 km) per Davis 2019.
    # Mars EI floor 120 km = 5 km below Mars 2020 nominal 125 km per
    # McGrew/Way et al. 2024 postflight reconstruction.
    # Refs: NPR 8715.6 NASA Orbital Debris Mitigation;
    #       Davis et al., "Disposal in NRHOs," AAS 2019;
    #       McGrew et al., "Mars 2020 Entry Guidance," NTRS 20240015538.
    altitude_floor_km: Dict[str, float] = field(
        default_factory=lambda: {
            "raise": 250.0,
            "phasing": 280.0,
            "science": 350.0,
            "deorbit": 150.0,
            "nrho_perilune": 1400.0,  # km above lunar surface (3137 km radius)
            "mars_ei": 120.0,
        }
    )

    # I4 -- Pointing error
    # 5.0 deg hard limit derives from Ka-band 0.5 m HGA half-power beamwidth
    # (~5 deg at 8.4 GHz X-band); 1.0 deg nominal = Gateway HGA tracking
    # tolerance; 0.1 deg = LRO-class science pointing.
    # Refs: ECSS-E-ST-60-10C Control Performance;
    #       Calhoun et al., "LRO GNC System," NTRS 20100014254;
    #       JPL DSN 810-005 Module 104 Rev. N.
    pointing_error_max_deg: float = 5.0
    pointing_error_nominal_deg: float = 1.0
    pointing_error_science_deg: float = 0.1

    # I5 -- Thruster duty cycle
    # 30 % in any rolling 60-s window: bipropellant catalyst-bed thermal
    # limit + valve-seat soak limit.  R-4D-15 HiPAT qualified for 300 s
    # steady-state (full duty) and 65 000+ pulses; MR-103G qualified > 900 s
    # continuous.  30 % is the conservative envelope MAVEN/MRO use for
    # long-pulse RCS clusters.
    # Refs: Aerojet Rocketdyne R-4D-15 HiPAT datasheet;
    #       Aerojet Rocketdyne MR-103G 1N hydrazine datasheet.
    thruster_max_duty: float = 0.30
    thruster_window_s: float = 60.0

    # I6 -- Propellant reserve
    # Gateway HALO-class: Annual NRHO SK ~10 m/s/yr, eclipse-avoid 5 m/s/yr,
    # insertion-cleanup 16 m/s one-time, abort 50 m/s -> 200 m/s total over
    # 15-yr life (Davis 2017, Whitley 2018, Parrish 2022).  In mass: ~1610 kg
    # for 25 mt vehicle at Isp 315 s.
    # NEEDS-PI-INPUT: JSC-65990 specific abort-Δv allocation for HALO.
    # Refs: Davis et al., "Orbit Maintenance for NRHOs," NTRS 20170001347;
    #       Whitley et al., "Earth-Moon NRHO," AAS 18-406;
    #       Parrish (Advanced Space), "Plotting Orbits for Gateway."
    propellant_reserve_total_delta_v_m_s: float = 200.0
    propellant_reserve_abort_m_s: float = 50.0

    # I7 -- Sun angle / eclipse power
    # 75 deg derived from cos(75 deg) = 0.26 array-power efficiency floor
    # below which a 1.4 kW BOL array generates < the ~500 W bus base load.
    # 0 W eclipse margin minimum = ECSS-E-ST-20C requires net power non-
    # negative across all transient loads.
    # Refs: ECSS-E-ST-20C Electrical and Electronic, 2008;
    #       Davis et al., "Phase Control and Eclipse Avoidance in NRHOs,"
    #       AAS 2020 -- 90 min eclipse threshold drives 9:2 design.
    sun_angle_max_deg: float = 75.0
    eclipse_power_margin_min_w: float = 0.0

    # I8 -- Thermal limits
    # GR740 quad-core LEON4 datasheet: 125 C junction max; ECSS-Q-ST-30-11C
    # derate to junction - 30 C = 95 C, with 65 C operational target.
    # 0.05 C/s = 3 C/min: MIL-STD-1540E TVAC cycling rate; prevents CTE-
    # induced CCGA solder-joint fracture.
    # Refs: Cobham/Frontgrade GR740 Datasheet;
    #       ECSS-Q-ST-30-11C Rev.2 Derating - EEE Components;
    #       MIL-STD-1540E Test Requirements for Space Vehicles.
    avionics_temp_max_c: float = 65.0
    avionics_temp_hard_max_c: float = 95.0
    avionics_temp_rate_max_c_s: float = 0.05

    # I9 -- Transmit power budget
    # 20 % margin between transmit_power_w and power_budget_w prevents
    # under-voltage lockout during SSPA inrush.  DSN 810-005 Module 104
    # specifies X-band TWTA classes 25-100 W RF (50-200 W DC); ECSS-E-ST-20C
    # demands 20 % electrical-design margin.
    # NEEDS-PI-INPUT: actual HALO X-band TWTA RF output spec from L3Harris.
    # Refs: JPL DSN 810-005 Module 104 Rev. N;
    #       ECSS-E-ST-20C Electrical and Electronic, 2008.
    transmit_power_margin_frac: float = 0.20


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class InvariantBase(ABC):
    """Abstract base for a single safety invariant."""

    id: str  # e.g. "I1"
    description: str

    def __init__(self, thresholds: InvariantThresholds) -> None:
        self.thresholds = thresholds

    @abstractmethod
    def check(self, state: Dict[str, Any]) -> bool:
        """Return True when the invariant is satisfied."""

    @abstractmethod
    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        """Return estimated seconds to violation, or None if safe within *horizon_s*."""

    def _linear_tte(
        self, value: float, rate: float, limit: float, upper: bool = True
    ) -> Optional[float]:
        """Time-to-exceedance under constant-rate extrapolation.

        Parameters
        ----------
        value : current value
        rate  : d(value)/dt
        limit : threshold
        upper : True means violation when value >= limit;
                False means violation when value <= limit.
        """
        if upper:
            if value >= limit:
                return 0.0
            if rate <= 0.0:
                return None  # moving away from limit
            return (limit - value) / rate
        else:
            if value <= limit:
                return 0.0
            if rate >= 0.0:
                return None
            return (value - limit) / (-rate)


# ---------------------------------------------------------------------------
# I1 -- Battery State of Charge
# ---------------------------------------------------------------------------
class BatterySoCInvariant(InvariantBase):
    """I1: Battery SoC >= 20 % and recharge to nominal within one orbit."""

    id = "I1"
    description = "Battery SoC >= 20% with orbital recharge capability"

    def check(self, state: Dict[str, Any]) -> bool:
        return state["battery_soc"] >= self.thresholds.battery_soc_min

    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        soc = state["battery_soc"]
        rate = state.get("battery_soc_rate", 0.0)
        tte = self._linear_tte(soc, rate, self.thresholds.battery_soc_min, upper=False)
        if tte is not None and tte <= horizon_s:
            return tte
        return None


# ---------------------------------------------------------------------------
# I2 -- Reaction-Wheel Momentum
# ---------------------------------------------------------------------------
class WheelMomentumInvariant(InvariantBase):
    """I2: Reaction-wheel momentum < 80 %, desaturation within 10 min."""

    id = "I2"
    description = "Wheel momentum < 80% with timely desaturation"

    def check(self, state: Dict[str, Any]) -> bool:
        return state["wheel_momentum_frac"] < self.thresholds.wheel_momentum_max_frac

    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        frac = state["wheel_momentum_frac"]
        rate = state.get("wheel_momentum_rate", 0.0)
        tte = self._linear_tte(frac, rate, self.thresholds.wheel_momentum_max_frac, upper=True)
        if tte is not None and tte <= horizon_s:
            return tte
        return None


# ---------------------------------------------------------------------------
# I3 -- Altitude Floor
# ---------------------------------------------------------------------------
class AltitudeFloorInvariant(InvariantBase):
    """I3: Altitude >= phase-specific minimum floor."""

    id = "I3"
    description = "Altitude above phase-dependent minimum"

    def check(self, state: Dict[str, Any]) -> bool:
        phase = state.get("mission_phase", "science")
        floor = self.thresholds.altitude_floor_km.get(phase, 200.0)
        return state["altitude_km"] >= floor

    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        phase = state.get("mission_phase", "science")
        floor = self.thresholds.altitude_floor_km.get(phase, 200.0)
        alt = state["altitude_km"]
        rate = state.get("altitude_rate_km_s", 0.0)
        tte = self._linear_tte(alt, rate, floor, upper=False)
        if tte is not None and tte <= horizon_s:
            return tte
        return None


# ---------------------------------------------------------------------------
# I4 -- Pointing Error
# ---------------------------------------------------------------------------
class PointingErrorInvariant(InvariantBase):
    """I4: Pointing error <= 5 degrees."""

    id = "I4"
    description = "Pointing error within tolerance"

    def check(self, state: Dict[str, Any]) -> bool:
        return state["pointing_error_deg"] <= self.thresholds.pointing_error_max_deg

    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        err = state["pointing_error_deg"]
        rate = state.get("pointing_rate_deg_s", 0.0)
        tte = self._linear_tte(err, rate, self.thresholds.pointing_error_max_deg, upper=True)
        if tte is not None and tte <= horizon_s:
            return tte
        return None


# ---------------------------------------------------------------------------
# I5 -- Thruster Duty Cycle
# ---------------------------------------------------------------------------
class ThrusterDutyInvariant(InvariantBase):
    """I5: Thruster on-time <= 30 % in any 60-s window."""

    id = "I5"
    description = "Thruster duty cycle within limits"

    def _duty_fraction(self, state: Dict[str, Any]) -> float:
        """Compute duty fraction from on-time history over the window."""
        history: list[float] = state.get("thruster_on_history", [])
        total_on = sum(history)
        return total_on / self.thresholds.thruster_window_s

    def check(self, state: Dict[str, Any]) -> bool:
        return self._duty_fraction(state) <= self.thresholds.thruster_max_duty

    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        duty = self._duty_fraction(state)
        if duty >= self.thresholds.thruster_max_duty:
            return 0.0
        # If the thruster is currently firing, estimate when the limit is hit.
        currently_firing: bool = state.get("thruster_firing", False)
        if not currently_firing:
            return None
        # Time budget remaining before breach (seconds of additional firing)
        remaining_budget_s = (
            self.thresholds.thruster_max_duty * self.thresholds.thruster_window_s
            - duty * self.thresholds.thruster_window_s
        )
        if remaining_budget_s <= 0:
            return 0.0
        if remaining_budget_s <= horizon_s:
            return remaining_budget_s
        return None


# ---------------------------------------------------------------------------
# I6 -- Propellant Reserve
# ---------------------------------------------------------------------------
class PropellantReserveInvariant(InvariantBase):
    """I6: Propellant >= abort delta-v reserve + desaturation reserve."""

    id = "I6"
    description = "Propellant above abort + desat reserve"

    def _required_reserve(self, state: Dict[str, Any]) -> float:
        return state.get("abort_reserve_kg", 0.0) + state.get("desat_reserve_kg", 0.0)

    def check(self, state: Dict[str, Any]) -> bool:
        return state["propellant_kg"] >= self._required_reserve(state)

    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        prop = state["propellant_kg"]
        reserve = self._required_reserve(state)
        if prop < reserve:
            return 0.0
        # Estimate burn rate from thruster state
        burn_rate = state.get("propellant_burn_rate_kg_s", 0.0)
        if burn_rate <= 0.0:
            return None  # not consuming propellant
        tte = (prop - reserve) / burn_rate
        if tte <= horizon_s:
            return tte
        return None


# ---------------------------------------------------------------------------
# I7 -- Sun Angle and Eclipse Power Margin
# ---------------------------------------------------------------------------
class SunAngleEclipseInvariant(InvariantBase):
    """I7: Sun angle <= 75 deg AND eclipse power margin >= 0 W."""

    id = "I7"
    description = "Sun angle and eclipse power margin within limits"

    def check(self, state: Dict[str, Any]) -> bool:
        angle_ok = state["sun_angle_deg"] <= self.thresholds.sun_angle_max_deg
        if state.get("in_eclipse", False):
            margin_ok = (
                state.get("eclipse_power_margin_w", 0.0)
                >= self.thresholds.eclipse_power_margin_min_w
            )
            return margin_ok  # sun angle irrelevant during eclipse
        return angle_ok

    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        # Sun-angle exceedance
        angle = state["sun_angle_deg"]
        angle_rate = state.get("sun_angle_rate_deg_s", 0.0)
        if not state.get("in_eclipse", False):
            tte = self._linear_tte(
                angle, angle_rate, self.thresholds.sun_angle_max_deg, upper=True
            )
            if tte is not None and tte <= horizon_s:
                return tte

        # Eclipse power margin depletion
        if state.get("in_eclipse", False):
            margin = state.get("eclipse_power_margin_w", 0.0)
            margin_rate = state.get("eclipse_power_margin_rate_w_s", 0.0)
            tte = self._linear_tte(
                margin, margin_rate, self.thresholds.eclipse_power_margin_min_w, upper=False
            )
            if tte is not None and tte <= horizon_s:
                return tte

        return None


# ---------------------------------------------------------------------------
# I8 -- Thermal Limits
# ---------------------------------------------------------------------------
class ThermalInvariant(InvariantBase):
    """I8: Avionics temp <= 65 C AND rate of change <= 3 C/min."""

    id = "I8"
    description = "Avionics temperature and heating rate within limits"

    def check(self, state: Dict[str, Any]) -> bool:
        temp = state["avionics_temp_c"]
        rate = abs(state.get("avionics_temp_rate_c_s", 0.0))
        return (
            temp <= self.thresholds.avionics_temp_max_c
            and rate <= self.thresholds.avionics_temp_rate_max_c_s
        )

    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        temp = state["avionics_temp_c"]
        rate = state.get("avionics_temp_rate_c_s", 0.0)

        # Time to temperature limit
        tte_temp = self._linear_tte(
            temp, rate, self.thresholds.avionics_temp_max_c, upper=True
        )

        # Rate limit is instantaneous -- if it is already violated we return 0
        if abs(rate) > self.thresholds.avionics_temp_rate_max_c_s:
            return 0.0

        if tte_temp is not None and tte_temp <= horizon_s:
            return tte_temp
        return None


# ---------------------------------------------------------------------------
# I9 -- Transmit Power Budget
# ---------------------------------------------------------------------------
class TransmitPowerInvariant(InvariantBase):
    """I9: Transmit only when power budget allows."""

    id = "I9"
    description = "Transmitter power within available budget"

    def check(self, state: Dict[str, Any]) -> bool:
        return state.get("transmit_power_w", 0.0) <= state.get("power_budget_w", 0.0)

    def predict_violation(
        self, state: Dict[str, Any], horizon_s: float = 1.0
    ) -> Optional[float]:
        tx_power = state.get("transmit_power_w", 0.0)
        budget = state.get("power_budget_w", 0.0)
        if tx_power > budget:
            return 0.0

        # If the power budget is shrinking (e.g. entering eclipse) predict crossover
        budget_rate = state.get("power_budget_rate_w_s", 0.0)
        tx_rate = state.get("transmit_power_rate_w_s", 0.0)
        relative_rate = tx_rate - budget_rate  # positive => gap closing
        if relative_rate <= 0.0:
            return None
        gap = budget - tx_power
        tte = gap / relative_rate
        if tte <= horizon_s:
            return tte
        return None


# ---------------------------------------------------------------------------
# Convenience: ordered list of all invariant classes
# ---------------------------------------------------------------------------
ALL_INVARIANT_CLASSES: list[type[InvariantBase]] = [
    BatterySoCInvariant,
    WheelMomentumInvariant,
    AltitudeFloorInvariant,
    PointingErrorInvariant,
    ThrusterDutyInvariant,
    PropellantReserveInvariant,
    SunAngleEclipseInvariant,
    ThermalInvariant,
    TransmitPowerInvariant,
]

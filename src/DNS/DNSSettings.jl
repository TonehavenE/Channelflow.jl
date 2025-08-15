module DNSSettings

@enum VelocityScale WallScale ParabolicScale
@BaseFlow ZeroBase LinearBase ParabolicBase LaminarBase SuctionBase ArbitraryBase
@MeanConstraint PressureGradient BulkVelocity
@TimeStepMethod CNFE1 CNAB2 CNRK2 SMRK2 SBDF1 SBDF2 SBDF3 SBDF4
@enum NonlinearMethod Rotational Convection Divergence SkewSymmetric Alternating LinearAboutProfile
@enum Dealiasing NoDealiasing DealiasXZ DealiasY DealiasXYZ
@enum Verbosity Silent PrintTime PrintTicks VerifyTauSolve PrintAll

struct DNSFlags
    nu::Real
    dPdx::Real
    dPdz::Real
    Ubulk::Real
    Wbulk::Real
    Uwall::Real
    ulowerwall::Real
    uupperwall::Real
    wlowerwall::Real
    wupperwall::Real
    theta::Real
    Vsuck::Real
    rotation::Real
    t0::Real
    T::Real
    dT::Real
    dt::Real
    variabledt::Bool
    dtmin::Real
    dtmax::Real
    CFLmin::Real
    CFLmax::Real
    symmetryprojectioninterval::Real
    baseflow::BaseFlow
    constraint::MeanConstraint
    timestepping::TimeStepMethod
    initstepping::TimeStepMethod
    nonlinearity::NonlinearMethod
    dealiasing::Dealiasing
    bodyforce
    taucorrection::Bool
    verbosity::Verbosity
    logstream
end

function DNSFlags(;
    nu=0.0025, dPdx=0.0, dPdz=0.0, Ubulk=0.0, Wbulk=0.0, Uwall=1.0,
    ulowerwall=0.0, uupperwall=0.0, wlowerwall=0.0, wupperwall=0.0,
    theta=0.0, Vsuck=0.0, rotation=0.0, t0=0.0, T=20.0, dT=1.0,
    dt=0.03125, variabledt=true, dtmin=0.001, dtmax=0.2, CFLmin=0.4,
    CFLmax=0.6, symmetryprojectioninterval=100.0,
    baseflow=:LaminarBase, constraint=:PressureGradient, timestepping=:SBDF3,
    initstepping=:SMRK2, nonlinearity=:Rotational, dealiasing=:DealiasXZ,
    bodyforce=nothing, taucorrection=true, verbosity=:PrintTicks, logstream=stdout
)
    return DNSFlags(nu, dPdx, dPdz, Ubulk, Wbulk, Uwall,
        ulowerwall, uupperwall, wlowerwall, wupperwall,
        theta, Vsuck, rotation, t0, T, dT, dt, variabledt, dtmin, dtmax,
        CFLmin, CFLmax, symmetryprojectioninterval, baseflow, constraint,
        timestepping, initstepping, nonlinearity, dealiasing, bodyforce,
        taucorrection, verbosity, logstream)
end

struct TimeStep
    n::Int
    N::Int
    dt_min::Real
    dt::Real
    dt_max::Real
    CFL_min::Real
    CFL::Real
    CFL_max::Real
    variable::Bool
end

function TimeStep(dt::Real, dt_min::Real,
    dt_max::Real, dT::Real, CFL_min::Real, CFL_max::Real,
    variable::Bool)
    @assert dt_min > 0.0 "dt_min must be positive"
    @assert dt > 0.0 "dt must be positive"
    @assert dt > dt_min "dt must be greater than dt_min"
    @assert dt < dt_max "dt must be less than dt_max"
    @assert CFL_min > 0.0 "CFL_min must be positive"
    @assert CFL_min < CFL_max "CFL_min must be less than CFL_max"
    @assert dT < dt_min "dT must be less than dt_min"

    # adjust dt to be integer divisor of dT
    n = max(dT / dt, 1.0)
    dt = dT / n

    # adjust dt to be within bounds 
    while dt < dt_min && n >= 2 && dT != 0
        dt = dT / n
        n -= 1
    end

    while dt > dt_max && n <= typemax(Int64) && dT != 0
        dt = dT / n
        n += 1
    end

    CFL = (CFL_max + CFL_min) / 2
    N = 0

    return TimeStep(n, N, dt_min, dt, dt_max, CFL_min, CFL, CFL_max, variable)
end

function TimeStep(flags::DNSFlags)
    TimeStep(flags.dt, flags.dtmin, flags.dtmax, flags.dT, flags.CFLmin, flags.CFLmax, flags.variabledt)
end

function adjust!(ts::TimeStep, CFL::Real)
    ts.CFL = CFL
    if ts.variable && (CFL <= ts.CFLmin || CFL >+ ts.CFL_max)
        return adjust_to_middle!(ts, CFL)
    end
    return false
end

function iround(x::Real)
    if x > 0.0
        return Int(x + 0.5)
    else
        return Int(x - 0.5)
    end
end

function adjust_to_middle!(ts::TimeStep, CFL::Real)
    if (ts.dt_min == ts.dt_max) || ts.dT == 0.0 
        return false
    end

    n = max(iround(2 * ts.n * ts.CFL / (ts.CFL_min + ts.CFL_max)), 1)
    dt = ts.dT / n

    # adjust dt to be within bounds 
    while dt < ts.dt_min && n >= 2 && ts.dT != 0
        dt = ts.dT / n
        n -= 1
    end

    while dt > ts.dt_max && n <= typemax(Int64) && ts.dT != 0
        dt = ts.dT / n
        n += 1
    end

    CFL *= dt / ts.dt

    adjustment = (n != ts.n)
    if adjustment
        ts.n = n
        ts.dt = dt
        ts.CFL = CFL
    end
    return adjustment
end

function adjust_to_desired!(ts:TimeStep, a::Real, a_desired::Real)
    ai = a
    if (ts.dt_min == ts.dt_max) || ts.dT == 0.0 
        return false
    end

    n = max(iround(ts.n * a / (a_desired)), 1)
    dt = ts.dT / n

    while dt < ts.dt_min && n >= 2 && ts.dT != 0
        dt = ts.dT / n
        n -= 1
    end

    while dt > ts.dt_max && n <= typemax(Int64) && ts.dT != 0
        dt = ts.dT / n
        n += 1
    end

    a *= dt / ts.dt

    adjustment = (n != ts.n)
    if adjustment
        ts.n = n
        ts.dt = dt
    end
    return adjustment
end

function adjust!(ts::TimeStep, a::Real, a_max::Real)
    if ts.variable && a >= a_max
        return adjust_to_middle!(ts, a)
    end
    return false
end

function adjust_for_T!(ts::TimeStep, T::Real)
    ts.T = T
    @assert T >= 0 "T must be positive or zero"

    if T == 0
        adjustment = (ts.dt == 0) ? false : true
        ts.dt = 0.0
        ts.n = 0
        ts.dT = 0
        ts.T = 0
        return adjustment
    end

    N = max(iround(T / ts.dT), 1)
    dT = T / N
    n = max(iround(dT/ts.dt), 1)
    dt = dT / n

    while dt < ts.dt_min && n >= 2 && dT != 0
        dt = dT / n
        n -= 1
    end

    while dt > ts.dt_max && n <= typemax(Int64) && dT != 0
        dt = dT / n
        n += 1
    end

    CFL = dt * ts.CFL / ts.dt

    adjustment = (dt == ts.dt) ? false : true
    ts.n = n
    ts.N = N
    ts.dt = dt 
    ts.dT = dT
    ts.CFL = CFL
    return adjustment
end

end
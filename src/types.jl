# GPMessageTypes for RxGP.jl

const IN_OUT = Union{Real, Array{<:Number}, PointMass, UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily}
const NOISE_Wg = Union{Wishart, PointMass}
const NOISE_w = Union{GammaShapeRate, PointMass}
const TYPE_Norm = Union{UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily}
const TYPE_ContLogPdf = Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}

# The jitter added to K_uu wherever it is factorised: in the meta, in the rules towards θ and in
# the M-step helpers, so that they regularise the same GP.
const KUU_JITTER = 1e-8

# [Type aliases](@id types-reference)

RxGPs defines several type aliases used throughout the message passing rules to constrain method signatures.

| Alias | Union members | Used for |
|:------|:-------------|:---------|
| `IN_OUT` | `Real`, `Array{<:Number}`, `PointMass`, `UnivariateNormalDistributionsFamily`, `MultivariateNormalDistributionsFamily` | Input and output edge types |
| `NOISE_Wg` | `Wishart`, `PointMass` | Matrix noise precision (`UniSGP_dID`'s `Wg`) |
| `NOISE_w` | `GammaShapeRate`, `PointMass` | Scalar noise precision |

These aliases are not exported; the rules use them to declare the inputs they accept.

Wherever ``K_{uu}`` is factorised, in the metas, the rules towards ``\theta`` and the M-step objectives, RxGPs adds the same jitter, `RxGPs.KUU_JITTER = 1e-8`, to its diagonal.

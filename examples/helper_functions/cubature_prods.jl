import BayesBase
# This file is intended to be included standalone when working with non-conjugate products in RxInfer.jl.
# We include it here to showcase our examples that need to combine non-conjugate messages from the univariate
# and univariate_dID nodes with Guassian priors.

function BayesBase.prod(::GenericProd, left::NormalDistributionsFamily, right::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) 
    if left isa MultivariateGaussianDistributionsFamily
        m,c = approximate_meancov(srcubature(), (x) -> exp(right.logpdf(x)), mean(left), cov(left))
        if isnan(m[1])
            return left 
        else
            return MvNormalMeanCovariance(m,c + 1e-6I)
        end
    elseif left isa UnivariateGaussianDistributionsFamily
        m,v = approximate_meancov(ghcubature(21), (x) -> exp(right.logpdf(x)), mean(left), var(left))
        if isnan(m) || isnan(v)
            return left 
        else
            return NormalMeanVariance(m,v + 1e-6)
        end
    end
end
function BayesBase.prod(::GenericProd, left::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, right::NormalDistributionsFamily) 
    if right isa MultivariateGaussianDistributionsFamily
        m,c = approximate_meancov(srcubature(), (x) -> exp(left.logpdf(x)), mean(right), cov(right))
        if isnan(m[1])
            return right
        else
            return MvNormalMeanCovariance(m,c + 1e-6I)
        end
    elseif right isa UnivariateGaussianDistributionsFamily
        m,v = approximate_meancov(ghcubature(21), (x) -> exp(left.logpdf(x)), mean(right), var(right))
        if isnan(m) || isnan(v)
            return right 
        else
            return NormalMeanVariance(m,v + 1e-6)
        end
    end
end
function BayesBase.prod(::GenericProd, prod_of::ProductOf, right::NormalDistributionsFamily) 
    if right isa MultivariateGaussianDistributionsFamily
        m,c = approximate_meancov(srcubature(), (x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)), mean(right), cov(right))
        if isnan(m[1])
            return right
        else
            return MvNormalMeanCovariance(m,c + 1e-6I)
        end
    elseif right isa UnivariateGaussianDistributionsFamily
        m,v = approximate_meancov(ghcubature(21), (x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)), mean(right), var(right))
        if isnan(m) || isnan(v)
            return right 
        else
            return NormalMeanVariance(m,v + 1e-6)
        end
    end
end
function BayesBase.prod(::GenericProd, left::NormalDistributionsFamily, prod_of::ProductOf) 
    if left isa MultivariateGaussianDistributionsFamily
        m,c = approximate_meancov(srcubature(), (x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)), mean(left), cov(left))
        if isnan(m[1])
            return left 
        else
            return MvNormalMeanCovariance(m,c + 1e-6I)
        end
    elseif left isa UnivariateGaussianDistributionsFamily
        m,v = approximate_meancov(ghcubature(21), (x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)), mean(left), var(left))
        if isnan(m) || isnan(v)
            return left 
        else
            return NormalMeanVariance(m,v + 1e-6)
        end
    end
end
function BayesBase.prod(::GenericProd, left::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, right::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) 
    return ProductOf(left, right)
end

BayesBase.default_prod_rule(::Type{<:NormalDistributionsFamily}, ::Type{<:Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}}) = GenericProd()
BayesBase.default_prod_rule(::Type{<:Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}}, ::Type{<:NormalDistributionsFamily}) = GenericProd()
BayesBase.default_prod_rule(::Type{<:Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}}, ::Type{<:Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}}) = GenericProd()
BayesBase.default_prod_rule(::Type{<:NormalDistributionsFamily}, ::Type{<:ProductOf}) = GenericProd()
BayesBase.default_prod_rule(::Type{<:ProductOf}, ::Type{<:NormalDistributionsFamily}) = GenericProd()
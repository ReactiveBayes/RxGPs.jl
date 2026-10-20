using RxGPs
using Documenter

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

# `make docs-serve` sets DOCS_DRAFT: the pages build without running their examples
const draft = get(ENV, "DOCS_DRAFT", "false") == "true"

makedocs(;
    draft = draft,
    warnonly = draft,
    modules=[RxGPs],
    authors="HoangMHNguyen <m.h.n.hoang@tue.nl>, ofSingularMind <a.h.ledbetter@tue.nl>, and contributors",
    sitename="RxGPs.jl",
    checkdocs = :exports,
    format=Documenter.HTML(;
        canonical="https://ReactiveBayes.github.io/RxGPs.jl",
        edit_link="main",
        assets=[
            "assets/header.css",
            "assets/header.js",
            "assets/theme.css",
        ],
    ),
    pages=[
        "Home" => "index.md",
        "User guide" => [
            "Getting started" => "manuals/getting-started.md",
        ],
        "Library" => [
            "Factor nodes"            => "library/nodes.md",
            "Meta objects"            => "library/meta.md",
            "Kernel specification"    => "library/kernels.md",
            "Approximation methods"   => "library/approximation.md",
            "Prediction"              => "library/prediction.md",
            "Cache utilities"         => "library/cache.md",
            "Helper utilities"        => "library/helpers.md",
            "Type aliases"            => "library/types.md",
        ],
        "Examples" => [
            "Notebook examples" => "examples/overview.md",
            "Usage walkthrough"  => "examples/usage.md",
        ],
        "Contributing" => "contributing/guide.md",
    ],
)

deploydocs(;
    repo="github.com/ReactiveBayes/RxGPs.jl",
    devbranch="main",
    push_preview=true,
)

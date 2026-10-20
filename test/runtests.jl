using TestItemRunner, Test, MessagePassingRulesTestUtils, RxGPs, Aqua

# Selection, as in the rule packages: `Pkg.test(test_args = ["node_rule"])` runs one directory,
# `tag:rules` the items with a tag, `name:UniSGP` the items whose name contains the text.
const SELECTED_PATHS = String[]
const SELECTED_NAMES = String[]
const SELECTED_TAGS = Symbol[]

for arg in ARGS
    if startswith(arg, "tag:")
        push!(SELECTED_TAGS, Symbol(arg[(length("tag:") + 1):end]))
    elseif startswith(arg, "name:")
        push!(SELECTED_NAMES, arg[(length("name:") + 1):end])
    else
        push!(SELECTED_PATHS, join(split(arg, ":"), "/"))
    end
end

function is_selected(ti)
    isempty(SELECTED_PATHS) || any(p -> occursin(p, ti.filename), SELECTED_PATHS) || return false
    isempty(SELECTED_NAMES) || any(n -> occursin(n, ti.name), SELECTED_NAMES) || return false
    isempty(SELECTED_TAGS) || any(t -> t in ti.tags, SELECTED_TAGS) || return false
    return true
end

# Whether any item was left out by a selection: the coverage gate means something only after the
# whole suite.
const FILTERED_OUT = Ref(false)
test_item_filter(ti) = is_selected(ti) || (FILTERED_OUT[] = true; false)

@run_package_tests(filter = test_item_filter)

if !FILTERED_OUT[]
    # Every rule and average energy must be selected by some test: a table case or a direct call.
    @testset "rule coverage" begin
        gaps = check_rule_coverage(RxGPs)
        foreach(println, gaps)
        @test isempty(gaps)
    end
    @testset "Aqua" begin
        Aqua.test_all(RxGPs)
    end
end

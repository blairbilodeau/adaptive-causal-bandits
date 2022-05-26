using Distributions
using LinearAlgebra
using Plots

output_path = ".../figures/" # your file path here

include("algorithms-paper.jl")

# helper function to compute the expected regret at the end of a realization of a game
function compute_expected_regret(expected_reward, arm_played)
    T = size(arm_played)[1]
    reward = 0
    for t=1:T
        reward += expected_reward[arm_played[t]]
    end
    best_reward = T * findmax(expected_reward)[1]
    regret = best_reward - reward
    regret
end

# worst-case data when the environment is conditionally benign
function worst_case_unconfounded_data(T,K)
    eps = 0.0005
    Delta = (K*log(T)/T)^(1/2)
    Z0mean = vcat([1-eps],repeat([eps],K-1))
    Z1mean = 1 .- Z0mean
    dist = hcat(Z0mean,Z1mean)

    reward_data = zeros((T,K))
    context_data = Int.(zeros((T,K)))

    for t in 1:T
        for a in 1:K
            Z = rand(Binomial(1,dist[a,2]))
            Y = rand(Binomial(1,1/2 + (1-Z)*Delta))

            reward_data[t,a] = Y
            context_data[t,a] = Z+1
        end
    end

    expected_reward = 1/2 .+ vcat([(1-eps)*Delta],repeat([eps*Delta],K-1))

    reward_data, context_data, expected_reward, dist
end

# worst-case data when the environment is not conditionally benign
function worst_case_confounded_data(T,K)
    p01 = 6/8
    p11 = 7/8
    dist = vcat(repeat([1-p01 p01], Int(K/2)), repeat([1-p11 p11], K-Int(K/2)))

    reward_data = zeros((T,K))
    context_data = Int.(zeros((T,K)))

    m01 = 5.5/6
    m11 = 1/6

    for t in 1:T
        for a in 1:K
            X = Int(a>Int(K/2))
            Z = rand(Binomial(1,dist[a,2]))

            if X == 0 && Z == 0
                Y = rand(Binomial(1,1-m01))
            elseif X == 1 && Z == 0
                Y = rand(Binomial(1,1-m11))
            elseif X == 0 && Z == 1
                Y = rand(Binomial(1,m01))
            else
                Y = rand(Binomial(1,m11))
            end

            reward_data[t,a] = Y
            context_data[t,a] = Z+1
        end
    end

    m0 = m01*p01 + (1-m01)*(1-p01)
    m1 = m11*p11 + (1-m11)*(1-p11)

    expected_reward = vcat(repeat([m0], Int(K/2)), repeat([m1], Int(K/2)))

    reward_data, context_data, expected_reward, dist
end

function run_simulation(Tvals, K, M, data_generator)
    max_T = Tvals[length(Tvals)]
    ucb_expected_regrets = zeros(length(Tvals))
    cucb_expected_regrets = zeros(length(Tvals))
    cucb2_expected_regrets = zeros(length(Tvals))
    hacucb_expected_regrets = zeros(length(Tvals))
    corral_expected_regrets = zeros(length(Tvals))

    for sim in 1:M

        for i in 1:length(Tvals)

            T = Tvals[i]
            reward_data, context_data, expected_reward, dist = data_generator(T,K)

            ucb_arms = UCB(reward_data, 1/T^2)
            cucb_arms = CUCB(reward_data, context_data, dist, 1/T^2)
            cucb2_arms = CUCB2(reward_data, context_data, dist)
            hacucb_arms = HACUCB(reward_data, context_data, dist, 1/T^2)
            corral_arms = Corral(reward_data, context_data, dist, 1/T^2, 1/(40*6*sqrt(2*T)*(log(T))^(3/2)))

            ucb_expected_regrets[i] += compute_expected_regret(expected_reward, ucb_arms)
            cucb_expected_regrets[i] += compute_expected_regret(expected_reward, cucb_arms)
            cucb2_expected_regrets[i] += compute_expected_regret(expected_reward, cucb2_arms)
            hacucb_expected_regrets[i] += compute_expected_regret(expected_reward, hacucb_arms)
            corral_expected_regrets[i] += compute_expected_regret(expected_reward, corral_arms)
        end
    end

    avg_ucb_expected_regrets = ucb_expected_regrets / M
    avg_cucb_expected_regrets = cucb_expected_regrets / M
    avg_cucb2_expected_regrets = cucb2_expected_regrets / M
    avg_hacucb_expected_regrets = hacucb_expected_regrets / M
    avg_corral_expected_regrets = corral_expected_regrets / M

    avg_ucb_expected_regrets, avg_cucb_expected_regrets, avg_cucb2_expected_regrets, avg_hacucb_expected_regrets, avg_corral_expected_regrets

end

# conditionally benign plots
Tvals = 500:250:3000
min_T = Tvals[1]
max_T = Tvals[length(Tvals)]
M = 300
K=20

ucb, cucb, cucb2, hcucb, corral = run_simulation(Tvals, K, M, worst_case_unconfounded_data)

cb_plot = plot(Tvals, ucb, legend=:topleft, label="UCB", title = "Conditionally Benign Environment (|A|=" * string(K) * ", |Z|=2)",
                xlabel="T", ylabel="Regret(T)", xlim=(min_T-10,max_T+100),ylim=(0,710),yticks=([5, 100,200,300,400,500,600,700], ["$i" for i in [0, 100,200,300,400,500,600,700]]),xticks=([525, 1000,1500,2000,2500,3000], ["$i" for i in [500, 1000,1500,2000,2500,3000]]),
                linecolor = :blue, xtickfont=font(12), ytickfont=font(12), legendfont=font(10), xguidefontsize=12, yguidefontsize=12,
                markershape = :rect, markersize = 3, markercolor = :blue)
plot!(Tvals, corral, label="Corral", linestyle=:dashdotdot, linewidth=3, linecolor = :black, markershape = :cross, markercolor = :black, markersize = 3)
plot!(Tvals, cucb, label="C_UCB", linestyle=:dash, linecolor = :darkorange, markershape = :circle, markercolor = :darkorange, markersize = 3)
plot!(Tvals, cucb2, label="C_UCB_2", linestyle=:dashdot, linecolor = :red, markershape = :xcross, markersize = 3, markercolor = :red)
plot!(Tvals, hcucb, label="HAC_UCB*", linestyle=:dot, linewidth=3, linecolor = :green, markershape = :utriangle, markercolor = :green, markersize = 3)
cb_plot
savefig(cb_plot,
    output_path *
    "cb.pdf"
)

# worst case plots
Tvals = 500:500:5000
min_T = Tvals[1]
max_T = Tvals[length(Tvals)]
M = 300
K = 20


worst_ucb, worst_cucb, worst_cucb2, worst_hcucb, worst_corral = run_simulation(Tvals, K, M, worst_case_confounded_data)

worst_plot = plot(Tvals, worst_ucb, legend=:none, label="UCB", title = "Worst Case Environment (|A|=" * string(K) * ", |Z|=2)",
                xlabel="T", ylabel="Regret(T)", xlim=(min_T-10,max_T+100),ylim=(0,1210),yticks=([5,200,400,600,800,1000,1200], ["$i" for i in [0,200,400,600,800,1000,1200]]),xticks=([525, 1000,2000,3000,4000,5000], ["$i" for i in [500, 1000,2000,3000,4000,5000]]),
                linecolor = :blue, xtickfont=font(12), ytickfont=font(12), xguidefontsize=12, yguidefontsize=12,
                markershape = :rect, markersize = 3, markercolor = :blue)
plot!(Tvals, worst_corral, label="Corral", linestyle=:dashdotdot, linewidth=3, linecolor = :black, markershape = :cross, markercolor = :black, markersize = 3)
plot!(Tvals, worst_cucb, label="C_UCB", linestyle=:dash, linecolor = :darkorange, markershape = :circle, markercolor = :darkorange, markersize = 3)
plot!(Tvals, worst_cucb2, label="C_UCB_2", linestyle=:dashdot, linecolor = :red, markershape = :xcross, markersize = 3, markercolor = :red)
plot!(Tvals, worst_hcucb, label="HAC_UCB*", linestyle=:dot, linewidth=3, linecolor = :green, markershape = :utriangle, markercolor = :green, markersize = 3)
worst_plot
savefig(worst_plot,
    output_path *
    "worst.pdf"
)

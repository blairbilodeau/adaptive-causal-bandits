using Roots

# UCB algorithm
# input: TxK matrix
# output: Tx1 vector of actions
function UCB(data, delta)

    # params
    T,K = size(data)

    # Tx1 vector of actions
    arm_played = repeat([0], T)

    # Kx1 vector of empirical arm means
    arm_means = repeat([0.0], K)

    # Kx1 vector of number of pulls
    arm_counts = repeat([0.0], K)

    for t in 1:T
        arm_bounds = sqrt.(0.5*log(1/delta) ./ max.(arm_counts,1))
        ucb_bounds = arm_means + arm_bounds
        arm = findmax(ucb_bounds)[2]
        arm_count = arm_counts[arm]
        reward = data[t,arm]
        arm_means[arm] = (arm_means[arm] * arm_count + reward) / (arm_count + 1)
        arm_counts[arm] = arm_count + 1
        arm_played[t] = arm
    end

    arm_played
end

# Causal UCB algorithm
# input: TxK matrix of rewards, TxK matrix of Z's, KxZ matrix of probs
# output: Tx1 vector of actions
function CUCB(reward_data, context_data, dist, delta)

    # params
    T,K = size(reward_data)
    Z = size(dist)[2]

    # Tx1 vector of actions
    arm_played = repeat([0], T)

    # Zx1 vector of empirical arm means
    context_means = repeat([0.0], Z)

    # Zx1 vector of number of pulls
    context_counts = repeat([0.0], Z)

    for t in 1:T
        context_bounds = sqrt.(0.5*log(1/delta) ./ max.(context_counts,1))
        #context_bounds = sqrt(0.5*log(Z*t^2/2) / t)
        ucb_context_bounds = context_means + context_bounds
        pseudoucb_bounds = dist * ucb_context_bounds
        arm = findmax(pseudoucb_bounds)[2]
        context = context_data[t,arm]
        context_count = context_counts[context]
        reward = reward_data[t,arm]
        context_means[context] = (context_means[context] * context_count + reward) / (context_count + 1)
        context_counts[context] = context_count + 1
        arm_played[t] = arm
    end

    arm_played
end

# Causal UCB 2 algorithm
# input: TxK matrix of rewards, TxK matrix of Z's, KxZ matrix of probs
# output: Tx1 vector of actions
function CUCB2(reward_data, context_data, dist)

    # params
    T,K = size(reward_data)
    Z = size(dist)[2]

    # Tx1 vector of actions
    arm_played = repeat([0], T)

    # Zx1 vector of empirical arm means
    context_means = repeat([0.0], Z)

    # Zx1 vector of number of pulls
    context_counts = repeat([0.0], Z)

    # Zx1 vector of smallest prob of observing each post-context
    min_probs = mapslices(minimum, dist; dims=1)
    arm_zetas = dist ./ min_probs * [1,1]

    for t in 1:T
        arm_bounds = sqrt(0.5*log(Z*t^2) / t) * arm_zetas
        ucb_bounds = dist * context_means + arm_bounds
        arm = findmax(ucb_bounds)[2]
        context = context_data[t,arm]
        context_count = context_counts[context]
        reward = reward_data[t,arm]
        context_means[context] = (context_means[context] * context_count + reward) / (context_count + 1)
        context_counts[context] = context_count + 1
        arm_played[t] = arm
    end

    arm_played
end

# Hypothesis Tested Adaptive Causal UCB algorithm
# input: TxK matrix of rewards, TxK matrix of Z's, KxZ matrix of probs
# output: Tx1 vector of actions
function HACUCB(reward_data, context_data, dist, delta)

    # params
    T,K = size(reward_data)
    Z = size(dist)[2]

    # Tx1 vector of actions
    arm_played = repeat([0], T)

    # Zx1 vector of empirical arm means
    context_means = repeat([0.0], Z)

    # Zx1 vector of number of pulls
    context_counts = repeat([0.0], Z)

    # Kx1 vector of empirical arm means
    arm_means = repeat([0.0], K)

    # Kx1 vector of number of pulls
    arm_counts = repeat([0.0], K)

    # first, explore
    t = 1
    for dummy_t in 1:round(sqrt(T)/K + 1)
        for arm in 1:K
            arm_count = arm_counts[arm]

            context = context_data[t,arm]
            context_count = context_counts[context]

            reward = reward_data[t,arm]

            context_means[context] = (context_means[context] * context_count + reward) / (context_count + 1)
            context_counts[context] = context_count + 1

            arm_means[arm] = (arm_means[arm] * arm_count + reward) / (arm_count + 1)
            arm_counts[arm] = arm_count + 1

            arm_played[t] = arm
            t = t+1
        end
    end

    play_causal = true
    while t <= T
        context_bounds = sqrt.(0.5*log(1/delta) ./ max.(context_counts,1))
        loose_context_bounds = sqrt.(2*log(1/delta) ./ max.(context_counts,1))
        ucb_context_bounds = context_means + context_bounds
        pseudoucb_bounds = dist * ucb_context_bounds

        arm_bounds = sqrt.(0.5*log(1/delta) ./ max.(arm_counts,1))
        loose_arm_bounds = sqrt.(2*log(1/delta) ./ max.(arm_counts,1))
        ucb_bounds = arm_means + arm_bounds

        if play_causal
            play_causal = (sum((ucb_bounds - pseudoucb_bounds) .<= loose_arm_bounds)==K) && (sum((pseudoucb_bounds - ucb_bounds) .<= (dist * loose_context_bounds))==K)
        end
        if play_causal
            arm = findmax(pseudoucb_bounds)[2]
            arm_count = arm_counts[arm]

            context = context_data[t,arm]
            context_count = context_counts[context]

            reward = reward_data[t,arm]

            context_means[context] = (context_means[context] * context_count + reward) / (context_count + 1)
            context_counts[context] = context_count + 1

            arm_means[arm] = (arm_means[arm] * arm_count + reward) / (arm_count + 1)
            arm_counts[arm] = arm_count + 1

            arm_played[t] = arm
        else
            arm = findmax(ucb_bounds)[2]
            arm_count = arm_counts[arm]
            reward = reward_data[t,arm]
            arm_means[arm] = (arm_means[arm] * arm_count + reward) / (arm_count + 1)
            arm_counts[arm] = arm_count + 1
            arm_played[t] = arm
        end

        t = t+1
    end

    arm_played
end

# Corral UCB algorithm (adapted to scale with reward size)
# input: TxK matrix of rewards, TxK matrix of Z's, KxZ matrix of probs
# output: Tx1 vector of actions
function Corral(reward_data, context_data, dist, delta, eta)

    # params
    T,K = size(reward_data)
    Z = size(dist)[2]

    # Tx1 vector of actions
    arm_played = repeat([0], T)

    ## following 2 quantities only updated when c-ucb played
    # Zx1 vector of empirical arm means
    context_means = repeat([0.0], Z)

    # Zx1 vector of number of pulls
    context_counts = repeat([0.0], Z)

    ## following 2 quantities only updated when ucb played
    # Kx1 vector of empirical arm means
    arm_means = repeat([0.0], K)

    # Kx1 vector of number of pulls
    arm_counts = repeat([0.0], K)

    # corral params initialization
    gamma = 1/T # smoothing param
    beta = exp(1/log(T)) # learning rate scaling param
    causal_rho = 4 # importance weight bound for C-UCB
    ucb_rho = 4 # importance weight bound for UCB
    causal_prob = 0.5 # prob of playing C-UCB
    causal_prob_bar = causal_prob # smoothed causal prob

    for t in 1:T

        # randomly pick between C-UCB (bernoulli = 1) and UCB (bernoulli = 0)
        play_causal = convert(Bool, rand(Binomial(1, causal_prob_bar)))

        if play_causal

            # ucb confidence bounds prescribed by Theorem 5.4 of Arora et al. (2021)
            context_bounds = sqrt.(4*causal_rho*log(t) ./ max.(context_counts,1)) .+ ((4/3)*causal_rho*log(t) ./ max.(context_counts,1))
            ucb_context_bounds = context_means + context_bounds
            pseudoucb_bounds = dist * ucb_context_bounds

            arm = findmax(pseudoucb_bounds)[2]

            context = context_data[t,arm]
            context_count = context_counts[context]

            reward = reward_data[t,arm]
            iw_reward = reward / causal_prob

            context_means[context] = (context_means[context] * context_count + iw_reward) / (context_count + 1)
            context_counts[context] = context_count + 1

            # have to turn rewards into losses
            causal_loss = (1-reward) / causal_prob
            ucb_loss = 0

            arm_played[t] = arm

        else

            arm_bounds = sqrt.(4*ucb_rho*log(t) ./ max.(arm_counts,1)) .+ ((4/3)*ucb_rho*log(t) ./ max.(arm_counts,1))
            ucb_bounds = arm_means + arm_bounds

            arm = findmax(ucb_bounds)[2]
            arm_count = arm_counts[arm]

            reward = reward_data[t,arm]
            iw_reward = reward / (1-causal_prob)

            arm_means[arm] = (arm_means[arm] * arm_count + iw_reward) / (arm_count + 1)
            arm_counts[arm] = arm_count + 1

            causal_loss = 0
            ucb_loss = (1-reward) / (1-causal_prob)

            arm_played[t] = arm

        end

        # do log barrier update

        # solve for lambda
        lambda_func = lam -> (1 - 1/(1/causal_prob + eta*(causal_loss - lam)) - 1/(1/(1-causal_prob) + eta*(ucb_loss - lam)))
        lambda = fzero(lambda_func, (causal_loss + ucb_loss) / 2)

        # update log barrier weights and handle rounding issues
        causal_prob = 1/(1/causal_prob + eta*(causal_loss - lambda))
        if causal_prob < 1e-8
            causal_prob = 1e-8
        elseif causal_prob > 1-(1e-8)
            causal_prob = 1-(1e-8)
        end

        causal_prob_bar = (1-gamma)*causal_prob * gamma/2

        # update params
        if 1/causal_prob_bar > causal_rho
            causal_rho = 2/causal_prob_bar
            eta = beta * eta
        end
        if 1/(1-causal_prob_bar) > ucb_rho
            ucb_rho = 2/(1-causal_prob_bar)
            eta = beta * eta
        end

    end

    arm_played
end

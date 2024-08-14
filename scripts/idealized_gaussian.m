% Generates ideal Gaussian arrays and compute variance- and information-based
% metrics of their variability

% NOTE: Probably need to look into the Cholesky decomposition way of
% generating correlated Gaussian arrays. Ref: 
% https://math.stackexchange.com/questions/446093/generate-correlated-normal-random-variables

N=100; % Number of ensemble members
m=10000; % Number of data points in each member
f_proto = randn(1,m);
NORMALIZATION_METHOD = 'probability';

% rho = 0.45;
rho = linspace(0,1,100);
gam_std = zeros(1, length(rho));
gam_shn = zeros(1, length(rho));
for k=1:length(rho)
    % Generate the idealized Gaussian arrays
    MU = zeros(1,N);
    SIGMA = zeros(N, N);
    for i=1:N
        for j=1:N
            if (i~=j)
                SIGMA(i,j) = rho(k);
            else
                SIGMA(i,j) = 1;
            end
        end
    end
    R = mvnrnd(MU, SIGMA, m);
    f = zeros(N,m);
    for n=1:N
%         f(n,:) = rho(k)*f_proto + (1-rho(k))*randn(1,m);
        f(n,:) = R(:,n);
    end
    g = mean(f,1); % "signal" 
    eta = f - g; % "noise" 

    % Flatten f and g for entropy calculation
    f_flat = zeros(1,N*m);
    g_flat = zeros(1,N*m);
    for n=1:N
        f_flat((n-1)*m+1 : n*m) = f(n,:);
        g_flat((n-1)*m+1 : n*m) = g;
    end

    % Discretize into bins and calculate mutual information
    N_BINS=60;
    [P_joint, Xe, Ye] = histcounts2(f_flat, g_flat, N_BINS, 'normalization', NORMALIZATION_METHOD);
    [P_f, ~] = histcounts(f_flat, N_BINS, 'normalization', NORMALIZATION_METHOD);
    [P_g, ~] = histcounts(g_flat, N_BINS, 'normalization', NORMALIZATION_METHOD);
    I_fg = 0;
    for i=1:N_BINS
        for j=1:N_BINS
            Pij = P_joint(i,j); Pi=P_f(i); Pj=P_g(j);
            if Pij>0 && Pi>0 && Pj>0
                I_fg = I_fg + Pij*log2(Pij/ (Pi*Pj));
            end
        end
    end
    H_f = sum(-P_f .* log2(P_f+1e-7));
    gam_shn(k) = 1 - I_fg/H_f;

    % Calculate Total Correlation
    B = accumarray(discretize(f', 30), 1);
    p_marg = calc_marginals(f, 30);
    
    V_g = var(g);
    V_eta_arr = zeros(N,1);
    for i=1:N
        V_eta_arr(i) = mean(eta(i,:).^2);
    end
    V_eta = mean(V_eta_arr);
    gam_std(k) = sqrt(V_eta / (V_eta + V_g));
end

%% Plotting
figure(); 

subplot(1,2,1);
hold on; plot(rho, gam_std); plot(rho, gam_shn); legend(["Variance", "Shannon"])
xlabel('$\rho$','interpreter','latex'); ylabel("$\gamma$",'interpreter','latex');

subplot(1,2,2);
hold on; plot(rho, 1-gam_std); plot(rho, 1-gam_shn); legend(["Variance", "Shannon"])
xlabel('$\rho$','interpreter','latex'); ylabel("$1-\gamma$",'interpreter','latex');

function pxi = marginal(total_joint, i)
    dims = size(total_joint);
    pxi = total_joint;
    for j=1:length(dims)
        if j~=i
            pxi = sum(pxi, j);
        end
    end
    pxi = reshape(pxi, 1, numel(pxi));
end

function marg = calc_marginals(data, num_states)
    s = size(data);
    marg = zeros(s(1), num_states);
    for i=1:num_vars
        [marg(i,:), ~] = histcounts(data(i,:), num_states, 'normalization', 'probability');
    end
end

rho = linspace(0,1,50);

N = 20;
m = 5000;

MI = zeros(1, length(rho));
TC = zeros(1, length(rho));
for k=1:length(rho)
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
    R = mvnrnd(MU, SIGMA, m)';
    [I, H_X] = mutual_info(R, mean(R, 1));
    MI(k) = I/H_X;
    TC(k) = total_info(R);
end
figure(); hold on;
plot(rho, MI);
plot(rho, TC);
legend(["MI", "TC"])

function TC = total_info(X)
    [N,~] = size(X);
    TC = 0;
    H = zeros(1, N);
    for i=1:(N-1)
        [I, H_X, H_x] = mutual_info(X(1:i,:), X(i+1,:));
        if i==1
            H(i) = H_X;
            H(i+1) = H_x;
        else
            H(i+1) = H_x;
        end
        TC = TC + I;
    end
    TC = TC / (sum(H) - max(H));
end

function [I, H_X, H_x] = mutual_info(X, x)
    [N,~] = size(X);
    [~,m] = size(x);

    X_flat = zeros(1,N*m);
    x_flat = zeros(1,N*m);
    for n=1:N
        X_flat((n-1)*m+1 : n*m) = X(n,:);
        x_flat((n-1)*m+1 : n*m) = x;
    end

    % Discretize into bins and calculate mutual information
    N_BINS=60;
    NORMALIZATION_METHOD = 'probability';
    [P_joint, ~, ~] = histcounts2(X_flat, x_flat, N_BINS, 'normalization', NORMALIZATION_METHOD);
    [P_X, ~] = histcounts(X_flat, N_BINS, 'normalization', NORMALIZATION_METHOD);
    [P_x, ~] = histcounts(x_flat, N_BINS, 'normalization', NORMALIZATION_METHOD);
    I = 0;
    for i=1:N_BINS
        for j=1:N_BINS
            Pij = P_joint(i,j); Pi=P_X(i); Pj=P_x(j);
            if Pij>0 && Pi>0 && Pj>0
                I = I + Pij*log2(Pij/ (Pi*Pj));
            end
        end
    end
    H_X = sum(-P_X .* log2(P_X+1e-7));
    H_x = sum(-P_x .* log2(P_x+1e-7));
end
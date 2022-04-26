function prob = HMM(q)

% plot and return the probability
load sp500;

weeks = length(price_move);

% Initializing alpha and beta arrays
alphas = zeros(weeks,2);
betas = alphas;

% Constructing the emission and transition matrices
e_matrix = [q,(1-q); (1-q), q];
t_matrix = [0.8,0.2; 0.2, 0.8];


% Initializing values
if price_move(1) == 1
    alphas(1,1) = e_matrix(1,2) * 0.8;
    alphas(1,2) = e_matrix(2,2) * 0.2;
else
    alphas(1,1) = e_matrix(1,1) * 0.8;
    alphas(1,2) = e_matrix(2,1) * 0.2;
end
betas(end,:) = 1;

% Computing alphas
for w = 2:weeks
    for k = 1:2
       if price_move(w) == 1
            alphas(w,k) = e_matrix(k,2) * sum(alphas((w-1),:) * t_matrix(:,k));
       else
            alphas(w,k) = e_matrix(k,1) * sum(alphas((w-1),:) * t_matrix(:,k));
       end
    end
end

% Computing the probability
prob = alphas(end,2)/sum(alphas(end,:));

% Computing betas
for w = weeks - 1: -1 : 1
    for k = 1 : 2
        for j = 1 : 2
            if price_move(w + 1) == 1
                betas(w,k) = betas(w,k) + (betas(w + 1,j) * e_matrix(j,2) * t_matrix(j,k));
            else
                betas(w,k) = betas(w,k) + (betas(w + 1,j) * e_matrix(j,1) * t_matrix(j,k));
            end
        end
    end
end


% Plotting the probabilities

figure
xx = 1 : weeks;
yy = (alphas(:,2).*betas(:,2))/sum(alphas(end,:));

plot(xx,yy)
title(['Graph for q = ', num2str(q)]);

end

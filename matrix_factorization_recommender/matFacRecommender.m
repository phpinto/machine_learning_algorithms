function [U, V] = matFacRecommender(rateMatrix, lowRank)

    name = 'Ribeiro Pinto, Pedro Henrique';
    disp(name); % Do not delete this line.
    
    % Parameters
    maxIter = 4000; % Choose your own.
    learningRate = 1e-4; % Choose your own.
    regularizer = 5; % Choose your own.
    minDiff = 0.01;
    
    
    % Random initialization:
    [n1, n2] = size(rateMatrix);
    U = rand(n1, lowRank) / lowRank;
    V = rand(n2, lowRank) / lowRank;

    % Gradient Descent:
    i = 0;
    
    while (i <= maxIter)
        U_step = 2 * learningRate * ((((rateMatrix - (U * V')).* (rateMatrix > 0)) * V) + (regularizer * U));
        V_step = 2 * learningRate * ((((rateMatrix - (U * V')).* (rateMatrix > 0))' * U) + (regularizer * V));
        if ((norm([U_step;V_step], 'fro'))^2) < minDiff
           break 
        end
        U = U + U_step;
        V = V  + V_step;
        i = i + 1;
    end
    
end

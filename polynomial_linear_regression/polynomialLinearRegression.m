function theta = polynomialLinearRegression(X, Y, k)

    powers = [0:k,0];
    aug_matrix = zeros(k+1, k+2);

    for row = 1:k+1
        for col = 1:k+2
            if col < k+2
                aug_matrix(row,col) = sum((X.^powers(col)));
            else
                aug_matrix(row,col) = sum((X.^powers(col)).*Y);
            end
        end
        powers = powers + 1;
    end

    theta = rref(aug_matrix);
    theta = theta(:,end);
    
end

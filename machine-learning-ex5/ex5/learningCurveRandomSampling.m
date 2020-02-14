function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda, repeated_random_sampling)
    
m = size(Xval, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for j = 1:repeated_random_sampling
  for i = 1:m
    shuffled = randperm(size(X, 1));
    shuffled_index = shuffled(1:i);
    X_train_sub = X(shuffled_index, :);
    y_train_sub = y(shuffled_index);
    shuffled_2 = randperm(size(Xval, 1));
    shuffled_index_2 = shuffled_2(1:i);
    Xval_sub = X(shuffled_index_2, :);
    yval_sub = y(shuffled_index_2);
    [theta] = trainLinearReg(X_train_sub, y_train_sub, lambda);
    error_train(i) = error_train(i) + linearRegCostFunction(X_train_sub, y_train_sub, theta, 0);
    error_val(i) = error_val(i) + linearRegCostFunction(Xval_sub, yval_sub, theta, 0);
    end
end

error_train = error_train / repeated_random_sampling;
error_val = error_val / repeated_random_sampling;

end
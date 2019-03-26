% test the classifier on  test data
function predicts = logistic_prediction(data, w)
    N = size(data,1);
    predicts = zeros(N,1);
    for i=1:N
        temp = sigmoid([1.0 data(i,:)] * w);
        if temp >= 0.5
            predicts(i) = 1;
        else
            predicts(i) = 0;
        end
    end
end


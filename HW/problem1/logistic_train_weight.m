% train logistic regression classifier
function w = logistic_train_weight(data, labels, w0, totalIter, epsilon)
    learning_rate = 0.003;
    [N,D] = size(data);
    w = w0;
    precost = 0;
    for j = 1:totalIter
        gradient = zeros(D+1,1);
        for k=1:N
            %????(sigmoid([1.0 data(k,:)] * w) - labels(k))*[1.0 data(k,:)]';
            gradient = gradient + (sigmoid([1.0 data(k,:)] * w) - labels(k))*[1.0 data(k,:)]';
        end
        w = w - learning_rate * gradient;
        cost = error_rate(data, labels, w);
        
        %????
        if j > 1 
            if abs(cost-precost) / cost <= epsilon
            %if cost <= epsilon
                break;
            end
        end
        precost = cost;
    end
end
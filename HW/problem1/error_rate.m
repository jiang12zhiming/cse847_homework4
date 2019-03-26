function loss = error_rate(data, labels, w)
    [N,D] = size(data);
    temp = 0.0;
    for m = 1:N
        %????[1.0 data(m,:)]
        predict = sigmoid([1.0 data(m,:)] * w);
        if predict < 0.5
            predict = 0;
        else
            predict = 1;
        end
        temp = temp + abs(predict - labels(m));
    end
    loss = temp / N;
end

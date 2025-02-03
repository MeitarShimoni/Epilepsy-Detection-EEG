function features = extractEEGFeatures(coefficients, fs)
    % Extract features from a given sub-band (D4 or D5)
    SD = std(coefficients); % Standard Deviation
    BP = bandpower(coefficients, fs, [0.5, 4]); % Band Power
    SE = -sum((coefficients.^2) .* log(coefficients.^2 + eps)); % Shannon Entropy
    LE = -sum(log(coefficients.^2 + eps).^2); % Log-Energy Entropy

    features = [BP, SD, SE, LE];
end
